"""
Synthetic Audio Training Manager
================================
Provides on-demand model retraining pipeline with active progress tracking,
flexible dataset discovery (standard Echoes hierarchy or custom real/synthetic folders),
feature extraction, ensemble model fitting, domain threshold calibration,
and bundle persistence.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

SYNTH_ROOT = Path(__file__).parent / "syntehetic audio detection"
if str(SYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(SYNTH_ROOT))

from echoes_pipeline import DEFAULT_DB_NAME, AUDIO_EXTENSIONS, FingerprintDB
from train_and_eval_40pct import (
    collect_balanced_dataset,
    extract_and_sync_features,
    build_master_ensemble,
    save_model_bundle,
    DEFAULT_MODEL_PATH,
    SEED,
)


class SyntheticTrainManager:
    """Manages retraining of the PRISM AI Audio Ensemble Model."""

    def __init__(self, synth_dir: Optional[Path] = None):
        self.synth_dir = synth_dir or SYNTH_ROOT
        self.models_dir = self.synth_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "prism_echoes_ensemble.pkl"
        self.db_path = self.synth_dir / DEFAULT_DB_NAME

        self._lock = threading.Lock()
        self.progress: Dict[str, Any] = {
            "active": False,
            "step": 0,
            "total_steps": 5,
            "percent": 0.0,
            "message": "Idle",
            "current_action": "",
            "error": None,
            "metrics": None,
            "elapsed_sec": 0.0,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return model availability and training progress."""
        with self._lock:
            model_exists = self.model_path.exists()
            model_info = {}
            if model_exists:
                try:
                    mtime = self.model_path.stat().st_mtime
                    model_info["last_modified"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                    model_info["size_mb"] = round(self.model_path.stat().st_size / (1024 * 1024), 2)
                except Exception:
                    pass

            return {
                "model_ready": model_exists,
                "model_path": str(self.model_path),
                "model_info": model_info,
                "progress": dict(self.progress),
            }

    def start_training(self, dataset_dir: str | Path, limit: int = 1200, test_size: float = 0.20) -> None:
        """Start model retraining in a background worker thread."""
        if self.progress["active"]:
            raise RuntimeError("Model training is already running.")

        t = threading.Thread(
            target=self._train_worker,
            args=(Path(dataset_dir), limit, test_size),
            daemon=True,
        )
        t.start()

    def _train_worker(self, dataset_dir: Path, limit: int, test_size: float) -> None:
        t_start = time.perf_counter()
        with self._lock:
            self.progress.update({
                "active": True,
                "step": 1,
                "total_steps": 5,
                "percent": 5.0,
                "current_action": "Scanning and preparing dataset...",
                "message": f"Scanning dataset in '{dataset_dir}' (target limit: {limit})...",
                "error": None,
                "metrics": None,
                "elapsed_sec": 0.0,
            })

        try:
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

            # Step 1: Collect Dataset Items (Standard Echoes or Custom)
            db = FingerprintDB(self.db_path)
            cached_paths = set(db.get_cached_map().keys())

            items = self._discover_dataset(dataset_dir, limit=limit, cached_paths=cached_paths)
            if len(items) < 10:
                raise ValueError(
                    f"Found only {len(items)} audio files in '{dataset_dir}'. "
                    "A minimum of 10 labeled audio files (real vs synthetic) is required for training."
                )

            with self._lock:
                self.progress.update({
                    "step": 2,
                    "percent": 25.0,
                    "current_action": "Extracting PRISM uniform acoustic fingerprints...",
                    "message": f"Extracted/syncing fingerprints for {len(items)} audio files...",
                    "elapsed_sec": round(time.perf_counter() - t_start, 1),
                })

            # Step 2: Feature Extraction and SQLite Synchronization
            num_cpus = min(os.cpu_count() or 4, 8)
            enriched = extract_and_sync_features(items, db_path=self.db_path, num_workers=num_cpus)

            # Step 3: Train/Validation Split
            with self._lock:
                self.progress.update({
                    "step": 3,
                    "percent": 50.0,
                    "current_action": "Splitting dataset & scaling features...",
                    "message": "Performing stratified train/validation split...",
                    "elapsed_sec": round(time.perf_counter() - t_start, 1),
                })

            df = pd.DataFrame(enriched)
            df["strat_key"] = df["domain"] + "_" + df["label"].astype(str)

            # Verify stratify feasibility
            vc = df["strat_key"].value_counts()
            stratify_col = df["strat_key"] if (vc.min() >= 2 and len(vc) > 1) else None

            train_df, val_df = train_test_split(
                df, test_size=test_size, stratify=stratify_col, random_state=SEED
            )

            X_train = np.array(train_df["fingerprint"].tolist())
            y_train = train_df["label"].values
            X_val = np.array(val_df["fingerprint"].tolist())
            y_val = val_df["label"].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Step 4: Multi-Tree Ensemble Fitting & Domain Threshold Calibration
            with self._lock:
                self.progress.update({
                    "step": 4,
                    "percent": 70.0,
                    "current_action": "Training master multi-tree ensemble...",
                    "message": "Training HistGB + RF + ExtraTrees ensemble & calibrating thresholds...",
                    "elapsed_sec": round(time.perf_counter() - t_start, 1),
                })

            model = build_master_ensemble(seed=SEED)

            # Calibrate domain thresholds via 5-fold cross-validation
            cv_splits = min(5, len(y_train) // 2) if len(y_train) >= 4 else 2
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
            try:
                cv_probs = cross_val_predict(model, X_train_scaled, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            except Exception:
                # Fallback if splits fail
                model.fit(X_train_scaled, y_train)
                cv_probs = model.predict_proba(X_train_scaled)[:, 1]

            train_df = train_df.copy()
            train_df["cv_prob"] = cv_probs

            domain_thresholds: Dict[str, float] = {}
            for dom in ["environmental", "music", "speech", "general"]:
                sub = train_df[train_df["domain"] == dom]
                if len(sub) > 0 and len(np.unique(sub["label"])) > 1:
                    best_th, best_score = 0.50, -100.0
                    y_dom = sub["label"].values
                    p_dom = sub["cv_prob"].values
                    for th in np.linspace(0.40, 0.70, 61):
                        p_bin = (p_dom >= th).astype(int)
                        tn = np.sum((y_dom == 0) & (p_bin == 0))
                        fp = np.sum((y_dom == 0) & (p_bin == 1))
                        tp = np.sum((y_dom == 1) & (p_bin == 1))
                        fn = np.sum((y_dom == 1) & (p_bin == 0))
                        r_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        s_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        bal_acc = (r_acc + s_acc) / 2.0
                        penalty = max(0.0, 0.90 - r_acc) * 5.0
                        score = bal_acc - penalty
                        if score > best_score:
                            best_score = score
                            best_th = float(th)
                    domain_thresholds[dom] = round(best_th, 3)
                else:
                    domain_thresholds[dom] = 0.50

            # Fit final model on full training set
            model.fit(X_train_scaled, y_train)

            # Step 5: Validation Evaluation & Model Saving
            with self._lock:
                self.progress.update({
                    "step": 5,
                    "percent": 90.0,
                    "current_action": "Validating ensemble & saving model bundle...",
                    "message": "Evaluating validation metrics and writing model bundle to disk...",
                    "elapsed_sec": round(time.perf_counter() - t_start, 1),
                })

            val_probs = model.predict_proba(X_val_scaled)[:, 1]
            val_preds = np.zeros(len(y_val), dtype=int)
            for i, (_, row) in enumerate(val_df.iterrows()):
                th = domain_thresholds.get(row["domain"], 0.50)
                val_preds[i] = 1 if val_probs[i] >= th else 0

            val_acc = float(accuracy_score(y_val, val_preds))
            val_f1 = float(f1_score(y_val, val_preds, zero_division=0))

            domain_accs: Dict[str, float] = {}
            for dom, grp in val_df.groupby("domain"):
                d_idx = [i for i, (_, r) in enumerate(val_df.iterrows()) if r["domain"] == dom]
                if d_idx:
                    d_acc = accuracy_score(y_val[d_idx], val_preds[d_idx])
                    domain_accs[dom] = round(float(d_acc) * 100, 2)

            # Persist bundle
            save_model_bundle(
                scaler=scaler,
                model=model,
                save_model_path=self.model_path,
                domain_thresholds=domain_thresholds,
                seed=SEED,
            )

            t_total = time.perf_counter() - t_start
            metrics = {
                "validation_accuracy_pct": round(val_acc * 100, 2),
                "f1_score": round(val_f1, 4),
                "domain_accuracy_pct": domain_accs,
                "domain_thresholds": domain_thresholds,
                "train_samples": len(train_df),
                "validation_samples": len(val_df),
                "total_samples": len(items),
                "duration_sec": round(t_total, 2),
            }

            with self._lock:
                self.progress.update({
                    "active": False,
                    "percent": 100.0,
                    "step": 5,
                    "current_action": "Training Complete!",
                    "message": f"Retrained successfully! Validation Accuracy: {val_acc*100:.1f}%, F1: {val_f1:.4f} in {t_total:.1f}s",
                    "metrics": metrics,
                    "elapsed_sec": round(t_total, 1),
                })
            print(f"[SYNTH TRAIN] Model retrained successfully in {t_total:.2f}s: {metrics}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            with self._lock:
                self.progress.update({
                    "active": False,
                    "error": str(exc),
                    "message": f"Training failed: {exc}",
                })

    def _discover_dataset(
        self,
        dataset_dir: Path,
        limit: int = 1200,
        cached_paths: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Intelligently collect dataset items from standard Echoes directory or custom hierarchy."""
        # 1. Check if standard Echoes layout exists
        esc = dataset_dir / "ESC-50"
        music = dataset_dir / "music_data"
        lj = dataset_dir / "LJSpeech-1.1"

        if esc.exists() or music.exists() or lj.exists():
            per_domain = math.ceil(limit / 3.0)
            return collect_balanced_dataset(
                dataset_dir, per_domain_target=per_domain, cached_paths=cached_paths, seed=SEED
            )

        # 2. Custom Dataset Discovery
        # Search for real and synthetic files across audio extensions
        all_audios: List[Path] = []
        for p in dataset_dir.rglob("*"):
            if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in AUDIO_EXTENSIONS:
                all_audios.append(p)

        if not all_audios:
            return []

        real_keywords = ["real", "human", "authentic", "original", "clean", "esc", "gtzan", "ljspeech", "libri"]
        synth_keywords = ["synth", "ai", "fake", "generated", "vocoder", "gan", "diffusion", "suno", "udio", "wavefake"]

        items: List[Dict[str, Any]] = []
        real_files: List[Path] = []
        synth_files: List[Path] = []

        for p in all_audios:
            full_str = str(p).lower()
            is_synth = any(k in full_str for k in synth_keywords)
            is_real = any(k in full_str for k in real_keywords)

            if is_synth and not is_real:
                synth_files.append(p)
            elif is_real and not is_synth:
                real_files.append(p)
            else:
                # If ambiguous, check immediate parent folder name
                parent = p.parent.name.lower()
                if "fake" in parent or "synth" in parent or "ai" in parent:
                    synth_files.append(p)
                else:
                    real_files.append(p)

        # Balance classes up to limit
        half_needed = limit // 2
        rng = random.Random(SEED)
        rng.shuffle(real_files)
        rng.shuffle(synth_files)

        sel_real = real_files[:half_needed]
        sel_synth = synth_files[:half_needed]

        def _infer_domain(fpath: Path) -> str:
            s = str(fpath).lower()
            if any(k in s for k in ["speech", "voice", "vocal", "talk", "words"]):
                return "speech"
            if any(k in s for k in ["music", "song", "instrument", "beat", "melody"]):
                return "music"
            if any(k in s for k in ["env", "noise", "nature", "ambient", "sound"]):
                return "environmental"
            return "general"

        for f in sel_real:
            items.append({
                "file_path": str(f.resolve()),
                "label": 0,
                "category": "Real Audio",
                "domain": _infer_domain(f),
                "source": "Custom Dataset (Real)",
                "generator": "Real Audio",
            })

        for f in sel_synth:
            items.append({
                "file_path": str(f.resolve()),
                "label": 1,
                "category": "Synthetic Audio",
                "domain": _infer_domain(f),
                "source": "Custom Dataset (Synthetic)",
                "generator": "AI Synthesized",
            })

        return items
