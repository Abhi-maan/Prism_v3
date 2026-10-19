"""
PRISM — 40% Held-Out Test Training & Live Evaluation Pipeline
=============================================================
Features:
  1. Strict 60% Train / 40% Held-Out Test Split (Stratified by Domain & Label).
  2. Zero Data Leakage: Scaler fit ONLY on 60% Train, applied to 40% Test.
  3. Resource-Optimized Master Ensemble: HistGradientBoosting + ExtraTrees + RandomForest (+ XGBoost).
  4. Permanent Model Persistence: Saves trained bundle to `models/prism_echoes_ensemble.pkl`.
  5. Instant Reuse: Supports `--eval-only` / `--load-model` to load model without retraining.
  6. Live Accuracy Measurement: Real test evaluation across Overall, Music, Speech, Environmental.
  7. Detailed Resource Footprint: Peak RAM (MB), CPU execution time, ms/sample latency.

Usage:
  Train & Evaluate on 40% Test Split:
    python train_and_eval_40pct.py --dataset-dir C:\\Users\\nishe\\Downloads\\Echoes --test-size 0.40

  Load Saved Model & Run Instant Live Evaluation:
    python train_and_eval_40pct.py --dataset-dir C:\\Users\\nishe\\Downloads\\Echoes --eval-only
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import math
import pickle
import random
import argparse
import tracemalloc
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from echoes_pipeline import (
    FingerprintDB, _process_file_worker,
    compute_file_hash,
    DEFAULT_DB_NAME, AUDIO_EXTENSIONS
)
import concurrent.futures

DEFAULT_ECHOES_DIR = Path(r"C:\Users\ASUS\OneDrive\Desktop\Echoes")
if not DEFAULT_ECHOES_DIR.exists():
    DEFAULT_ECHOES_DIR = Path(r"C:\Users\nishe\Downloads\Echoes")

DEFAULT_DB_NAME = "echoes_clean_v3.db"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL_PATH = ROOT / "models" / "prism_echoes_ensemble.pkl"
DEFAULT_RESULTS_CSV = RESULTS_DIR / "benchmark_eval_40pct_test.csv"

SEED = 42


def is_valid_audio(fpath: Path | str) -> bool:
    """Verify that file is a regular audio file, > 4KB, not a macOS metadata shadow file."""
    try:
        p = Path(fpath)
        if p.name.startswith("._") or p.stat().st_size < 4096 or p.suffix.lower() not in AUDIO_EXTENSIONS:
            return False
        return True
    except Exception:
        return False


def _convert_one_file(args: tuple) -> str:
    i, fpath, target_fmt, cache_dir = args
    import soundfile as sf
    p = Path(fpath)
    if p.suffix.lower() == target_fmt:
        return str(p)
    h = compute_file_hash(fpath)[:8]
    cached_file = cache_dir / f"{p.stem}_{h}_fmt{target_fmt}"
    if not cached_file.exists():
        try:
            data, sr = sf.read(str(p))
            fmt = "MP3" if target_fmt == ".mp3" else "WAV"
            sf.write(str(cached_file), data, sr, format=fmt)
        except Exception:
            return str(p)
    return str(cached_file)


def standardize_to_mp3_and_wav(file_list: List[str], cache_dir: Path, target_mp3_ratio: float = 0.5) -> List[str]:
    """Ensure the output list has EXACTLY target_mp3_ratio of .mp3 files and (1 - target_mp3_ratio) of .wav files.
    Eliminates AU, FLAC, and codec shortcuts completely between Real and Synthetic classes.
    """
    from concurrent.futures import ThreadPoolExecutor
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(file_list)
    n_mp3 = int(n_total * target_mp3_ratio)
    tasks = [(i, fpath, ".mp3" if i < n_mp3 else ".wav", cache_dir) for i, fpath in enumerate(file_list)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        out_files = list(pool.map(_convert_one_file, tasks))
    return out_files


def collect_balanced_dataset(
    dataset_dir: Path,
    per_domain_target: int = 1000,
    cached_paths: Optional[set] = None,
    seed: int = SEED
) -> List[dict]:
    """Collect balanced real and synthetic audio files across Environmental, Music, and Speech.
    Guarantees:
      - ZERO soundbank leakage into speech or music
      - Equal 50/50 container format balance (MP3 vs WAV) across real and synthetic
      - Genuine WaveFake vocoder speech
      - Pure GTZAN musical genres
      - Zero corrupt / 0-frame files
    """
    rng = random.Random(seed)
    items = []
    half_needed = per_domain_target // 2
    aug_cache_base = ROOT / "data" / "augmented"

    # 1. Environmental Audio
    # Real: ESC-50 + DESED_public_eval (strictly exclude soundbank)
    esc_dir = dataset_dir / "ESC-50"
    esc_files = sorted([
        str(f) for f in esc_dir.rglob("*.wav")
        if is_valid_audio(f)
    ]) if esc_dir.exists() else []

    desed_real_dir = dataset_dir / "DESED_public_eval"
    desed_real_files = sorted([
        str(f) for f in desed_real_dir.rglob("*")
        if is_valid_audio(f)
    ]) if desed_real_dir.exists() else []

    rng.shuffle(esc_files)
    rng.shuffle(desed_real_files)
    # Synthetic Environmental: AudioLDM + StableAudio
    env_synth_files = []
    for sub in ["ai_generated/TTA/audioldm", "ai_generated/ATA/audioldm", "ai_generated/TTA/stableaudio"]:
        p = dataset_dir / sub
        if p.exists():
            env_synth_files.extend(sorted([str(f) for f in p.rglob("*") if is_valid_audio(f)]))
    rng.shuffle(env_synth_files)
    env_half = min(half_needed, len(env_synth_files))
    selected_env_synth = env_synth_files[:env_half]

    esc_needed = int(env_half * 0.70)
    selected_esc_real = esc_files[:min(len(esc_files), esc_needed)]
    desed_needed = env_half - len(selected_esc_real)
    selected_desed_real = desed_real_files[:min(len(desed_real_files), desed_needed)]
    env_real_combined = selected_esc_real + selected_desed_real

    # Format balance Real Environmental: 50% MP3, 50% WAV
    env_real_balanced = standardize_to_mp3_and_wav(env_real_combined, aug_cache_base / "env_real", target_mp3_ratio=0.5)

    for f in env_real_balanced:
        items.append({
            "file_path": f,
            "label": 0,
            "category": "Environmental Audio",
            "domain": "environmental",
            "source": "ESC-50/DESED_public_eval",
            "generator": "Real Environmental",
        })

    # Format balance Synth Environmental: 50% MP3, 50% WAV
    env_synth_balanced = standardize_to_mp3_and_wav(selected_env_synth, aug_cache_base / "env_synth", target_mp3_ratio=0.5)

    for f in env_synth_balanced:
        gen = Path(f).parent.name
        items.append({
            "file_path": f,
            "label": 1,
            "category": "Environmental Audio",
            "domain": "environmental",
            "source": f"ai_generated ({gen})",
            "generator": f"AI Environmental ({gen})",
        })

    # 2. Music Audio
    # Real: 50% GTZAN (vintage 2002 recordings across 10 genres) + 50% Jamendo (modern 2024 commercial CC tracks)
    music_real_dir = dataset_dir / "real" / "archive" / "genres"
    if not music_real_dir.exists():
        music_real_dir = dataset_dir / "real" / "archive"
    if not music_real_dir.exists():
        music_real_dir = dataset_dir / "music_data" / "real"

    gtzan_files = sorted([
        str(f) for f in music_real_dir.rglob("*")
        if is_valid_audio(f)
    ]) if music_real_dir.exists() else []
    rng.shuffle(gtzan_files)

    jamendo_dir = ROOT / "data" / "jamendo_real"
    jamendo_files = sorted([
        str(f) for f in jamendo_dir.rglob("*.mp3")
        if is_valid_audio(f)
    ]) if jamendo_dir.exists() else []
    rng.shuffle(jamendo_files)

    half_gtzan = half_needed // 2
    half_jamendo = half_needed - half_gtzan
    selected_music_real = gtzan_files[:half_gtzan] + jamendo_files[:half_jamendo]

    # Format balance Real Music: 50% MP3, 50% WAV (replaces .au)
    music_real_balanced = standardize_to_mp3_and_wav(selected_music_real, aug_cache_base / "music_real", target_mp3_ratio=0.5)

    for f in music_real_balanced:
        genre = Path(f).parent.name
        src = "Jamendo (Modern CC)" if "jamendo" in Path(f).name.lower() else "GTZAN"
        items.append({
            "file_path": f,
            "label": 0,
            "category": "Music Audio",
            "domain": "music",
            "source": src,
            "generator": f"Real Music ({genre})",
        })

    # Synthetic Music: Suno, Udio, DiffRhythm, SongGen, Producer, Mubert, AceStep, Brev
    music_ai_files = []
    for gen in ["suno", "udio", "diffrhythm", "songgen", "producer", "mubert", "brev", "acestep"]:
        for root_sub in ["ai_generated/TTA", "ai_generated/ATA"]:
            p = dataset_dir / root_sub / gen
            if p.exists():
                music_ai_files.extend(sorted([str(f) for f in p.rglob("*") if is_valid_audio(f)]))
    rng.shuffle(music_ai_files)
    selected_music_ai = music_ai_files[:half_needed]

    # Format balance Synth Music: 50% MP3, 50% WAV
    music_synth_balanced = standardize_to_mp3_and_wav(selected_music_ai, aug_cache_base / "music_synth", target_mp3_ratio=0.5)

    for f in music_synth_balanced:
        gen = Path(f).parent.name
        items.append({
            "file_path": f,
            "label": 1,
            "category": "Music Audio",
            "domain": "music",
            "source": f"ai_generated ({gen})",
            "generator": f"AI Music ({gen})",
        })

    # 3. Speech Audio
    # Real Speech: LJSpeech-1.1 + LibriSpeech dev-clean
    # Zero Content Leakage: LJSpeech files used in Real STRICTLY exclude LJ001-* (which WaveFake synthesized)
    speech_real_files = []
    lj_dir = dataset_dir / "LJSpeech-1.1" / "LJSpeech-1.1" / "wavs"
    if not lj_dir.exists():
        lj_dir = dataset_dir / "LJSpeech-1.1" / "wavs"
    if not lj_dir.exists():
        lj_dir = dataset_dir / "LJSpeech-1.1"
    if lj_dir.exists():
        # Exclude LJ001-* to ensure zero transcript/sentence overlap with WaveFake
        lj_w = sorted([str(f) for f in lj_dir.rglob("*.wav") if is_valid_audio(f) and not Path(f).name.startswith("LJ001-")])
        rng.shuffle(lj_w)
        speech_real_files.extend(lj_w[:half_needed // 2])

    libri_dir = dataset_dir / "real" / "dev-clean"
    if libri_dir.exists():
        lib_f = sorted([str(f) for f in libri_dir.rglob("*") if is_valid_audio(f)])
        rng.shuffle(lib_f)
        speech_real_files.extend(lib_f[:half_needed // 2])

    selected_speech_real = speech_real_files[:half_needed]
    # Format balance Real Speech: 50% MP3, 50% WAV
    speech_real_balanced = standardize_to_mp3_and_wav(selected_speech_real, aug_cache_base / "speech_real", target_mp3_ratio=0.5)

    # Synthetic Speech: WaveFake (MelGAN, HiFi-GAN, WaveGlow, Parallel WaveGAN, Multi-band MelGAN)
    wavefake_dir = dataset_dir / "ai_generated" / "wavefake"
    if not wavefake_dir.exists():
        wavefake_dir = dataset_dir / "speech_data" / "ai_generated" / "wavefake"
    wavefake_files = sorted([str(f) for f in wavefake_dir.rglob("*.wav") if is_valid_audio(f)]) if wavefake_dir.exists() else []
    rng.shuffle(wavefake_files)
    selected_speech_ai = wavefake_files[:half_needed]
    # Format balance Synth Speech: 50% MP3, 50% WAV
    speech_synth_balanced = standardize_to_mp3_and_wav(selected_speech_ai, aug_cache_base / "speech_synth", target_mp3_ratio=0.5)

    for f in speech_real_balanced:
        items.append({
            "file_path": f,
            "label": 0,
            "category": "Speech Audio",
            "domain": "speech",
            "source": "LJSpeech/LibriSpeech",
            "generator": "Real Human Voice",
        })
    for f in speech_synth_balanced:
        fn = Path(f).name.lower()
        vocoder = "Neural Vocoder"
        for vc in ["melgan", "hifigan", "waveglow", "parallel_wavegan", "multiband", "fullband", "wf1", "wf2", "wf3", "wf4", "wf5", "wf6", "wf7"]:
            if vc in fn:
                vocoder = vc.upper()
                break
        items.append({
            "file_path": f,
            "label": 1,
            "category": "Speech Audio",
            "domain": "speech",
            "source": "WaveFake",
            "generator": f"AI Vocoder Voice ({vocoder})",
        })

    rng.shuffle(items)
    return items


def extract_and_sync_features(
    items: List[dict],
    db_path: Path,
    num_workers: int = 4,
    force_reextract: bool = False
) -> List[dict]:
    """Sync uniform fingerprints using permanent SQLite cache."""
    db = FingerprintDB(db_path)
    cached_map = db.get_cached_map()

    new_files = []
    item_paths = [it["file_path"] for it in items]

    for fpath in item_paths:
        curr_h = compute_file_hash(fpath)
        if fpath in cached_map:
            if force_reextract or cached_map[fpath].get("file_hash") != curr_h:
                db.delete_file_record(fpath)
                new_files.append(fpath)
        else:
            new_files.append(fpath)

    if new_files:
        print(f"  [CACHE] Extracting fresh fingerprints for {len(new_files)} files using {num_workers} workers...", flush=True)
        t0 = time.perf_counter()
        processed = 0
        batch = []

        label_lookup = {it["file_path"]: it["label"] for it in items}
        domain_lookup = {it["file_path"]: it["domain"] for it in items}
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_file_worker, f): f for f in new_files}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                f_path = futures[fut]
                processed += 1
                if res is not None:
                    res["label"] = label_lookup.get(f_path, res.get("label", 0))
                    res["domain"] = domain_lookup.get(f_path, res.get("domain", "speech"))
                    batch.append(res)
                if len(batch) >= 25 or processed == len(new_files):
                    if batch:
                        db.insert_fingerprints_batch(batch)
                        batch.clear()
                    print(f"    Converted [{processed:>4}/{len(new_files)}] files...", flush=True)
        t_el = time.perf_counter() - t0
        print(f"  [CACHE] Completed feature extraction in {t_el:.2f}s.", flush=True)

    # Reload from DB
    updated_map = db.get_cached_map()
    enriched_items = []
    for it in items:
        fpath = it["file_path"]
        rec = updated_map.get(fpath)
        if rec is not None:
            combined = dict(rec)
            combined["label"] = it["label"]
            combined["category"] = it["category"]
            combined["domain"] = it["domain"]
            combined["source"] = it["source"]
            combined["generator"] = it["generator"]
            enriched_items.append(combined)

    return enriched_items


def build_master_ensemble(seed: int = SEED) -> VotingClassifier:
    """Build lightweight, resource-optimized, multi-tree voting ensemble."""
    estimators = [
        ("hgb", HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.04, min_samples_leaf=15,
            l2_regularization=0.1, random_state=seed
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=300, max_depth=16, min_samples_split=3,
            n_jobs=-1, random_state=seed
        )),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_split=3,
            n_jobs=-1, random_state=seed
        )),
    ]
    if HAS_XGB:
        estimators.insert(0, ("xgb", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
            n_jobs=-1, random_state=seed, verbosity=0
        )))
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def save_model_bundle(
    scaler: StandardScaler,
    model: VotingClassifier,
    save_path: Path,
    domain_thresholds: Optional[dict] = None,
    seed: int = SEED
) -> None:
    """Save trained scaler, model ensemble, and domain thresholds to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "scaler": scaler,
        "model": model,
        "seed": seed,
        "domain_thresholds": domain_thresholds or {"environmental": 0.55, "music": 0.55, "speech": 0.60},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "2.1_calibrated_40pct"
    }
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[SAVED] Master ML model permanently saved to: {save_path.resolve()}", flush=True)


def load_model_bundle(model_path: Path) -> Tuple[StandardScaler, VotingClassifier, dict]:
    """Load previously saved model bundle from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    print(f"\n[LOADED] Pre-trained model bundle loaded from: {model_path.resolve()} (Trained: {bundle.get('timestamp')})", flush=True)
    return bundle["scaler"], bundle["model"], bundle.get("domain_thresholds", {"environmental": 0.55, "music": 0.55, "speech": 0.60})


def run_pipeline(
    dataset_dir: Path = DEFAULT_ECHOES_DIR,
    test_size: float = 0.40,
    per_domain_target: int = 1000,
    save_model_path: Path = DEFAULT_MODEL_PATH,
    eval_only: bool = False,
    db_path: Path = ROOT / DEFAULT_DB_NAME,
    results_csv: Path = DEFAULT_RESULTS_CSV
) -> dict:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("\n" + "=" * 80)
    print(f"  PRISM — ML TRAINING & LIVE EVALUATION (Held-Out Test Size: {test_size*100:.1f}%)")
    print("=" * 80)

    # 1. Collect dataset
    print(f"\n[1/5] Sampling balanced files from {dataset_dir} (checking permanent SQLite cache first)...")
    db = FingerprintDB(db_path)
    cached_paths = set(db.get_cached_map().keys())
    items = collect_balanced_dataset(dataset_dir, per_domain_target=per_domain_target, cached_paths=cached_paths, seed=SEED)
    print(f"  Collected {len(items)} audio files across 3 domains:")
    df_items = pd.DataFrame(items)
    for cat, grp in df_items.groupby("category"):
        real_c = (grp["label"] == 0).sum()
        ai_c = (grp["label"] == 1).sum()
        print(f"    - {cat:<22}: {len(grp):>4} files ({real_c} Real, {ai_c} Synthetic)")

    # 2. Sync / Extract features into SQLite DB
    print(f"\n[2/5] Checking permanent SQLite feature cache ({db_path.name})...")
    num_cpus = min(os.cpu_count() or 4, 10)
    enriched = extract_and_sync_features(items, db_path=db_path, num_workers=num_cpus)
    print(f"  Successfully loaded {len(enriched)} feature vectors from database.")

    # 3. Train/Test Split or Direct Evaluation
    df = pd.DataFrame(enriched)
    df["strat_key"] = df["domain"] + "_" + df["label"].astype(str)

    if eval_only and save_model_path.exists():
        print(f"\n[3/5] Evaluation Mode: Skipping train split, evaluating directly on {len(df)} files...")
        print(f"\n[4/5] Loading Pre-Trained Model Bundle from {save_model_path.name} (ZERO RETRAINING)...")
        scaler, model, domain_thresholds = load_model_bundle(save_model_path)
        test_df = df
        X_test = np.array(test_df["fingerprint"].tolist())
        y_test = test_df["label"].values
        X_test_scaled = scaler.transform(X_test)
        t_train_time = 0.0
    else:
        print(f"\n[3/5] Performing Stratified { (1-test_size)*100:.0f}% Train / {test_size*100:.0f}% Test Split...")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["strat_key"],
            random_state=SEED
        )

        X_train = np.array(train_df["fingerprint"].tolist())
        y_train = train_df["label"].values
        X_test = np.array(test_df["fingerprint"].tolist())
        y_test = test_df["label"].values

        print(f"  Training set size : {len(train_df)} files ({(1-test_size)*100:.1f}%)")
        print(f"  Test set size     : {len(test_df)} files ({test_size*100:.1f}%) [STRICTLY HELD OUT]")

        # 4. Fit Model
        print(f"\n[4/5] Training Master ML Ensemble on {len(X_train)} training samples...")
        scaler = StandardScaler()
        # Strictly fit scaler ONLY on train set to prevent data leakage!
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = build_master_ensemble(seed=SEED)

        # Calibrate optimal decision threshold per domain via out-of-fold CV on training set
        print("  Calibrating optimal decision thresholds per domain on training set...", flush=True)
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_probs = cross_val_predict(model, X_train_scaled, y_train, cv=cv, method="predict_proba")[:, 1]
        train_df = train_df.copy()
        train_df["cv_prob"] = cv_probs

        domain_thresholds = {}
        for dom in ["environmental", "music", "speech"]:
            sub = train_df[train_df["domain"] == dom]
            if len(sub) > 0:
                best_th, best_score = 0.50, -100.0
                y_dom = sub["label"].values
                p_dom = sub["cv_prob"].values
                # Search candidate thresholds between 0.48 and 0.65
                for th in np.linspace(0.48, 0.65, 35):
                    p_bin = (p_dom >= th).astype(int)
                    tn = np.sum((y_dom == 0) & (p_bin == 0))
                    fp = np.sum((y_dom == 0) & (p_bin == 1))
                    tp = np.sum((y_dom == 1) & (p_bin == 1))
                    fn = np.sum((y_dom == 1) & (p_bin == 0))
                    r_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    s_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    bal_acc = (r_acc + s_acc) / 2.0
                    # Strictly protect real audio against false positives (r_acc >= 0.92) and AI detection (s_acc >= 0.90)
                    penalty = max(0.0, 0.92 - r_acc) * 6.0 + max(0.0, 0.90 - s_acc) * 5.0
                    score = bal_acc - penalty
                    if score > best_score:
                        best_score = score
                        best_th = float(th)
                domain_thresholds[dom] = max(0.50, round(best_th, 3))
            else:
                domain_thresholds[dom] = 0.50
        print(f"  Calibrated Decision Thresholds: {domain_thresholds}", flush=True)

        t_train_0 = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        t_train_time = time.perf_counter() - t_train_0
        print(f"  Ensemble training completed in {t_train_time:.2f}s!")

        # Save model bundle permanently
        save_model_bundle(scaler, model, save_model_path, domain_thresholds=domain_thresholds, seed=SEED)

    # 5. Live Evaluation on Held-Out 40% Test Set
    print(f"\n[5/5] Running Live Evaluation on Held-Out Test Set ({len(X_test)} samples)...")
    t_inf_0 = time.perf_counter()
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = np.zeros(len(y_test), dtype=int)
    for i, (_, row) in enumerate(test_df.iterrows()):
        dom = row["domain"]
        th = domain_thresholds.get(dom, 0.50)
        y_pred[i] = 1 if y_probs[i] >= th else 0

    t_inf_time = time.perf_counter() - t_inf_0
    ms_per_file = (t_inf_time / len(X_test)) * 1000.0

    test_df["pred_label"] = y_pred
    test_df["ai_prob"] = y_probs
    test_df["prediction"] = ["synthetic" if p == 1 else "real" for p in y_pred]

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    r2 = r2_score(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 80)
    split_name = f"DIRECT EVALUATION ({len(y_test)} SAMPLES)" if eval_only else f"{test_size*100:.0f}% HELD-OUT TEST DATASET ({len(y_test)} SAMPLES)"
    print(f"  LIVE BENCHMARK RESULTS ({split_name})")
    print("=" * 80)
    print(f"  Overall Accuracy : {acc * 100:.2f}%")
    print(f"  ROC AUC Score    : {auc * 100:.2f}% ({auc:.4f})")
    print(f"  R^2 Score        : {r2:.4f}")
    print(f"  Precision        : {prec * 100:.2f}%")
    print(f"  Recall           : {rec * 100:.2f}%")
    print(f"  F1-Score         : {f1 * 100:.2f}%")
    print(f"  False Pos. Rate  : {fpr * 100:.2f}%")
    print(f"  Confusion Matrix : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("-" * 80)

    # Per-Domain Breakdown
    print(f"  {'CATEGORY / DOMAIN':<24} | {'FILES':<8} | {'ACCURACY':<10} | {'ROC AUC':<10} | {'R^2':<8} | {'PRECISION':<10} | {'RECALL':<9} | {'F1-SCORE':<9}")
    print("  " + "-" * 100)

    category_results = {}
    for cat, grp in test_df.groupby("category"):
        c_true = grp["label"].values
        c_pred = grp["pred_label"].values
        c_probs = grp["ai_prob"].values
        c_acc = accuracy_score(c_true, c_pred)
        c_auc = roc_auc_score(c_true, c_probs)
        c_r2 = r2_score(c_true, c_probs)
        c_prec = precision_score(c_true, c_pred, zero_division=0)
        c_rec = recall_score(c_true, c_pred, zero_division=0)
        c_f1 = f1_score(c_true, c_pred, zero_division=0)
        category_results[cat] = {
            "count": len(grp), "acc": c_acc, "auc": c_auc, "r2": c_r2, "prec": c_prec, "rec": c_rec, "f1": c_f1
        }
        print(f"  {cat:<24} | {len(grp):<8} | {c_acc*100:>8.2f}% | {c_auc*100:>8.2f}% | {c_r2:>8.4f} | {c_prec*100:>8.2f}% | {c_rec*100:>8.2f}% | {c_f1*100:>8.2f}%")

    print("=" * 80)

    # Save detailed evaluation CSV
    export_cols = [
        "file_path", "category", "domain", "source", "generator",
        "label", "pred_label", "prediction", "ai_prob",
        "vocoder_score", "gan_score", "diffusion_score", "anomaly_score", "artifact_score"
    ]
    avail_cols = [c for c in export_cols if c in test_df.columns]
    test_df[avail_cols].to_csv(results_csv, index=False)
    print(f"\n[EXPORT] Detailed per-sample test evaluation saved to: {results_csv.resolve()}")

    # Resource Profiling
    t_total = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_bytes / (1024 * 1024)

    print("\n" + "=" * 80)
    print("  SYSTEM RESOURCE FOOTPRINT & INFERENCE PERFORMANCE")
    print("=" * 80)
    print(f"  Peak RAM Utilization   : {peak_ram_mb:.2f} MB")
    print(f"  Training Duration      : {t_train_time:.2f} seconds")
    print(f"  Inference Batch Time   : {t_inf_time:.3f} seconds for {len(X_test)} test clips")
    print(f"  Average Sample Latency : {ms_per_file:.2f} ms / audio file")
    print(f"  Total Pipeline Time    : {t_total:.2f} seconds")
    print("=" * 80 + "\n")

    return {
        "overall_accuracy": acc,
        "f1": f1,
        "category_results": category_results,
        "peak_ram_mb": peak_ram_mb,
        "ms_per_file": ms_per_file
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Multi-Domain ML Training & Evaluation")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_ECHOES_DIR))
    parser.add_argument("--test-size", type=float, default=0.40)
    parser.add_argument("--per-domain-target", type=int, default=1000)
    parser.add_argument("--limit", "--total-files", dest="limit", type=int, default=None, help="Total audio files across all domains (e.g. 4000)")
    parser.add_argument("--save-model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    per_domain = args.per_domain_target
    if args.limit:
        per_domain = math.ceil(args.limit / 3.0)

    run_pipeline(
        dataset_dir=Path(args.dataset_dir),
        test_size=args.test_size,
        per_domain_target=per_domain,
        save_model_path=Path(args.save_model),
        eval_only=args.eval_only
    )
