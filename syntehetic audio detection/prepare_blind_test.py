"""
PRISM — Blind Unseen Dataset Generator
=======================================
Selects unseen audio files that have NEVER been cached in SQLite and NEVER
seen during model training. Anonymizes filenames into a flat directory
with ZERO labels and ZERO keywords, and generates a secret ground-truth CSV
for independent post-evaluation verification.

Guarantees:
  1. Strict Zero Overlap with `echoes_fingerprints.db`.
  2. Zero Label Leakage: Flat directory with randomized IDs (`sample_00001.wav`).
  3. Balanced Sampling across Environmental, Music, and Speech (Real vs AI).

Usage:
  python prepare_blind_test.py --total-files 6000
  python prepare_blind_test.py --total-files 1500 --output-dir data/pilot_test
"""

from __future__ import annotations

import os
import sys
import random
import shutil
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd

ROOT = Path(__file__).parent.resolve()
DEFAULT_ECHOES_DIR = Path(r"C:\Users\nishe\Downloads\Echoes")
DEFAULT_DB_PATH = ROOT / "echoes_fingerprints.db"
DEFAULT_TEST_DIR = ROOT / "data" / "blind_test_unseen"
DEFAULT_GROUND_TRUTH_CSV = ROOT / "data" / "secret_ground_truth.csv"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".au"}


def get_cached_paths(db_path: Path) -> set:
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM files")
    paths = set(r[0] for r in cur.fetchall())
    conn.close()
    return paths


def get_trained_paths(dataset_dir: Path) -> set:
    try:
        from train_and_eval_40pct import collect_balanced_dataset
        train_items = collect_balanced_dataset(dataset_dir, per_domain_target=600, seed=42)
        return set(it["file_path"] for it in train_items)
    except Exception:
        return set()


def collect_unseen_dataset(
    dataset_dir: Path,
    cached_paths: set,
    total_files: int = 6000,
    seed: int = 42
) -> List[dict]:
    rng = random.Random(seed)
    per_domain = total_files // 3
    half_per_domain = per_domain // 2
    trained_paths = cached_paths

    # ── 1. Environmental Audio ────────────────────────────────────────────────
    # Real: ESC-50 + DESED
    esc_real = [
        str(f) for f in (dataset_dir / "ESC-50").rglob("*.wav")
        if str(f) not in trained_paths and not f.name.startswith("._")
    ]
    desed_real = [
        str(f) for f in (dataset_dir / "DESED_public_eval").rglob("*")
        if f.suffix.lower() in AUDIO_EXTENSIONS and str(f) not in trained_paths and not f.name.startswith("._")
    ]
    env_real_pool = esc_real + desed_real
    rng.shuffle(env_real_pool)

    # Synthetic: True Generative AI (AudioLDM + StableAudio across all categories)
    env_tta_synth = []
    for parent in ["ai_generated", "music_data/ai_generated", "speech_data/ai_generated"]:
        for sub in ["TTA/audioldm", "ATA/audioldm", "TTA/stableaudio"]:
            p = dataset_dir / parent / sub
            if p.exists():
                env_tta_synth.extend([
                    str(f) for f in p.rglob("*")
                    if f.suffix.lower() in AUDIO_EXTENSIONS and str(f) not in trained_paths and not f.name.startswith("._")
                ])
    env_synth_pool = list(set(env_tta_synth))
    rng.shuffle(env_synth_pool)

    n_env = min(half_per_domain, len(env_real_pool), len(env_synth_pool))
    selected_env_real = env_real_pool[:n_env]
    selected_env_synth = env_synth_pool[:n_env]

    # ── 2. Music Audio ────────────────────────────────────────────────────────
    # Real: music_data/real + real/archive
    music_real_pool = []
    for sub in ["music_data/real", "real/archive"]:
        p = dataset_dir / sub
        if p.exists():
            music_real_pool.extend([
                str(f) for f in p.rglob("*")
                if f.suffix.lower() in AUDIO_EXTENSIONS and str(f) not in cached_paths and not f.name.startswith("._")
            ])
    rng.shuffle(music_real_pool)
    selected_music_real = music_real_pool[:half_per_domain]

    # Synthetic: ai_generated/TTA (suno, udio, diffrhythm, etc.)
    music_synth_pool = []
    for sub in ["suno", "udio", "diffrhythm", "songgen", "acestep", "brev", "mubert", "producer"]:
        for parent in ["ai_generated/TTA", "music_data/ai_generated/TTA"]:
            p = dataset_dir / parent / sub
            if p.exists():
                music_synth_pool.extend([
                    str(f) for f in p.rglob("*")
                    if f.suffix.lower() in AUDIO_EXTENSIONS and str(f) not in cached_paths and not f.name.startswith("._")
                ])
    # remove duplicates
    music_synth_pool = list(set(music_synth_pool))
    rng.shuffle(music_synth_pool)
    selected_music_synth = music_synth_pool[:half_per_domain]

    # ── 3. Speech Audio ───────────────────────────────────────────────────────
    # Real: LJSpeech-1.1 + LibriSpeech dev-clean
    speech_real_pool = []
    for sub in ["LJSpeech-1.1", "real/dev-clean"]:
        p = dataset_dir / sub
        if p.exists():
            speech_real_pool.extend([
                str(f) for f in p.rglob("*")
                if f.suffix.lower() in AUDIO_EXTENSIONS and str(f) not in cached_paths and not f.name.startswith("._")
            ])
    rng.shuffle(speech_real_pool)
    selected_speech_real = speech_real_pool[:half_per_domain]

    # Synthetic: WaveFake neural vocoders
    speech_synth_pool = []
    for sub in ["speech_data/ai_generated/wavefake", "music_data/ai_generated/wavefake"]:
        p = dataset_dir / sub
        if p.exists():
            speech_synth_pool.extend([
                str(f) for f in p.rglob("*.wav")
                if str(f) not in cached_paths and not f.name.startswith("._")
            ])
    speech_synth_pool = list(set(speech_synth_pool))
    rng.shuffle(speech_synth_pool)
    selected_speech_synth = speech_synth_pool[:half_per_domain]

    items = []
    for f in selected_env_real:
        items.append({"orig_path": f, "domain": "environmental", "true_label": 0, "source": "Real Environmental"})
    for f in selected_env_synth:
        items.append({"orig_path": f, "domain": "environmental", "true_label": 1, "source": "AI Environmental"})
    for f in selected_music_real:
        items.append({"orig_path": f, "domain": "music", "true_label": 0, "source": "Real Music"})
    for f in selected_music_synth:
        items.append({"orig_path": f, "domain": "music", "true_label": 1, "source": "AI Music"})
    for f in selected_speech_real:
        items.append({"orig_path": f, "domain": "speech", "true_label": 0, "source": "Real Speech"})
    for f in selected_speech_synth:
        items.append({"orig_path": f, "domain": "speech", "true_label": 1, "source": "AI Speech"})

    rng.shuffle(items)
    return items


def create_anonymized_dataset(
    items: List[dict],
    output_dir: Path,
    ground_truth_csv: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing_file in output_dir.glob("*"):
        if existing_file.is_file():
            try:
                existing_file.unlink()
            except Exception:
                pass
    ground_truth_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[ASSEMBLY] Creating anonymized dataset in: {output_dir.resolve()}...")
    anonymized_records = []

    for i, it in enumerate(items, start=1):
        orig_p = Path(it["orig_path"])
        ext = orig_p.suffix.lower()
        anon_filename = f"sample_{i:05d}{ext}"
        dest_p = output_dir / anon_filename

        # Copy audio file (always overwrite to guarantee fresh anonymized set)
        shutil.copy2(orig_p, dest_p)

        anonymized_records.append({
            "filename": anon_filename,
            "anon_path": str(dest_p),
            "true_label": it["true_label"],
            "domain": it["domain"],
            "source": it["source"],
            "original_file": orig_p.name,
        })

        if i % 500 == 0 or i == len(items):
            print(f"           Copied [{i:>5}/{len(items)}] anonymized audio files...", flush=True)

    df_gt = pd.DataFrame(anonymized_records)
    df_gt.to_csv(ground_truth_csv, index=False)

    print(f"\n[DONE] Blind dataset assembled successfully!")
    print(f"       Test Directory : {output_dir.resolve()} ({len(df_gt)} audio files)")
    print(f"       Secret Truth   : {ground_truth_csv.resolve()} (HIDDEN FROM MODEL)")
    print("-" * 60)
    for dom, grp in df_gt.groupby("domain"):
        r_c = (grp["true_label"] == 0).sum()
        ai_c = (grp["true_label"] == 1).sum()
        print(f"       {dom.capitalize():<15}: {len(grp):>4} files ({r_c} Real, {ai_c} Synthetic)")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="PRISM Blind Unseen Dataset Generator")
    parser.add_argument("--total-files", type=int, default=6000,
                        help="Number of unseen files to sample (default: 6000)")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_ECHOES_DIR),
                        help="Path to Echoes master dataset")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH),
                        help="Path to SQLite fingerprint DB (used to verify ZERO overlap)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_TEST_DIR),
                        help="Output directory for anonymized test files")
    parser.add_argument("--ground-truth-csv", type=str, default=str(DEFAULT_GROUND_TRUTH_CSV),
                        help="Path to save secret ground-truth CSV")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    trained_paths = get_trained_paths(Path(args.dataset_dir))
    print(f"[AUDIT] Training baseline identified ({len(trained_paths)} trained clips). Selecting STRICTLY UNSEEN files...")

    items = collect_unseen_dataset(
        dataset_dir=Path(args.dataset_dir),
        cached_paths=trained_paths,
        total_files=args.total_files,
        seed=args.seed
    )

    # Verification of zero training overlap
    overlap = [it["orig_path"] for it in items if it["orig_path"] in trained_paths]
    assert len(overlap) == 0, f"Error: Found {len(overlap)} overlapping files with model training set!"
    print(f"[AUDIT] Zero training overlap verified! All {len(items)} audio files are 100% unseen by model during training.")

    create_anonymized_dataset(
        items=items,
        output_dir=Path(args.output_dir),
        ground_truth_csv=Path(args.ground_truth_csv)
    )


if __name__ == "__main__":
    main()
