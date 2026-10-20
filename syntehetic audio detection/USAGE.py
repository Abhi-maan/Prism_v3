"""
PRISM — Spectral Artifact Extractor
Usage examples
==============

Single file
-----------
from prism.sonic.tier_a import analyze_file

result = analyze_file("path/to/audio.wav")
print(result)
# {
#   "file_path":       "...",
#   "domain":          "speech",
#   "artifact_score":  0.7231,
#   "vocoder_score":   0.8100,
#   "gan_score":       0.5500,
#   "diffusion_score": 0.4900,
#   "sub_scores":      {...},
#   "flags":           ["vocoder_artifact"],
#   "prediction":      "synthetic",
# }


Batch dataset
-------------
from prism.sonic.tier_a import analyze_dataset

df = analyze_dataset(
    folder_path="data/audio_samples/",
    output_csv="results/tier_a_results.csv",
    ground_truth_csv="data/labels.csv",  # optional — enables threshold sweep
)


Threshold sweep output example
-------------------------------
Threshold sweep results
--------------------------------------------------
 Threshold  Precision   Recall       F1   Accuracy
--------------------------------------------------
      0.40     0.8321   0.9012   0.8652     0.8410
      0.45     0.8612   0.8741   0.8676     0.8550
      0.50     0.8900   0.8401   0.8643     0.8620
      0.55     0.9150   0.7912   0.8487     0.8590
      0.60     0.9350   0.7401   0.8262     0.8530
--------------------------------------------------



Ground-truth CSV format
-----------------------
filename,label
clip_001.wav,0
clip_002.wav,1
...
  label: 0 = real, 1 = synthetic


ECHOES & PRISM — Production Commands & Blind Testing
--------------------------------------------------------
1. DEFAULT COMMAND: Direct AI Audio Filtering (ZERO Retraining):
   # Directly filter AI audio from any folder or mentor dataset:
   python filter_audio.py --input-dir path/to/audios

   # Filter and isolate detected AI audio into a quarantine directory:
   python filter_audio.py --input-dir path/to/audios --quarantine-dir path/to/ai_quarantine

   # Custom output report:
   python filter_audio.py --input-dir path/to/audios --output-csv results/ai_filtered_report.csv

2. EXPLICIT COMMAND: Train Master Model (Only when requested):
   # Retrain the master model ensemble on 1800 clips:
   python train_model.py --limit 1800

   # Retrain on custom dataset directory:
   python train_model.py --dataset-dir path/to/dataset --limit 3000

3. BLIND UNSEEN TESTING (Zero Label Leakage Verification):
   # Assemble 600 or 6000 unseen blind test clips (flat directory, secret labels):
   python prepare_blind_test.py --total-files 6000

   # Filter blind audio with zero retraining:
   python filter_audio.py --input-dir data/blind_test_unseen --output-csv results/blind_preds.csv

   # Independently audit accuracy against secret ground-truth (>90% target):
   python verify_blind_test.py --preds results/blind_preds.csv --ground-truth data/secret_ground_truth.csv

4. TEST STORED FINGERPRINTS (Direct Database Benchmark):
   # Test 3000 stored fingerprinted files directly from SQLite (echoes_fingerprints.db):
   python test_stored_fingerprints.py --limit 3000

   # Balanced sampling with custom CSV output:
   python test_stored_fingerprints.py --limit 3000 --sampling balanced --output-csv results/fingerprints_3000.csv
"""
