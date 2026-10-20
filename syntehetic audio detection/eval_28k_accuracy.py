import sys
import os
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "syntehetic audio detection"))

from app import run_synthetic_detection, ALLOWED_AUDIO
from echoes_pipeline import infer_label_from_path

echoes_dir = Path(r"C:\Users\ASUS\OneDrive\Desktop\Echoes")
all_files = []
for root, dirs, files in os.walk(echoes_dir):
    for f in files:
        p = Path(root) / f
        if p.suffix.lower() in ALLOWED_AUDIO:
            all_files.append(p)

print(f"Running detection on {len(all_files)} files...")
t0 = time.perf_counter()
res = run_synthetic_detection(all_files)
t1 = time.perf_counter()

results = res.get("results", [])
print(f"Completed in {t1 - t0:.2f} seconds ({res.get('avg_query_time_ms')} ms avg query time).\n")

y_true = []
y_pred = []
by_folder = {}

for p, r in zip(all_files, results):
    if "error" in r:
        continue
    gt = infer_label_from_path(str(p))
    if gt not in (0, 1):
        continue
    pred = 1 if r.get("prediction") == "synthetic" else 0
    y_true.append(gt)
    y_pred.append(pred)

    parts = str(p).replace("/", "\\").split("Echoes\\")
    folder = parts[-1].split("\\")[0] if len(parts) > 1 else "root"
    if folder not in by_folder:
        by_folder[folder] = {"true": [], "pred": []}
    by_folder[folder]["true"].append(gt)
    by_folder[folder]["pred"].append(pred)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("=" * 70)
print(f"  GLOBAL BENCHMARK METRICS (N = {len(y_true)} verified files)")
print("=" * 70)
print(f"  Overall Accuracy : {acc * 100:.2f}%")
print(f"  Precision        : {prec * 100:.2f}%")
print(f"  Recall           : {rec * 100:.2f}%")
print(f"  F1-Score         : {f1 * 100:.2f}%")
print(f"  Confusion Matrix : TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")
print("\nAccuracy by Dataset Category:")
for f_name, d in sorted(by_folder.items()):
    sub_acc = accuracy_score(d["true"], d["pred"])
    total_sub = len(d["true"])
    real_sub = sum(1 for x in d["true"] if x == 0)
    ai_sub = sum(1 for x in d["true"] if x == 1)
    print(f"  - {f_name:24s}: {sub_acc * 100:6.2f}% ({total_sub} clips | {real_sub} Real, {ai_sub} AI)")
