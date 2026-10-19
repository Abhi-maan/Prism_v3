import pickle
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

b = pickle.load(open('models/prism_echoes_ensemble.pkl', 'rb'))
scaler = b['scaler']
model = b['model']
db = sqlite3.connect('echoes_fingerprints.db')
cur = db.cursor()

# 1. Environmental: ESC-50 / DESED_public_eval (Real) + AudioLDM / StableAudio (AI)
cur.execute('''
    SELECT fp.fingerprint_blob, f.file_path, 0 as label, 'environmental' as domain
    FROM files f JOIN fingerprints fp ON f.id=fp.file_id
    WHERE f.file_path LIKE '%ESC-50%' OR f.file_path LIKE '%DESED_public_eval%'
''')
env_real = cur.fetchall()

cur.execute('''
    SELECT fp.fingerprint_blob, f.file_path, 1 as label, 'environmental' as domain
    FROM files f JOIN fingerprints fp ON f.id=fp.file_id
    WHERE f.file_path LIKE '%audioldm%' OR f.file_path LIKE '%stableaudio%'
''')
env_ai = cur.fetchall()

# 2. Music: real vs TTA synth
cur.execute('''
    SELECT fp.fingerprint_blob, f.file_path, 0 as label, 'music' as domain
    FROM files f JOIN fingerprints fp ON f.id=fp.file_id
    WHERE f.file_path LIKE '%music_data%real%' OR f.file_path LIKE '%real%archive%'
''')
music_real = cur.fetchall()

cur.execute('''
    SELECT fp.fingerprint_blob, f.file_path, 1 as label, 'music' as domain
    FROM files f JOIN fingerprints fp ON f.id=fp.file_id
    WHERE (f.file_path LIKE '%suno%' OR f.file_path LIKE '%udio%' OR f.file_path LIKE '%diffrhythm%' OR f.file_path LIKE '%songgen%')
      AND f.file_path NOT LIKE '%audioldm%'
''')
music_ai = cur.fetchall()

# 3. Speech: LJSpeech / LibriSpeech vs WaveFake
cur.execute('''
    SELECT fp.fingerprint_blob, f.file_path, 0 as label, 'speech' as domain
    FROM files f JOIN fingerprints fp ON f.id=fp.file_id
    WHERE f.file_path LIKE '%LJSpeech%' OR f.file_path LIKE '%dev-clean%'
''')
speech_real = cur.fetchall()

cur.execute('''
    SELECT fp.fingerprint_blob, f.file_path, 1 as label, 'speech' as domain
    FROM files f JOIN fingerprints fp ON f.id=fp.file_id
    WHERE f.file_path LIKE '%wavefake%'
''')
speech_ai = cur.fetchall()

print(f"Env: Real={len(env_real)}, AI={len(env_ai)}")
print(f"Music: Real={len(music_real)}, AI={len(music_ai)}")
print(f"Speech: Real={len(speech_real)}, AI={len(speech_ai)}")

# Take 500 of each
import random
rng = random.Random(42)

def sample_500(lst):
    l = list(lst)
    rng.shuffle(l)
    return l[:500]

selected = (
    sample_500(env_real) + sample_500(env_ai) +
    sample_500(music_real) + sample_500(music_ai) +
    sample_500(speech_real) + sample_500(speech_ai)
)

print(f"\nTotal selected: {len(selected)}")

X = np.array([np.frombuffer(r[0], dtype=np.float32) for r in selected])
y_true = np.array([r[2] for r in selected])
domains = np.array([r[3] for r in selected])

X_scaled = scaler.transform(X)
probs = model.predict_proba(X_scaled)[:, 1]

# Calibrated domain thresholds
th_map = {'environmental': 0.62, 'music': 0.35, 'speech': 0.65}
y_pred = np.array([1 if probs[i] >= th_map[domains[i]] else 0 for i in range(len(selected))])

print(f"Overall Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"Overall Precision: {precision_score(y_true, y_pred)*100:.2f}%")
print(f"Overall Recall: {recall_score(y_true, y_pred)*100:.2f}%")
print(f"Overall F1: {f1_score(y_true, y_pred)*100:.2f}%")

print("\nPer-Domain Breakdown:")
for d in ['environmental', 'music', 'speech']:
    m = (domains == d)
    acc = accuracy_score(y_true[m], y_pred[m])
    prec = precision_score(y_true[m], y_pred[m])
    rec = recall_score(y_true[m], y_pred[m])
    print(f"  {d.capitalize():<15}: Accuracy={acc*100:6.2f}%, Precision={prec*100:6.2f}%, Recall={rec*100:6.2f}%")
