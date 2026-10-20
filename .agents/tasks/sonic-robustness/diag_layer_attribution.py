"""Diagnostic: attribute Module 1 losses to a layer, per modification.

For each query (same edits / track selection as the benchmark script) records:
  retr_top1      pipeline.query ranked the true file #1 (retrieval itself works)
  pipe_accept    pipeline.query returned file_id == true (survives has_primary/has_secondary gate)
  app_deployed   what the app reports today: _classify_match with the stored
                 (missing, Windows) source path -> accepted only if res.confident
  app_localpath  _classify_match if the source file DID exist locally (the branch
                 used for user-ingested tracks such as Chaleya): waveform-correlation test
  rescue_top1    ranked #1 if the disabled pitch-hypothesis rescue were enabled
                 (in-memory copy of query_array; source not modified)
Usage: .venv/bin/python .agents/tasks/sonic-robustness/diag_layer_attribution.py [per_domain]
"""
import importlib.util
import inspect
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SONIC-fixed_1"))
import numpy as np
import librosa

spec = importlib.util.spec_from_file_location("bench", ROOT / "SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

import sonic_db_manager as sdm
from sonic.retrieval import pipeline as P

PER_DOMAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def main():
    src = inspect.getsource(P.query_array).replace("run_pitch_rescue = False", "run_pitch_rescue = True")
    ns = dict(vars(P)); exec(src, ns); qa_rescue = ns["query_array"]
    from sonic.io import load_audio

    db = sdm.SONICDatabaseManager(store_dir=ROOT / "sonic_db_store")
    while db.domain_index is None:
        time.sleep(0.5)
    idx = db.domain_index
    tracks, _ = B.select_tracks(db.manifest, B.DOWNLOADS / "Echoes", PER_DOMAIN, 42)
    agg = defaultdict(lambda: defaultdict(int))
    rng = np.random.default_rng(42)
    tmp = Path(tempfile.mkdtemp(prefix="m1_attr_"))
    try:
        for ti, t in enumerate(tracks):
            y, _ = librosa.load(str(t["local"]), sr=B.SR, mono=True)
            y = y.astype(np.float32)
            fid = t["entry"]["file_id"]
            for c, (kind, param) in B.CONDITIONS.items():
                qp = B.make_edit(kind, param, y, t["local"], tmp, f"{c}_{ti}", rng)
                if qp is None:
                    continue
                res = P.query(str(qp), idx)
                ok_acc = res.file_id == fid
                a = agg[c]
                a["n"] += 1
                a["retr_top1"] += bool(res.ranked_ids and res.ranked_ids[0] == fid)
                a["pipe_accept"] += ok_acc
                a["app_deployed"] += bool(ok_acc and res.confident)
                a["app_localpath"] += bool(ok_acc and sdm.SONICDatabaseManager._classify_match(str(qp), str(t["local"]), res)[0])
                if kind == "pitch":
                    yq = load_audio(str(qp), max_duration=idx.query_max_duration_sec)
                    r2 = qa_rescue(yq, idx)
                    a["rescue_top1"] += bool(r2.ranked_ids and r2.ranked_ids[0] == fid)
                qp.unlink(missing_ok=True)
            print(f"[{ti+1}/{len(tracks)}]", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = {c: dict(v) for c, v in agg.items()}
    json.dump(out, open(ROOT / ".agents/tasks/sonic-robustness/diag_layer_attribution.json", "w"), indent=1)
    print("| condition | n | retrieval top-1 | pipeline accepted | app today (src path missing) | app if src path exists | pitch rescue on: top-1 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for c, v in out.items():
        n = v["n"]
        f = lambda k: f"{v.get(k,0)}/{n}"
        print(f"| {c} | {n} | {f('retr_top1')} | {f('pipe_accept')} | {f('app_deployed')} | {f('app_localpath')} | {f('rescue_top1') if c.startswith('pitch') else '-'} |")


if __name__ == "__main__":
    main()
