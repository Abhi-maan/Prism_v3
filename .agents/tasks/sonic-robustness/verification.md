# Module 1 fix: verification log

All commands ran from `/Users/user84/Downloads/prism_v2` with `PY=.venv/bin/python` (Python 3.11) and `< /dev/null`. No packages were installed, nothing was committed, and `/Users/user84/Downloads/Prism-main 2` was not touched (no file there is newer than `plan.md`).

## Iteration 2: review finding B1 (FP rose from 0/118 to 1/118)
**Change.** New `RetrievalParams.min_contrast_long = 0.04` in `SONIC-fixed_1/sonic/retrieval/pipeline.py`. `_passes_abs` now requires a positive misalignment contrast at every aligned-window count:
- 0.08 below 10 windows (unchanged)
- 0.04 at 10 or more windows (new)

The calibration harness mirror (`calib_evidence.py` `_ok`, `OPEN`) was updated to match. The value was chosen on dev seed 7 plus the 784 calibration negatives only; see `experiments.md` E8. The held-out FP is what motivated the rule, and this is disclosed there.

| command | result |
|---|---|
| replay of the locked rule + floor ∈ {0, 0.01 … 0.05} on `experiments/tempo6.json` and the previous `experiments/final.json` (dev seed 7 + 784 calib negatives; throw-away script, deleted) | 0.00–0.04: dev 91.42 %, 0 FP; 0.05: 91.03 %. Adopted 0.04 |
| `$PY -m unittest discover -s SONIC-fixed_1/tests -p 'test_*.py'` | **24 tests OK** (3.2 s). The 2 new tests are in `SteadyContrastTest`: a detuned drone vs a different DB drone (11 windows, contrast ≈ 0) is rejected, and is accepted with the floor off; the held-out FP evidence is rejected by `_passes_abs` |
| `$PY -m py_compile SONIC-fixed_1/sonic/retrieval/pipeline.py SONIC-fixed_1/tests/test_pipeline_verify.py .agents/tasks/sonic-robustness/calib_evidence.py` | OK |
| `$PY .agents/tasks/sonic-robustness/calib_evidence.py --configs final --per-domain 12 --negatives 500 < /dev/null` (log `calib_final_e8.log`) | **dev edited 91.4 %, 0/784 calibration FPs**. All 1,835 decisions are identical to the iteration-1 dump |
| `$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --out-json .agents/tasks/sonic-robustness/benchmark_after.json --out-md .agents/tasks/sonic-robustness/benchmark_after.md < /dev/null` (held-out seed 42, full, log `benchmark_after.log`) | **pooled edited 2475/2809 = 88.1 %**, **FP 0/118**, latency mean 73.2 / p95 223.6 ms. The iteration-1 md is kept as `benchmark_after_v1_iter1.md` |
| `$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --via-http --out-json .agents/tasks/sonic-robustness/benchmark_after_http_quick.json --out-md .agents/tasks/sonic-robustness/benchmark_after_http_quick.md` | POST `/api/check` via the Flask test client: **90.3 %** pooled edited (618/684), **0/28 FP**, mean 77.3 / p95 229 ms |
| `$PY .agents/tasks/sonic-robustness/repro_web_path.py` | original → **Duplicate**; pitch +2 st → **Edited Audio**; speed +5 % → **Edited Audio**. All three return "Chaleya Jawan 320 Kbps.mp3", and HTTP single, HTTP 2-file batch and the direct call all agree (199 / 97 ms) |
| `uploads/latest_check_result.json` backed up to `/tmp/m1_latest_backup2.json` before the HTTP runs, then restored | `cmp` identical |
| `shasum sonic_db_store/manifest.json sonic_db_store/unified_archive.npz` | identical to `preflight.txt`; the store was not written |

### Held-out benchmark: BEFORE vs iteration 1 vs final (seed 42, full, `app.run_duplicate_detection`)
| Condition | n | BEFORE top-1 | AFTER iter 1 | AFTER (final) | Δ vs BEFORE | Δ vs iter 1 | wrong-match | mean / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| duplicate | 103 | 57.3% | 99.0% | 97.1% | +39.8 | -1.9 | 0 | 24.8 / 32.4 |
| pitch_-4st | 103 | 0.0% | 84.5% | 82.5% | +82.5 | -1.9 | 0 | 196.2 / 250.2 |
| pitch_-2st | 103 | 0.0% | 87.4% | 84.5% | +84.5 | -2.9 | 0 | 140.7 / 232.7 |
| pitch_-1st | 103 | 0.0% | 85.4% | 83.5% | +83.5 | -1.9 | 0 | 110.5 / 231.7 |
| pitch_+1st | 103 | 0.0% | 85.4% | 83.5% | +83.5 | -1.9 | 0 | 88.4 / 223.5 |
| pitch_+2st | 103 | 0.0% | 88.3% | 85.4% | +85.4 | -2.9 | 0 | 126.8 / 221.0 |
| pitch_+4st | 103 | 0.0% | 83.5% | 81.6% | +81.6 | -1.9 | 0 | 180.3 / 231.1 |
| speed_-20pct | 103 | 0.0% | 85.4% | 83.5% | +83.5 | -1.9 | 0 | 134.5 / 230.5 |
| speed_-10pct | 103 | 0.0% | 86.4% | 85.4% | +85.4 | -1.0 | 0 | 81.2 / 225.8 |
| speed_-5pct | 103 | 0.0% | 86.4% | 85.4% | +85.4 | -1.0 | 0 | 76.6 / 231.5 |
| speed_+5pct | 103 | 0.0% | 90.3% | 86.4% | +86.4 | -3.9 | 0 | 71.8 / 222.4 |
| speed_+10pct | 103 | 0.0% | 87.4% | 85.4% | +85.4 | -1.9 | 0 | 82.1 / 209.8 |
| speed_+20pct | 103 | 0.0% | 81.6% | 80.6% | +80.6 | -1.0 | 0 | 121.5 / 209.0 |
| mp3_64kbps | 103 | 49.5% | 99.0% | 97.1% | +47.6 | -1.9 | 0 | 26.5 / 35.3 |
| mp3_128kbps | 103 | 57.3% | 98.1% | 96.1% | +38.8 | -1.9 | 1 | 24.7 / 32.1 |
| noise_snr10db | 103 | 25.2% | 42.7% | 41.7% | +16.5 | -1.0 | 0 | 128.4 / 239.9 |
| noise_snr20db | 103 | 50.5% | 78.6% | 77.7% | +27.2 | -1.0 | 1 | 57.1 / 224.0 |
| gain_-6db | 103 | 56.3% | 99.0% | 97.1% | +40.8 | -1.9 | 0 | 22.2 / 31.1 |
| gain_-3db | 103 | 56.3% | 99.0% | 97.1% | +40.8 | -1.9 | 0 | 21.8 / 30.4 |
| gain_+3db | 103 | 56.3% | 99.0% | 97.1% | +40.8 | -1.9 | 0 | 21.8 / 30.9 |
| gain_+6db | 103 | 56.3% | 99.0% | 97.1% | +40.8 | -1.9 | 0 | 21.7 / 30.7 |
| highpass_150hz | 103 | 38.8% | 98.1% | 95.1% | +56.3 | -2.9 | 1 | 26.1 / 36.6 |
| highpass_200hz | 103 | 34.0% | 99.0% | 96.1% | +62.1 | -2.9 | 0 | 25.9 / 29.8 |
| trim_start_3s | 92 | 0.0% | 85.9% | 85.9% | +85.9 | +0.0 | 1 | 22.9 / 55.6 |
| trim_start_10s | 35 | 5.7% | 97.1% | 97.1% | +91.4 | +0.0 | 1 | 24.7 / 29.7 |
| trim_end_3s | 92 | 37.0% | 95.7% | 95.7% | +58.7 | +0.0 | 0 | 17.2 / 35.2 |
| trim_end_10s | 35 | 40.0% | 100.0% | 100.0% | +60.0 | +0.0 | 0 | 22.2 / 34.3 |
| timeshift_cut_5s | 48 | 2.1% | 100.0% | 100.0% | +97.9 | +0.0 | 0 | 23.0 / 28.4 |
| timeshift_cut_20s | 35 | 0.0% | 100.0% | 97.1% | +97.1 | -2.9 | 0 | 30.2 / 32.7 |
| timeshift_pad_5s | 103 | 0.0% | 99.0% | 97.1% | +97.1 | -1.9 | 0 | 24.3 / 34.7 |
| timeshift_pad_20s | 103 | 0.0% | 99.0% | 97.1% | +97.1 | -1.9 | 0 | 24.9 / 34.4 |
| **pooled edited (30 conditions)** | 2809 | **19.4%** | 89.9% | **88.1%** | +68.7 | -1.8 | 5 | 70.0 / 220.9 |

| Negatives (not in DB) | n | BEFORE FP | AFTER iter 1 FP | AFTER (final) FP |
|---|---:|---:|---:|---:|
| librispeech_test_clean | 40 | 0 | 0 | 0 |
| local_music_mp3_segment | 18 | 0 | 1 | 0 |
| local_speech_mp3_segment | 20 | 0 | 0 | 0 |
| esc50_time_reversed | 40 | 0 | 0 | 0 |
| **all** | 118 | 0 | 1 | **0** |

**Latency (final):**
- all 3,030 queries: mean 73.2 ms, p95 223.6 ms
- positives: 68.4 / 220.7 ms
- negatives: 191.5 / 249.3 ms
- BEFORE: all 11.9 / 15.5 ms

**Per domain (pooled edited, final):** music 96.0 %, speech 92.5 %, environment 74.5 % (iteration 1: 80.1 %), user song 28/30.

**Labels on correct rows:**
- duplicate → Duplicate: 99/100
- pitch/speed → Edited Audio: 1038/1038
- trims/shifts → Edited Audio: 518/518

**Wrong-file matches:** 13 → 5. All 5 are GTZAN `jazz.00046` → `jazz.00072`, which are the same audio. The genuine error from iteration 1 (`5-215445-A-47` → `Windows_of_the_Skull…`) no longer occurs.

**What the floor cost:** 52 rows across 4 ESC-50 steady-sound tracks plus 1 GTZAN row.
- The tracks are `3-117504-A-16` (wind), `5-215445-A-47` (airplane), and `4-172733-A-36` / `5-182007-A-36` (vacuum cleaner).
- That includes their unmodified `duplicate` rows: duplicate drops 99.0 → 97.1 %.
- Their correct alignments had contrast −0.001 to 0.038 over 10–13 windows. Steady sounds match about equally well at any offset, which is exactly the signature of the rejected FP.
- None of these rows was correct BEFORE, so there is **no regression vs BEFORE** on any (condition, track): 0 rows went from correct to wrong.
- Not adopted: exempting near-exact fine matches (fine BER ≤ `strong_fine_ber` 0.20) from the floor. On the iteration-1 evidence this would recover all 52 rows (max fine 0.199) and still reject the FP (fine 0.235). But dev data can't distinguish it from the adopted rule, so choosing it would mean tuning on the held-out set, and its margin (0.199 vs 0.20) is too thin to trust.

## Iteration 1 log (superseded numbers in §5–7 are kept for the record)
## 1. Preflight

| command | result |
|---|---|
| copy changed sources → `.agents/tasks/sonic-robustness/backup/src/**`; `git status --short` + `shasum` of manifest/archive → `preflight.txt` | done |
| `$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --seed 7 --out-json .agents/tasks/sonic-robustness/dev_before_seed7.json` | dev BEFORE: pooled edited **21.4 %** (147/688), 0/28 FP, attach 248 s |

## 2. Unit tests and compile check

| command | result |
|---|---|
| `$PY -m py_compile app.py sonic_db_manager.py SONIC-fixed_1/sonic/{io,config}.py SONIC-fixed_1/sonic/retrieval/pipeline.py SONIC-fixed_1/sonic/fingerprint/binary_embed.py SONIC-fixed_1/sonic/index/subfingerprint_index.py SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py .agents/tasks/sonic-robustness/calib_evidence.py` | OK |
| `$PY -m unittest discover -s SONIC-fixed_1/tests -p 'test_*.py' -v` | **22 tests OK** (2.9 s) |

The suite covers:

- **test_subfp_index:** contiguity, aligned distances, `vote()` unchanged.
- **test_hypotheses:** pitch/tempo hypotheses reduce Hamming distance; the vectorised codes are bit-identical to the index codes.
- **test_query_window:** 5 s / 20 s / −60 dB pads skipped; t=0 content untouched; −30 dB soft onset not cut.
- **test_pipeline_verify:** a copy, +2 st, ×1.1 tempo, 5 s pad and 3 s trim are each accepted as the right file with the right label. Unrelated and reversed audio are rejected. Legacy mode is reproducible.
- **test_classify_match:** same verdict whether the source path exists or not; a candidate without its own evidence is rejected; the audit loop flags only verified candidates; the length check works.
- **test_db_attach:** archive + delta attach equals a full rebuild, and the store is left untouched.

The repo had no pre-existing tests to run.

## 3. DB attach (startup)

| measurement | BEFORE | AFTER |
|---|---:|---:|
| standalone `SONICDatabaseManager()` | ~75 s (full `.npz` rebuild) | 68–70 s (archive + 1 delta `.npz`) |
| inside the app (`import app`; benchmark "DB attached") | 248 s | **66 s** |

Two changes produced this:

- The archive is now reused, with only the one missing entry (fid 28408) appended.
- In-app, `synthetic_train_manager` → `train_and_eval_40pct` sets `OMP_NUM_THREADS=1` at import, so faiss built the HNSW graphs single-threaded: 231 s in-app even with the archive path. `_rebuild_index` now raises faiss' OpenMP thread count only while it builds, then restores it.

Files: 28,409.

## 4. Calibration (dev seed 7 only)

All runs use `$PY .agents/tasks/sonic-robustness/calib_evidence.py`; details and tables are in `experiments.md`.

| command | result |
|---|---|
| `--configs baseline,fine_identity --per-domain 12 --negatives 500` (first version) | no 0-FP setting; root causes were silent windows and time-reversed copies (E0) |
| `--configs baseline,win8,ef128 ...`, `--configs tempo6,fine_identity,vhop4 ...`, each followed by `--sweep experiments/<c>.json` | E2/E3/E4 tables |
| `--configs final --per-domain 12 --negatives 500` (locked thresholds, real code path) | 1st run: dev 91.8 %, 2/784 FP. After the short-alignment tier (E7): **dev 91.4 %, 0/784 calibration FPs**, latency mean 107 ms |
| `$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --seed 7 --out-json .agents/tasks/sonic-robustness/dev_after_seed7_quick.json` | dev AFTER (app path): 93.9 % pooled edited, 1/28 FP (reversed helicopter clip, which led to `reverse_margin` 0.01, E5b) |

## 5. Held-out benchmark (seed 42, full), iteration-1 code (superseded by Iteration 2 above)

```
$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py \
  --out-json .agents/tasks/sonic-robustness/benchmark_after.json \
  --out-md   .agents/tasks/sonic-robustness/benchmark_after.md < /dev/null
```

This is the same command, seed and track/negative selection as BEFORE: 103 tracks, 2,912 positive queries and 118 negatives, through `app.run_duplicate_detection`. Log: `benchmark_after.log`. A first full run before the faiss-threads change gave the same 2525/2809 and 1/118.

| Condition | n | BEFORE top-1 | AFTER top-1 | Δ pts | AFTER wrong-match | AFTER mean / p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| duplicate | 103 | 57.3% | 99.0% | +41.8 | 0 | 20.6 / 31.5 |
| pitch_-4st | 103 | 0.0% | 84.5% | +84.5 | 1 | 193.0 / 247.9 |
| pitch_-2st | 103 | 0.0% | 87.4% | +87.4 | 0 | 137.0 / 227.8 |
| pitch_-1st | 103 | 0.0% | 85.4% | +85.4 | 1 | 104.1 / 230.2 |
| pitch_+1st | 103 | 0.0% | 85.4% | +85.4 | 0 | 83.3 / 229.6 |
| pitch_+2st | 103 | 0.0% | 88.3% | +88.3 | 0 | 123.0 / 217.8 |
| pitch_+4st | 103 | 0.0% | 83.5% | +83.5 | 1 | 177.1 / 229.7 |
| speed_-20pct | 103 | 0.0% | 85.4% | +85.4 | 1 | 126.1 / 227.1 |
| speed_-10pct | 103 | 0.0% | 86.4% | +86.4 | 1 | 73.0 / 225.2 |
| speed_-5pct | 103 | 0.0% | 86.4% | +86.4 | 2 | 68.9 / 224.1 |
| speed_+5pct | 103 | 0.0% | 90.3% | +90.3 | 0 | 63.7 / 220.5 |
| speed_+10pct | 103 | 0.0% | 87.4% | +87.4 | 1 | 76.3 / 207.2 |
| speed_+20pct | 103 | 0.0% | 81.5% | +81.5 | 0 | 116.8 / 206.0 |
| mp3_64kbps | 103 | 49.5% | 99.0% | +49.5 | 0 | 22.2 / 33.0 |
| mp3_128kbps | 103 | 57.3% | 98.1% | +40.8 | 1 | 20.5 / 29.4 |
| noise_snr10db | 103 | 25.2% | 42.7% | +17.5 | 0 | 124.9 / 233.5 |
| noise_snr20db | 103 | 50.5% | 78.6% | +28.1 | 1 | 53.7 / 212.4 |
| gain_-6db | 103 | 56.3% | 99.0% | +42.7 | 0 | 18.0 / 29.2 |
| gain_-3db | 103 | 56.3% | 99.0% | +42.7 | 0 | 17.5 / 28.5 |
| gain_+3db | 103 | 56.3% | 99.0% | +42.7 | 0 | 17.5 / 28.7 |
| gain_+6db | 103 | 56.3% | 99.0% | +42.7 | 0 | 17.5 / 29.0 |
| highpass_150hz | 103 | 38.8% | 98.1% | +59.2 | 1 | 20.2 / 28.6 |
| highpass_200hz | 103 | 34.0% | 99.0% | +65.0 | 0 | 20.1 / 26.5 |
| trim_start_3s | 92 | 0.0% | 85.9% | +85.9 | 1 | 22.5 / 55.5 |
| trim_start_10s | 35 | 5.7% | 97.1% | +91.4 | 1 | 23.8 / 29.3 |
| trim_end_3s | 92 | 37.0% | 95.7% | +58.7 | 0 | 16.7 / 34.8 |
| trim_end_10s | 35 | 40.0% | 100.0% | +60.0 | 0 | 21.4 / 33.9 |
| timeshift_cut_5s | 48 | 2.1% | 100.0% | +97.9 | 0 | 22.1 / 28.6 |
| timeshift_cut_20s | 35 | 0.0% | 100.0% | +100.0 | 0 | 24.0 / 28.0 |
| timeshift_pad_5s | 103 | 0.0% | 99.0% | +99.0 | 0 | 20.0 / 31.6 |
| timeshift_pad_20s | 103 | 0.0% | 99.0% | +99.0 | 0 | 20.5 / 31.5 |
| **pooled edited (30 conditions)** | 2809 | **19.4%** | **89.9%** | **+70.5** | 13 | 68.7 / 220.1 |

| Negatives (not in DB) | n | BEFORE FP | AFTER FP |
|---|---:|---:|---:|
| librispeech_test_clean | 40 | 0 | 0 |
| local_music_mp3_segment | 18 | 0 | 1 |
| local_speech_mp3_segment | 20 | 0 | 0 |
| esc50_time_reversed | 40 | 0 | 0 |
| **all** | 118 | 0 (0.0 %) | **1 (0.9 %)** |

**Latency** (in-process wall time per query):

| | BEFORE mean / p95 | AFTER mean / p95 |
|---|---:|---:|
| all queries | 11.9 / 15.5 ms | 68.7 / 220.1 ms |
| positives | – | 63.8 / 214.7 ms |
| negatives | 13.9 / 16.8 ms | 190.0 / 249.7 ms |

Negatives always run the full 15-hypothesis grid. Every condition's p95 is ≤ 248 ms.

**Per domain** (pooled edited): music 96.1 %, speech 92.5 %, environment (ESC-50) 80.1 %, user song 93.3 % (28/30 edits; the two noise conditions miss).

**Labels** on correct rows:

- duplicate → "Duplicate": 101/102
- pitch/speed → "Edited Audio": 1063/1063
- trims/shifts → "Edited Audio" (by length)
- mp3/gain/HPF → mostly "Duplicate" (same content, same length)

**The 1 false positive.** A 30 s segment of the non-DB `juniorsoundays-ai-music-1` matched the AI-generated DB track `You_Are_A_Monster_…_stableaudio_TTA_001.mp3` under the −3 st hypothesis. The evidence: coarse BER 0.021, fine 0.235 (threshold 0.25), 11 aligned windows, contrast 0.0085 (flat), and reverse 0.18. It is a steady, drone-like passage that matches a steady passage of the other track at any offset. The contrast rule only applies below 10 aligned windows, so it did not catch it. The thresholds were **not** retuned on this held-out result.

**The 13 wrong matches** (0.46 % of 2,809; at most 2 in any condition). 12 of them are content duplicates inside the DB:

- ESC `4-172733-A-36` ↔ `172733.wav` (×5)
- `5-182007-A-36` ↔ `182007.wav` (×2): the same Freesound recordings, also stored under another dataset's name
- GTZAN `jazz.00046` ↔ `jazz.00072` (×5): a known GTZAN repetition

One is a genuine error: `5-215445-A-47` → `Windows_of_the_Skull_Sarin_suno_TTA_002.mp3`.

Dev (seed 7, quick, app path) 93.9 % vs held-out (seed 42, full) 89.9 %. The pipeline-level dev number was 91.4 %, so the gap is within ~2–4 pts.

## 6. Extra: varispeed (resample speed change; not in the pooled number)

`$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --conditions duplicate --with-varispeed --negatives-per-source 1 --out-json .agents/tasks/sonic-robustness/benchmark_after_varispeed.json --out-md .agents/tasks/sonic-robustness/benchmark_after_varispeed.md`

| varispeed | −20 % | −10 % | −5 % | +5 % | +10 % | +20 % |
|---|---:|---:|---:|---:|---:|---:|
| top-1 | 50.5 % | 88.3 % | 94.2 % | 94.2 % | 85.4 % | 53.4 % |

Pooled: 480/618 = 77.7 %. Joint pitch+tempo hypotheses exist (`JOINT_VARISPEED_HYPOTHESES`) but are off by default and were not calibrated.

## 7. Web path (the code the UI uses)

Before these runs `uploads/latest_check_result.json` was copied to `/tmp/m1_latest_backup.json`. It was restored afterwards and `cmp` reports identical.

| command | result |
|---|---|
| `$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --via-http --out-json .agents/tasks/sonic-robustness/benchmark_after_http_quick.json` (multipart POST `/api/check` through the Flask test client, seed 42 quick) | **91.1 %** pooled edited (623/684), **0/28 FP**, mean 74.7 / p95 228 ms. The same queries in-process (from the full run) score 90.8 %; the 4 disagreements are all noise conditions, where the noise sample differs between runs |
| `$PY .agents/tasks/sonic-robustness/repro_web_path.py` | see below |

The user's files, against DB entry "Chaleya Jawan 320 Kbps.mp3" (fid 28408):

| upload | HTTP single | HTTP 2-file batch | direct `run_duplicate_detection` |
|---|---|---|---|
| original mp3 | matched, **Duplicate**, "Chaleya Jawan 320 Kbps.mp3" | – | same |
| pitch +2 st | matched, **Edited Audio**, "Chaleya Jawan 320 Kbps.mp3" (198 ms) | same (207 ms) | same (192 ms) |
| speed +5 % | matched, **Edited Audio**, "Chaleya Jawan 320 Kbps.mp3" (97 ms) | same (113 ms) | same (96 ms) |

BEFORE, both edited files came back `matched: false`.

The new additive response fields, checked through `POST /api/check` (`/tmp/m1_http_fields.py`):

- **pitch file:** `edit_hypothesis: "pitch +2 st"`, `similarity: 0.83`, `verification` with coarse 0.121, fine 0.171, 12 windows, reverse 0.32
- **speed file:** `edit_hypothesis: "length 190.8 s vs 200.4 s in DB (trim / time shift / tempo)"`, `similarity: 0.91`

## 8. Hygiene

- `git status --short sonic_db_store`: only the two pre-existing entries (` M sonic_db_store/manifest.json`, `?? sonic_db_store/fingerprints/fp_28408.npz`). The store was never written, so no backup or rebuild was needed.
- The PORT line in `app.py` is unchanged: `app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), ...)`.
- No commits.
- No `module1_bench_*`, `m1_calib_*` or `sonic_test_*` temp dirs are left.
- `uploads/latest_check_result.json` was restored.
- Not run: `diag_layer_attribution.py 10`. It is a BEFORE-stage diagnostic of the old fused-vote layers, and the full benchmark above supersedes it.
