# Module 1 ("Duplicate & Edited Audio Detection", SONIC-fixed_1): robustness fix report

## Summary

Module 1 failed on edited copies of audio in the reference DB, including the user's pitch +2 st and speed +5 % test files. Four independent defects compounded: three on the query path and one in the decision layer. With them fixed, the held-out benchmark (seed 42, 103 DB tracks, 2,809 edited queries, the app's own `run_duplicate_detection` path) gives:

- pooled edited top-1: **19.4 % → 88.1 %**
- unmodified duplicates: **57.3 % → 97.1 %**
- false positives: **0/118** benchmark negatives (BEFORE 0/118) and **0/784** calibration negatives
- latency: mean 73 ms, p95 224 ms
- no (condition, track) that was correct BEFORE is wrong now

Iteration 2 (review finding B1) added a contrast floor for long alignments. The iteration-1 code had reached 89.9 % with 1/118 FP; the trade-off is described under "Remaining limitations".

The user's three files now return Duplicate / Edited Audio / Edited Audio → "Chaleya Jawan 320 Kbps.mp3", through HTTP and through the direct call. Below target: noise SNR 10 dB (41.7 %) and noise 20 dB (77.7 %). Steady environmental clips under pitch/speed are also weak, which pulls the ESC-50 domain down to 74.5 %. Each is reported below with its cause.

## Root causes (evidence in `sonic-robustness/diagnosis.md`)

1. **The decision layer threw away correct matches (`sonic_db_manager._classify_match`).**
   - 28,408 of 28,409 DB entries have Windows source paths that don't exist on this machine. For those, a match was accepted only on the primary landmark fast path.
   - Every edited query escalates past that fast path, so every one was rejected.
   - Where the source did exist (the user's song), a 5 s time-domain waveform correlation was required. That correlation is ≈ 0 for any pitch, speed, trim or shift edit.
2. **The pipeline's absolute accept gate (score ≥ 600/900) was tuned on 200-file indexes.** At 28k files it rejected correct speed results that were ranked #1.
3. **The pitch stage was hard-disabled (`run_pitch_rescue = False`).** Pitch accuracy was 0 %.
4. **The query window was the first 5 s from t = 0.** Leading silence of ≥ 5 s left it empty, so time-shift "pad" was 0 %.

Two findings came up during this fix:

5. **Time-reversed copies.** Band-delta sub-fingerprints come from per-segment medians, so steady material (noise beds, drones, dense mixes) and its time reversal encode almost alike. Every calibration negative that passed a plain bit-error-rate threshold was a reversed copy of its own DB source. Silent windows also "matched" any file's silence. For the same reason, two *different* steady passages can match over many windows at any offset; iteration 2 found this through the one held-out FP, and handles it with the long-alignment contrast floor.
6. **Single-threaded index build in the app.** In-app startup took ~4 min. The archive held 28,408 files and the manifest 28,409, so a full `.npz` rebuild ran on every start. On top of that, `train_and_eval_40pct` (imported by the app) sets `OMP_NUM_THREADS=1`, so faiss built the 1.9M-window HNSW graphs single-threaded.

Also noted: the ">85–90 %" research figure (`report_data.json`) was measured at much milder edits (±0.5 st, ±0.5 % speed, SNR 30 dB) on 200-file per-domain indexes. It never covered the envelope the user requires.

## What changed

**Fix approach.** Acceptance now rests on fingerprint evidence: the Haitsma & Kalker (2002) aligned bit-error-rate test from the paper the sub-fingerprint code already cites. The pitch hypotheses that research had validated are restored, as per-hypothesis verification rather than an additive bonus.

### `SONIC-fixed_1/sonic/retrieval/pipeline.py`

- **`RetrievalParams`:** every tunable in one place, attached as `DomainIndex.params`. `decision_mode` is `"verify"` (default), `"legacy"` (the BEFORE gate) or `"recal"`.
- **`_verify_escalation`** runs when the primary landmark stage is not confident. For identity, then pitch ±1..4 st (frequency warp) and tempo ×0.8/0.9/0.95/1.05/1.1/1.2:
  1. Search the coarse and fine sub-fingerprint HNSW indexes for candidates.
  2. Estimate each candidate's time offset from the vote histogram.
  3. Compare the query's codes with the candidate's *stored* codes at that alignment on a dense 2-frame grid (`_coarse_scan` / `_coarse_align` / `_finish_align`), giving coarse and fine BER.
  4. Stop early once a strong match is verified.
- **Accept rule (`_passes_abs` / `_passes`):**
  - coarse ≤ 0.24 and fine ≤ 0.25 over ≥ 2 aligned windows
  - fine ≤ 0.15 when there are fewer than 4 windows
  - a 1-window tier for ~2 s clips (coarse ≤ 0.05, fine ≤ 0.10)
  - misalignment contrast ≥ 0.08 when there are fewer than 10 windows, and **≥ 0.04 at 10 or more windows** (`min_contrast_long`, iteration 2)
  - a **time-reversal check** (`_reverse_evidence`): forward must beat "candidate played backwards" by 0.01
- **Silent windows** (≥ 40 dB below the loudest) and all-zero stored windows are ignored.
- **Primary fast path:** acceptance unchanged. Only its Duplicate/Edited label now comes from aligned evidence.
- **`MatchResult` additions** (all additive): `hypothesis`, `evidence`, `candidate_evidence` (per candidate, with `passes`), `match_kind`. `file_id` is the verified decision.
- **`query()`** uses `load_query_window`; `query_array()` uses `skip_leading_silence`.

### Other SONIC modules

- **`sonic/fingerprint/binary_embed.py`:**
  - exact window geometry: `SUBFP_HOP_FRAMES`, `SUBFP_HOP_SEC_EXACT`
  - `TEMPO_HYPOTHESES`, `JOINT_VARISPEED_HYPOTHESES` (off)
  - vectorised `windowed_codes` / `windowed_codes_at`, bit-identical to the index-time codes
  - `hypothesis_codes`, `hypothesis_sub_fingerprints`
- **`sonic/index/subfingerprint_index.py`:**
  - per-file contiguous label ranges (`file_start`, `file_count`, `window_pos`), with a contiguity check
  - stored codes kept for aligned lookups
  - `search_matches`, `vote_detailed`, `aligned_distances`
  - `efSearch` set once at build (64), never per query
  - optional `vote(max_dist=)`; `vote()` is otherwise unchanged
- **`sonic/io.py`:** `skip_leading_silence` and the reworked `load_query_window`, which skips leading silence at −45 dB. A window that starts with content is decoded and returned exactly as before.

### `sonic_db_manager.py`

- **`_classify_match(..., cand_fid=, db_duration=)`** maps the pipeline's per-candidate evidence to `(matched, "Duplicate" | "Edited Audio")`.
  - It no longer depends on whether the source file exists, and no longer correlates waveforms.
  - A "duplicate" whose length differs from the DB entry is labelled Edited Audio.
  - The old classifier is kept only as `_classify_match_legacy`, for result objects without evidence.
- **Audit and intra-batch dedup** pass `cand_fid`, so each flagged candidate needs its own passing evidence. Without this, every ranked candidate would inherit the top match's verdict.
- **`query_file`** adds `edit_hypothesis`, `similarity` and `verification` to its response.
- **Startup:**
  - `_rebuild_index` attaches the archive plus delta `.npz` entries in memory, via `_attach_archive_with_delta` and `_assemble_domain_index`.
  - It raises faiss' thread count only while building.
  - `_manifest_by_id` is refreshed after every build.
  - The store is never written. Attach time is 248 s → **66 s** in-app.

### `app.py`

- passes the three new fields through
- the batch branch passes `cand_fid`
- PORT line unchanged

### Benchmark, tests, calibration

- **`SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py`:** `--with-varispeed` (reported separately, not pooled) and the new fields in each row. The default run is unchanged.
- **`SONIC-fixed_1/tests/`:** 24 stdlib `unittest` tests on synthetic audio (no DB needed). `SteadyContrastTest` covers the steady-drone false-positive case.
- **Calibration:** `sonic-robustness/calib_evidence.py`. Thresholds were tuned only on dev seed 7 plus 784 calibration negatives that don't overlap the held-out set; see `sonic-robustness/experiments.md`.

## Results (held-out seed 42, full benchmark, `app.run_duplicate_detection`)
| Condition | n | BEFORE | AFTER |
|---|---:|---:|---:|
| duplicate | 103 | 57.3 % | **97.1 %** |
| pitch −4 / −2 / −1 st | 103 each | 0 / 0 / 0 % | **82.5 / 84.5 / 83.5 %** |
| pitch +1 / +2 / +4 st | 103 each | 0 / 0 / 0 % | **83.5 / 85.4 / 81.6 %** |
| speed −20 / −10 / −5 % | 103 each | 0 / 0 / 0 % | **83.5 / 85.4 / 85.4 %** |
| speed +5 / +10 / +20 % | 103 each | 0 / 0 / 0 % | **86.4 / 85.4 / 80.6 %** |
| MP3 64 / 128 kbps | 103 each | 49.5 / 57.3 % | **97.1 / 96.1 %** |
| noise SNR 10 / 20 dB | 103 each | 25.2 / 50.5 % | **41.7 / 77.7 %** |
| gain −6 / −3 / +3 / +6 dB | 103 each | 56.3 % (each) | **97.1 %** (each) |
| high-pass 150 / 200 Hz | 103 each | 38.8 / 34.0 % | **95.1 / 96.1 %** |
| trim start 3 / 10 s | 92 / 35 | 0.0 / 5.7 % | **85.9 / 97.1 %** |
| trim end 3 / 10 s | 92 / 35 | 37.0 / 40.0 % | **95.7 / 100 %** |
| time shift cut 5 / 20 s | 48 / 35 | 2.1 / 0.0 % | **100 / 97.1 %** |
| time shift pad 5 / 20 s | 103 each | 0 / 0 % | **97.1 / 97.1 %** |
| **pooled edited (30 conditions)** | 2,809 | **19.4 %** | **88.1 %** |

- **False positives:** 0/118 on the benchmark negatives (BEFORE 0/118, iteration 1 1/118), and 0/784 calibration negatives. BEFORE had zero only because nothing escalated was ever accepted.
- **Wrong-DB-file matches:** 5/2,809 (0.18 %). All five are GTZAN `jazz.00046` matched to `jazz.00072`, which hold the same audio.
- **Latency** (in-process, per query):
  - all queries: mean 73.2 ms, p95 224 ms (BEFORE 11.9 / 15.5 ms)
  - negatives: mean 192 ms, p95 249 ms (they always run the full hypothesis grid)
- **HTTP path** (`--quick --via-http`, POST `/api/check`): 90.3 % pooled edited, 0/28 FP.
- **Labels on correct rows:** duplicate → "Duplicate" 99/100; pitch/speed → "Edited Audio" 1038/1038; trims/shifts → "Edited Audio" 518/518.
- **Per domain:** music 96.0 %, speech 92.5 %, environment 74.5 %, user song 28/30.

The full per-condition table, including iteration 1 and per-condition latency, is in `sonic-robustness/verification.md`.

User's files (`repro_web_path.py`, HTTP single, HTTP 2-file batch and direct call all agree):

| upload | BEFORE | AFTER |
|---|---|---|
| Chaleya Jawan 320 Kbps.mp3 | Duplicate | Duplicate → Chaleya Jawan 320 Kbps.mp3 |
| …pitch_shift_plus_2_semitones.mp3 | not matched | **Edited Audio** → Chaleya Jawan 320 Kbps.mp3 (`edit_hypothesis` "pitch +2 st") |
| …speed_plus_5_percent.mp3 | not matched | **Edited Audio** → Chaleya Jawan 320 Kbps.mp3 (length 190.8 s vs 200.4 s) |

## End-to-end web verification (live server, port 8060)

The server was restarted with the fixed code (`PORT=8060 nohup .venv/bin/python app.py > server.log 2>&1 &`). It returned `GET /` 200 within seconds, and the DB attached (28,409 files) about 60 s later. Each file was then sent as the UI sends it: multipart `POST /api/check` with `module=duplicate_detection`, `domain=auto`, `audio_files=<file>`, synchronous because there is one file. All 14 files were also sent as one `async=1` batch and polled through `GET /api/check/job/<job_id>` (completed, same verdict per file). Edits were generated with the benchmark's `make_edit`.

| upload | expected | returned | HTTP ms |
|---|---|---|---:|
| user `Chaleya_Jawan_pitch_shift_plus_2_semitones.mp3` | Chaleya Jawan 320 Kbps.mp3 | **Edited Audio** → Chaleya Jawan 320 Kbps.mp3 (pitch +2 st) | 215 |
| user `Chaleya_Jawan_speed_plus_5_percent.mp3` | same | **Edited Audio** → Chaleya Jawan 320 Kbps.mp3 (190.8 s vs 200.4 s) | 124 |
| song pitch −4 st | same | Edited Audio → Chaleya Jawan 320 Kbps.mp3 (pitch −4 st) | 235 |
| song speed −20 % | same | Edited Audio → Chaleya Jawan 320 Kbps.mp3 (nearest tempo hypothesis ×0.90) | 149 |
| song MP3 64 kbps | same | Duplicate → Chaleya Jawan 320 Kbps.mp3 | 50 |
| song noise SNR 10 dB | same | not matched (known limitation, see below) | 265 |
| song trim start 10 s | same | Edited Audio → Chaleya Jawan 320 Kbps.mp3 | 56 |
| song time shift +20 s (leading silence) | same | Edited Audio → Chaleya Jawan 320 Kbps.mp3 | 73 |
| GTZAN pitch +4 st | blues.00010.au | Edited Audio → blues.00010.au | 186 |
| GTZAN speed +10 % | blues.00010.au | Edited Audio → blues.00010.au | 34 |
| GTZAN noise SNR 10 dB | blues.00010.au | not matched (known limitation) | 215 |
| GTZAN high-pass 200 Hz | blues.00010.au | Duplicate → blues.00010.au | 32 |
| negative: `the_mountain-general-ai-137025.mp3` (not in DB) | no match | not matched | 202 |
| negative: LibriSpeech test-clean `1089-134686-0000.flac` (not in DB) | no match | not matched | 208 |

Result: 10/12 edits matched the correct DB file, 0/2 false positives. Both user cases pass. The two misses are SNR 10 dB white noise, the documented weak spot (41.7 % on the benchmark). Fixing it needs looser BER thresholds, which would bring false positives, so no code was changed in this step. Afterwards the temp script and audio were deleted, `uploads/latest_check_result.json` was restored, and the stale `/tmp/m1_*` files were removed. The server's own `_cleanup` removed every upload. The store backup is kept at `.agents/tasks/sonic-robustness/backup/` (`manifest.json`, `fp_28408.npz`, plus pre-fix sources in `src/`).

## Remaining limitations
- **Trade-off chosen in iteration 2: zero false positives over steady-sound recall.**
  - Iteration 1 accepted one non-DB AI-music segment as a different DB track. It was a steady, drone-like passage that matched another steady passage equally well at any offset: 11 windows, fine BER 0.235, contrast 0.0085.
  - Requiring misalignment contrast ≥ 0.04 at every window count fixed this (FP 1 → 0/118). The 0.04 value was chosen on dev data only, at no dev cost. The fix was *motivated* by the held-out result.
  - It cost 1.8 pts pooled (89.9 → 88.1 %). The loss is on four steady ESC-50 sounds (wind, airplane, two vacuum-cleaner clips), whose genuine copies, even unmodified ones, also have near-zero contrast. Duplicate accuracy went 99.0 → 97.1 %. None of these were detected BEFORE either.
  - **Not adopted:** exempting near-exact fine matches (fine BER ≤ 0.20) from the floor. This would recover those rows on the iteration-1 evidence and still reject the FP. But it can't be validated on dev data, and choosing it would amount to tuning on the held-out set.
- **Noise at SNR 10 dB: 41.7 %** (BEFORE 25.2 %), below the 50 % goal.
  - White noise at 10 dB randomises a large share of the band-delta bits. The true file's aligned BER rises to 0.25–0.40, which overlaps unrelated audio.
  - Candidate recall is not the problem (the true file is verified in ~90 % of dev cases), so loosening the BER thresholds would trade directly into false positives.
  - Neither an 8 s window (E2) nor a wider HNSW search (E3) helped by ≥ 1.5 pts.
  - Noise 20 dB is 77.7 %, under the 80 % goal.
- **Environmental clips (ESC-50) are the weak domain: 74.5 % pooled edited**, vs music 96.0 % and speech 92.5 %.
  - Many ESC sounds are steady (rain, engines, wind). They can't be told apart from their time reversal or from other steady clips, so they fail the time-reversal or contrast checks that keep false positives at zero.
  - trim start 3 s on ESC leaves 2 s of audio and only 1–2 aligned windows, so only strong matches pass.
- **Pitch ±4 st (81.6–82.5 %) and speed ±20 % (80.6–83.5 %)** are below 85 %, mostly because of ESC clips. ESC clips score 62–71 % on each pitch/speed severity, while music is ≥ 91 % and speech is ≥ 82 % on every one.
- **Landmark fast path (unchanged from BEFORE):** on synthetic data, a steady harmonic drone can be accepted by the primary landmark stage against a different drone of the same chord. The benchmark showed no FP from this path, and its acceptance was deliberately left as the original research behaviour.
- **Varispeed** (resample-type speed change, where pitch moves with tempo; not in the required list or the BEFORE set). Measured on iteration-1 code: ±5 % 94 %, ±10 % 85–88 %, ±20 % 51–53 %. Joint pitch+tempo hypotheses exist but are off and uncalibrated.
- **Latency** went from ~12 ms to a mean of ~73 ms (p95 224 ms). Edited and non-matching queries now try up to 15 edit hypotheses. The primary fast path and clean copies still answer in ~20–25 ms.
- **Content duplicates inside the reference DB** (GTZAN repeats, ESC/Freesound copies) make "the right file" ambiguous. Either answer is the same audio.
- **Not run:** `diag_layer_attribution.py` (a BEFORE diagnostic) and the E3 `top_k` / `max_dist` options; see `experiments.md`. Intra-batch detection without a DB still requires a confident primary match (`app.py` branch C), as before.

## How to rerun

From `/Users/user84/Downloads/prism_v2`:

```bash
PY=.venv/bin/python
# unit tests (no DB needed, ~3 s)
$PY -m unittest discover -s SONIC-fixed_1/tests -p 'test_*.py' -v
# full held-out benchmark (~66 s DB attach + ~15 min)
$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py \
  --out-json .agents/tasks/sonic-robustness/benchmark_after.json \
  --out-md   .agents/tasks/sonic-robustness/benchmark_after.md < /dev/null
# quick run / subset / HTTP path / varispeed
$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --out-json /tmp/m1_quick.json < /dev/null
$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --conditions pitch_+2st,speed_+5pct,timeshift_pad_5s < /dev/null
$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --via-http < /dev/null   # overwrites uploads/latest_check_result.json
$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --conditions duplicate --with-varispeed < /dev/null
# the user's three files through POST /api/check (also overwrites uploads/latest_check_result.json)
$PY .agents/tasks/sonic-robustness/repro_web_path.py < /dev/null
# calibration (dev seed 7): dump evidence, then sweep / check the locked thresholds
$PY .agents/tasks/sonic-robustness/calib_evidence.py --configs final --per-domain 12 --negatives 500 < /dev/null
$PY .agents/tasks/sonic-robustness/calib_evidence.py --sweep .agents/tasks/sonic-robustness/experiments/baseline.json
```

Artifacts are in `.agents/tasks/sonic-robustness/`:

- `benchmark_before.{json,md}` and `benchmark_after.{json,md,log}`
- `benchmark_after_v1_iter1.md` (iteration-1 numbers, 1/118 FP) and `calib_final_e8.log`
- `benchmark_after_varispeed.{json,md}` and `benchmark_after_http_quick.{json,md}`
- `dev_before_seed7.*` and `dev_after_seed7_quick.*`
- `experiments.md` and `experiments/*.json`
- `verification.md`
- source backup in `backup/src/`
