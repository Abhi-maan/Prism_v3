# Module 1 (SONIC duplicate / edited audio) — diagnosis + BEFORE baseline

Scope: diagnosis only. No changes to `app.py`, `sonic_db_manager.py` or `SONIC-fixed_1/sonic/**`.
The real store `sonic_db_store/` was only read. A backup of the current `manifest.json` and the
user-ingested `fp_28408.npz` is in `.agents/tasks/sonic-robustness/backup/`.
Pre-existing working-tree changes (not mine): `app.py` PORT line, `sonic_db_store/manifest.json`
plus untracked `fp_28408.npz` (the user's web-UI ingestion of "Chaleya Jawan 320 Kbps.mp3"), `server.log`.

## TL;DR

Module 1 fails on edited audio for several independent reasons. They stack, so fixing just one
does not get the accuracy back:

1. **The decision layer throws away correct matches (sonic_db_manager `_classify_match`). This is the main cause.**
   - 28,408 of the 28,409 DB entries have Windows source paths (`C:\Users\ASUS\...`) that don't exist on this Mac.
     For those, a match is accepted only if `res.confident` is true, i.e. only on the primary landmark fast path.
     Every edited query escalates to the secondary stage (`confident=False`), so it is always rejected,
     even when the pipeline ranked and accepted the right file.
   - Entries whose source does exist (the user's uploaded song) go through a time-domain waveform
     correlation of the first 5 s instead. Under pitch, speed, trim or time shift the waveform is no
     longer sample-aligned, so the correlation is about 0 and the match is rejected.
2. **The pipeline's final accept gate rejects correct speed-change results.** In `pipeline.query_array`,
   the gate is `has_primary` / `has_secondary` with absolute `score_top >= 600/900`. Speed edits rank
   the true file #1 in about 80% of cases, but only 10–29% pass the gate.
3. **The pitch stage is switched off.** `run_pitch_rescue = False` is hard-coded ("disabled for
   ultra-low latency"). Nothing else in the system tolerates ≥1 semitone, so pitch accuracy is 0%.
   With the stage switched back on (in-memory copy, not committed), the true file ranks #1 for
   65–87% of pitch queries at ±1..4 st.
4. **Leading silence / offset (time shift "pad") breaks the query window.** The query reads only the
   first 5 s from t=0 (`load_audio(max_duration=5)`). With ≥5 s of leading silence that window is
   all silence, so retrieval gets 0%. `sonic.io.load_query_window` (skips leading silence) exists but is never used.
5. **Scale / calibration drift between research and app.** The research tuned everything on per-domain
   indexes of 200–333 files. The app runs one 28,409-file index:
   - Sub-fingerprint IDF hub suppression is turned off above 20,000 codes.
   - A 68-minute DESED soundbank file (fid 9724) and two ESC clips (212, 488) top the coarse vote for
     unrelated queries.
   - Primary landmark windows aren't reproducible on long files. A 5 s window of the 200 s song shares
     only 13.6% of hashes with its own stored fingerprint, because the peak budget and percentile
     threshold are per-file.
   - The stored ESC-50 fingerprints were extracted with a different resampler than this machine uses
     (0–92% hash overlap with a fresh extraction of the same file).
6. **The research ">85–90%" baseline was measured at much milder edits than the required envelope.**
   `report_data.json` (Aug 13 2026) shows 92–100% per modification, but at ±0.5 st, ±0.5% speed,
   192 kbps, SNR 30/35 dB, ±1 dB, HPF 50 Hz, 0.05 s trims and ±20 ms shifts. It used per-domain
   200-file indexes and scored pipeline-only correctness (`res.file_id == true`), with no
   `_classify_match`. The required envelope (±1–4 st, 5–20%, …) was only explored in
   `10_augment_experiment.py` / `16_run_600_experiment.py`. Their results aren't in the repo, and
   `17_analyze_accuracy_drop.py` was written to explain an accuracy drop at those severities.
   So "it used to work" applies to milder edits than the user's; it does not carry over to the required envelope.

BEFORE (current code, full benchmark through `app.run_duplicate_detection`, 103 DB tracks, 2,912 queries):
- Unmodified duplicates: **57.3%**.
- Pitch (all levels): **0%**. Speed (all levels): **0%**. Time-shift pad: **0%**. Time-shift cut: 0–2%. Trim-start: 0–6%.
- MP3: 50–57%. Noise: 25–51%. Gain: 56%. HPF: 34–39%. Trim-end: 37–40%.
- Pooled edited accuracy: **19.4%**. False-positive rate on 118 negatives: **0%**.
- App latency: mean 11.9 ms, p95 15.5 ms.

## 1. Reproduction (user's own files, real DB)

Files: `~/Downloads/Chaleya Jawan 320 Kbps.mp3` (DB fid 28408, ingested through the UI on 2026-10-20 02:46,
stored path `uploads/db_staging_1792444566888/...`, which exists), `Chaleya_Jawan_pitch_shift_plus_2_semitones.mp3`,
`Chaleya_Jawan_speed_plus_5_percent.mp3`.

Measured edit parameters (`diag_pitch_rescue.py`, CQT long-term-spectrum lag): pitch file = +2.00 st at the same
tempo. Speed file = tempo ×1.050 (190.8 s vs 200.4 s), pitch unchanged (+0.00 st). That is a tempo-only stretch,
which is what the benchmark's `librosa.effects.time_stretch` reproduces.

Web path (`repro_web_path.py`: Flask test client → `POST /api/check`, module=duplicate_detection, domain=auto,
multipart `audio_files`, the same form the UI sends; single-file uploads are synchronous, ≤200 files):

| upload | matched | match_type | score | stage |
|---|---|---|---|---|
| original mp3 | true | Duplicate | 1008.0 | primary+secondary |
| pitch +2 st | **false** | – | 0.0 | – |
| speed +5 % | **false** | – | 0.0 | – |

The same results come back when both edits are uploaded together, and from a direct call to
`app.run_duplicate_detection([path], "auto")`. Failure reproduced.

Stage trace (`diag_stage_trace.py 28408 …`, true fid 28408):

| query | primary landmark: true rank / top fid (w) / confident | coarse sub-fp vote: true rank (top-3) | fine sub-fp: true rank | pipeline ranked / accepted | app |
|---|---|---|---|---|---|
| original | not in top-20 / 9724 (3.9) / no | 10 (9724, 212, 488) | **1** | #1 / accepted (1008) | Duplicate (waveform corr 1.0) |
| pitch +2 | not in top-20 / 9724 (2.5) / no | none | none | not in top-5 / `file_id=None` | not matched |
| speed +5 | not in top-20 / 9724 (2.6) / no | none | **1** | **#1** (553.7) / **rejected by accept gate** | not matched |

Even the unmodified original never matches on the primary landmark stage. Only the fine sub-fingerprint vote finds it.

## 2. Research benchmark vs app query path

| aspect | research (06/11/16 scripts, report_data.json) | app Module 1 today |
|---|---|---|
| index | `build_domain_index` per domain, 200–333 files, extracted on the same machine/run | one "general" `DomainIndex` over 28,409 files, fingerprints extracted on the author's Windows PC (+1 on this Mac), restored from per-file `.npz` |
| query call | `query(path, idx)` / `query_array(y, idx)` | `sonic_db.query_file(path)` → `query(path, idx)` → `_classify_match` |
| correctness | `res.file_id == true_id` | `matched` from `_classify_match` (path-exists check, waveform corr, thresholds) |
| query window | first 6/5/5 s (speech/music/env) from t=0 | first 5 s from t=0 (`QUERY_MAX_DURATION_SEC` fallback), `domain` arg ignored, no domain routing |
| stages | primary landmark → escalation: global HNSW + coarse + fine sub-fp votes (+ pitch-hypothesis rescue per the report's described fixes) | the same code, but `run_pitch_rescue = False` hard-coded |
| sub-fp IDF hub weighting | on (codes < 20k) | **off** (`n > 20000` → uniform weights) |
| accept gate | final `has_primary/has_secondary` gate in code (absolute 600/900 scores) | same gate, same absolute scores, 140× larger corpus |
| severities tested for the >90% claim | ±0.5 st, ±0.5 %, 192 kbps, 30/35 dB, ±1 dB, HPF 50 / LPF 10k, 0.05 s, ±20 ms | user envelope: ±1–4 st, 5–20 %, 64–128 kbps, 10–20 dB, ±3–6 dB, HPF 150–200, 3–10 s, 5–20 s |
| decoding / SR | float WAV at 22050 generated offline | upload decoded by soundfile (WAV/MP3/FLAC/OGG; M4A via librosa+audioread), mono mean, FFT resample to 22050, peak-normalised; same as ingestion |

## 3. Hypotheses

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| H1 | app doesn't use the research pipeline stages / params | **Partly confirmed.** Same `query()` code, but the pitch stage is disabled, the query cap is 5 s from t=0 with no domain routing, and a separate decision layer is added on top. | `pipeline.py` `run_pitch_rescue = False`. With it re-enabled in memory, pitch top-1 goes from 0–10/31 to 20–27/31 per level (table below). `app.run_duplicate_detection` → `sonic_db.query_file` → `query()` + `_classify_match`. |
| H1a | pitch/speed-invariant stage skipped | **Confirmed for pitch.** Speed is not skipped: retrieval ranks the true file #1 in 24–27/31. | attribution table |
| H1b | domain routing wrong | **Refuted.** No routing exists for DB queries: the `domain` arg only labels unmatched rows, and one unified index is queried. Not a cause by itself. The issue is per-domain calibration not transferring (H6). | `app.py` `_query_worker`, `sonic_db_manager._rebuild_index` (`category="general"`) |
| H1c | wrong SR / duration cap / mono / MP3-M4A decoding | **Refuted for correctness.** MP3/M4A/OGG decode to the expected 5 s window, ingestion and query share `load_audio`, mono mean and 22050 Hz. The duration cap matters only for leading silence (H7). | `soundfile 1.2.2` decodes MP3, M4A falls back to librosa/audioread (5.0 s window, verified). User MP3 queries yield 309–350 hashes, as expected. |
| H2 | stored fingerprints built with different params than query-time | **Partly confirmed.** Same parameters (same array shapes, `raw_vec` equal to 1e-4). Hashes drift on resampled sources: ESC-50 44.1 kHz shows 27% / 0% / 62–92% overlap between stored and fresh extraction. Native-22050 GTZAN/LJSpeech show 85–100%, the Mac-ingested song 100%. | `diag_index_health.py` §4, `diag_landmark_repro.py`. librosa/soxr doesn't reproduce the stored hashes either. The Windows decode/resample path differed. Hurts environmental primary-stage confidence, which then lands in H5. |
| H3 | FAISS/archive inconsistent with manifest (ID/name mismatch) | **Refuted** (no wrong-name mapping). The archive has 28,408 ids, contiguous 0..28407, equal to the manifest's first 28,408 ids. 300-sample check: 0 n_hash mismatches, 0 raw_vec mismatches. FAISS binary-HNSW self-search: 291/300 exact. Benchmark wrong-match count is 0 in every condition. | Side effect: manifest 28,409 ≠ archive 28,408, so `_rebuild_index` skips the archive and rebuilds from 28,409 `.npz` files on every start (75 s standalone, about 245 s inside the app). `_manifest_by_id` is not refreshed after a build (a linear-scan fallback covers it). |
| H4 | decision threshold rejects true #1 matches | **Confirmed. This is the dominant cause.** | `_classify_match`: when the source path is missing it returns `(True,"Duplicate") if res.confident else (False,None)`, so every escalated result is rejected. When the path exists, a time-domain correlation of the first 5 s is required (`corr ≥ 0.15–0.65`), which is about 0 under pitch/speed/trim/shift. There is also a same-filename shortcut (the benchmark renames uploads so it can't inflate results). Pipeline accept gate `has_primary/has_secondary` (absolute 600/900): speed true-#1 24–27/31, but accepted only 3–9/31. |
| H5 | top-k too small / only exact stage effective | **Confirmed as a scale effect.** At 28k files the primary landmark stage misses the true file for long tracks and edits. The coarse vote is hub-dominated (fids 9724 = 68-min DESED soundbank, 212, 488 top for unrelated queries) because sub-fp IDF is disabled above 20k codes. | stage trace; `subfingerprint_index._compute_idf_weights` early return; 5 s window vs full-file landmark overlap 13.6% (200 s song), 39–81% (30 s GTZAN), because the peak budget (`PEAKS_PER_SEC_TARGET × duration`, top-N globally) and the 60th-percentile threshold are computed per whole file |
| H6 | per-query domain choice wrong | **Refuted.** No per-query domain choice is made for DB queries. `build_domain_index("speech", …)` is only used for intra-batch checks when no DB is attached. | `app.py` lines ~304–312 |
| H7 | (found) leading silence / offset breaks the query | **Confirmed.** 0% for 5 s and 20 s leading silence. Retrieval also gets 0/31, so this is not a decision-layer issue. | `load_audio(max_duration=5)` from t=0. `io.load_query_window` is unused. |

Layer attribution (`diag_layer_attribution.py 10`: 31 tracks, 10 per domain + the user's song; same edits as the benchmark):

| condition | n | retrieval top-1 | pipeline accepted | app today (src path missing) | app if src path existed (corr branch) | top-1 with pitch rescue on |
|---|---:|---:|---:|---:|---:|---:|
| duplicate | 31 | 30 | 30 | 20 | 30 | - |
| pitch −4 / −2 / −1 st | 31 | 0 / 0 / 6 | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 0 | 20 / 23 / 24 |
| pitch +1 / +2 / +4 st | 31 | 10 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 27 / 26 / 23 |
| speed −20/−10/−5 % | 31 | 25 / 26 / 26 | 3 / 5 / 6 | 0 / 0 / 0 | 0 / 0 / 2 | - |
| speed +5/+10/+20 % | 31 | 24 / 25 / 27 | 9 / 5 / 3 | 0 / 0 / 0 | 1 / 0 / 0 | - |
| mp3 64 / 128 | 31 | 30 / 29 | 29 / 29 | 17 / 20 | 28 / 29 | - |
| noise SNR 10 / 20 | 31 | 12 / 23 | 12 / 23 | 6 / 17 | 11 / 23 | - |
| gain ±3, ±6 dB | 31 | 29 | 29 | 19 | 29 | - |
| HPF 150 / 200 | 31 | 30 / 30 | 29 / 23 | 14 / 12 | 27 / 21 | - |
| trim start 3 / 10 s | 28 / 11 | 25 / 10 | 17 / 10 | 0 / 1 | 0 / 0 | - |
| trim end 3 / 10 s | 28 / 11 | 27 / 10 | 23 / 10 | 11 / 5 | 21 / 10 | - |
| shift cut 5 / 20 s | 14 / 11 | 12 / 11 | 10 / 10 | 0 / 0 | 0 / 0 | - |
| shift pad 5 / 20 s | 31 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | - |

Reading the table: retrieval already finds the right file #1 for most non-pitch edits (speed, MP3, gain, HPF, trims,
cut-shift). The accept gate and `_classify_match` throw those results away. Pitch needs the disabled rescue stage, or
something better. Strong noise (10 dB) and leading silence are real retrieval weaknesses. Latency with the pitch
rescue on was about 45–50 ms per query (vs about 13 ms), measured in `diag_pitch_rescue.py`.

## 4. Root causes (for the fix step)

1. `sonic_db_manager._classify_match` + `query_file`. The decision depends on whether the original source file exists
   locally (`res.confident` only) or on time-domain waveform correlation. Neither is valid for edited audio. Acceptance
   needs to rest on fingerprint evidence (rank-1 margin / vote agreement / alignment) that holds whether or not the source path exists.
2. `pipeline.query_array` final accept gate (`has_primary` / `has_secondary` absolute 600/900 thresholds). It rejects
   correct top-1 escalated results (speed, HPF 200, trims). It was calibrated on 200-file indexes.
3. `pipeline.query_array` `run_pitch_rescue = False`. The pitch-hypothesis stage (±1..4 st warps) is disabled, so pitch is 0%.
4. Query window `load_audio(..., max_duration=5)` from t=0. Leading silence or offset ≥ window gives 0%.
   `io.load_query_window` exists unused. Long-file landmark non-locality also limits the primary stage.
5. Scale drift: sub-fp IDF disabled above 20k codes (hub files 9724/212/488 dominate the coarse vote). Stored ESC-50
   fingerprints differ from fresh extraction. Thresholds were tuned at 200 files/domain.
6. Startup: manifest (28,409) ≠ archive (28,408), so the full `.npz` rebuild runs on every start (about 4 min inside
   the app). This affects performance and availability, not correctness.

## 5. Benchmark

Script: `SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py`

- **Query path.** Each query is a real file passed to `app.run_duplicate_detection([path], "auto")`. That is the
  function `POST /api/check` with `module=duplicate_detection` calls, against the real `sonic_db_store`. `--via-http`
  sends real multipart POSTs through Flask's test client instead. Nothing is re-implemented.
- **Correct means:** `matched` is true AND `matched_filename` equals the true DB filename (top-1). Uploaded file names
  are prefixed `q_…` so the app's same-filename shortcut can't help.
- **Conditions (31):**
  - `duplicate`
  - pitch ±1/±2/±4 st (librosa pitch_shift)
  - speed ±5/±10/±20 % (librosa time_stretch, pitch kept; this matches the user's file)
  - real `.mp3` at 64/128 kbps
  - white noise SNR 10/20 dB
  - gain ±3/±6 dB (float WAV)
  - HPF 150/200 Hz (Butterworth-4, zero-phase)
  - trim 3/10 s from start and from end
  - time-shift cut 5/20 s (query starts N s in)
  - time-shift pad 5/20 s (N s of leading silence)
- **Eligibility.** Trim and cut need ≥2 s left after the edit, so short clips are skipped for those conditions and
  per-condition n is reported. ESC-50 clips are all 5 s, so they get no 10 s trims or cuts. LJSpeech clips are ≤10 s,
  so only some of them qualify.
- **Test tracks.** DB entries whose original audio exists locally, with unique filenames in the manifest (190
  LJSpeech names are duplicated by wavefake copies and excluded). Sampled with seed 42, 34 per domain:
  - environment: ESC-50 (`~/Downloads/Echoes/ESC-50`)
  - music: GTZAN (`Echoes/real/archive/genres`)
  - speech: LJSpeech + LibriSpeech dev-clean
  - plus the user's ingested song (domain `user_upload`)

  The Windows manifest paths are mapped to `~/Downloads/Echoes/...` only to load the source audio for editing. The
  DB itself is not touched. `--local-source-paths` optionally remaps them in memory to exercise the path-exists branch.
- **Negatives (118):**
  - 40 LibriSpeech test-clean utterances (`Echoes/_negatives`, 0 in the manifest)
  - 18 × 30 s segments of 4 local AI-music mp3s that are not in the DB
  - 20 × 30 s segments of a local talk mp3 that is not in the DB
  - 40 time-reversed ESC-50 clips (synthetic environmental hard negatives)
- **Output.** JSON (meta, per-condition accuracy with per-domain breakdown, wrong-match count, match types,
  latency mean/p95, negatives FPR per source, per-query rows) plus a markdown table. Temp audio goes in
  `tempfile.mkdtemp(prefix="module1_bench_")` and is removed in `finally`.
- **Runtime.** Full mode takes about 6 min of queries plus about 4 min DB attach. Quick mode (8 per domain, 8
  negatives per source) is roughly 2 min plus attach.

Rerun commands (from `/Users/user84/Downloads/prism_v2`):

```bash
# full (BEFORE was produced with exactly this)
.venv/bin/python SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py \
  --out-json .agents/tasks/sonic-robustness/benchmark_after.json \
  --out-md   .agents/tasks/sonic-robustness/benchmark_after.md < /dev/null
# quick iteration
.venv/bin/python SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --out-json /tmp/m1_quick.json < /dev/null
# subset of conditions
.venv/bin/python SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --conditions pitch_+2st,speed_+5pct,timeshift_pad_5s < /dev/null
# through the HTTP endpoint
.venv/bin/python SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --via-http < /dev/null
# diagnostics
.venv/bin/python .agents/tasks/sonic-robustness/repro_web_path.py
.venv/bin/python .agents/tasks/sonic-robustness/diag_layer_attribution.py 10
```

Run it with `< /dev/null` (or the scripts' `-nostdin` ffmpeg already handles it). A backgrounded run that inherits a
TTY can otherwise get SIGTTOU-suspended.

## 6. BEFORE results (current unmodified code, full mode)

Files: `.agents/tasks/sonic-robustness/benchmark_before.json`, `benchmark_before.md`, `benchmark_before.log`.
Run on 2026-10-20 03:48, git 057c0e19 plus the pre-existing uncommitted changes listed above (none to Module 1 logic).
103 tracks (34 env / 34 music / 34 speech / 1 user), 2,912 positive queries + 118 negatives, 0 generation failures.

| Condition | Group | n | Top-1 acc | env | music | speech | user | wrong-match | mean ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| duplicate | Duplicate | 103 | 57.3% | 18/34 | 13/34 | 27/34 | 1/1 | 0 | 12.5 | 18.2 |
| pitch_-4st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.4 | 14.9 |
| pitch_-2st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.4 | 14.9 |
| pitch_-1st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.3 | 14.9 |
| pitch_+1st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.3 | 14.8 |
| pitch_+2st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.3 | 14.9 |
| pitch_+4st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.3 | 15.0 |
| speed_-20pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.5 | 15.1 |
| speed_-10pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.4 | 14.8 |
| speed_-5pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.3 | 14.8 |
| speed_+5pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.0 | 14.6 |
| speed_+10pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 12.8 | 14.4 |
| speed_+20pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 12.4 | 14.7 |
| mp3_64kbps | Compression | 103 | 49.5% | 12/34 | 11/34 | 28/34 | 0/1 | 0 | 13.2 | 16.8 |
| mp3_128kbps | Compression | 103 | 57.3% | 17/34 | 13/34 | 28/34 | 1/1 | 0 | 13.1 | 17.1 |
| noise_snr10db | Noise | 103 | 25.2% | 4/34 | 7/34 | 15/34 | 0/1 | 0 | 12.1 | 14.9 |
| noise_snr20db | Noise | 103 | 50.5% | 14/34 | 12/34 | 26/34 | 0/1 | 0 | 10.8 | 14.4 |
| gain_-6db | Gain | 103 | 56.3% | 17/34 | 13/34 | 27/34 | 1/1 | 0 | 10.3 | 14.1 |
| gain_-3db | Gain | 103 | 56.3% | 17/34 | 13/34 | 27/34 | 1/1 | 0 | 10.0 | 13.8 |
| gain_+3db | Gain | 103 | 56.3% | 17/34 | 13/34 | 27/34 | 1/1 | 0 | 10.0 | 13.6 |
| gain_+6db | Gain | 103 | 56.3% | 17/34 | 13/34 | 27/34 | 1/1 | 0 | 9.9 | 13.8 |
| highpass_150hz | High-pass | 103 | 38.8% | 10/34 | 2/34 | 28/34 | 0/1 | 0 | 11.1 | 14.4 |
| highpass_200hz | High-pass | 103 | 34.0% | 9/34 | 0/34 | 26/34 | 0/1 | 0 | 11.3 | 14.5 |
| trim_start_3s | Trim | 92 | 0.0% | 0/34 | 0/34 | 0/23 | 0/1 | 0 | 10.1 | 14.0 |
| trim_start_10s | Trim | 35 | 5.7% | - | 2/34 | - | 0/1 | 0 | 13.5 | 14.5 |
| trim_end_3s | Trim | 92 | 37.0% | 2/34 | 13/34 | 18/23 | 1/1 | 0 | 8.6 | 13.7 |
| trim_end_10s | Trim | 35 | 40.0% | - | 13/34 | - | 1/1 | 0 | 11.4 | 13.7 |
| timeshift_cut_5s | Time shift | 48 | 2.1% | - | 1/34 | 0/13 | 0/1 | 0 | 12.4 | 14.5 |
| timeshift_cut_20s | Time shift | 35 | 0.0% | - | 0/34 | - | 0/1 | 0 | 13.7 | 16.4 |
| timeshift_pad_5s | Time shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 7.4 | 7.7 |
| timeshift_pad_20s | Time shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 7.3 | 7.6 |

Overall edited accuracy (all modified conditions pooled): 546/2809 = **19.4%** | latency mean 11.9 ms, p95 15.5 ms

| Negatives (not in DB) | n | false positives | FPR |
|---|---:|---:|---:|
| librispeech_test_clean | 40 | 0 | 0.0% |
| local_music_mp3_segment | 18 | 0 | 0.0% |
| local_speech_mp3_segment | 20 | 0 | 0.0% |
| esc50_time_reversed | 40 | 0 | 0.0% |
| **all** | 118 | 0 | 0.0% |

The 0% FPR comes from the same over-strict decision layer that causes the misses. Any fix that loosens acceptance
has to be re-checked against these 118 negatives.

## 7. Files / temp locations

- No temporary store was created: original sources exist locally, so the real store was queried read-only. There is nothing to clean up for a temp store.
- Backup: `.agents/tasks/sonic-robustness/backup/{manifest.json,fp_28408.npz}`.
- Diagnostic scripts (read-only): `.agents/tasks/sonic-robustness/{repro_web_path,diag_stage_trace,diag_index_health,diag_landmark_repro,diag_probe,diag_pitch_rescue,diag_layer_attribution}.py`. Data: `diag_probe_rows.json`, `diag_layer_attribution.json`.
- `uploads/latest_check_result.json` was overwritten by the test-client reproduction and then restored to the user's last content.
- Benchmark temp dirs `$TMPDIR/module1_bench_*` are removed by the script. One stale dir from an aborted first run was deleted.
