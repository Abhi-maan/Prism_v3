# Review (pass 2): Module 1 robustness fix (uncommitted, prism_v2)

Verdict: APPROVED (0 blockers)

## Prior blocker B1 (FP 0/118 → 1/118): resolved

- Code fix:
  - `pipeline.py:121-129`: adds `min_contrast_long = 0.04`.
  - `_passes_abs` (`pipeline.py:882-899`): now applies a contrast floor at every window count. Below 10 windows the floor is 0.08. At 10 or more windows it is 0.04. A missing contrast is rejected.
- Contrast source: it is computed for every verified candidate (`pipeline.py:799-804`). It is None only when no shifted offset overlaps.
- Tests: `SteadyContrastTest` (`tests/test_pipeline_verify.py:72-103`) covers both cases:
  - a drone with 11 windows and near-zero contrast is rejected, and accepted again with the floor off
  - the held-out FP evidence is rejected by `_passes_abs`
- Recorded results: 24 tests OK, plus the compile check.
- Calibration hygiene: per experiments.md E8, the value was chosen on dev seed 7 plus the 784 calibration negatives only.
  - Dev results: 0.00–0.04 cost nothing; 0.05 costs 0.4 pts.
  - The held-out motivation is disclosed in experiments.md, verification.md and report §limitations.
  - Re-calibration through the real path: dev 91.4 %, 0/784 FP (`calib_final_e8.log`).
- Held-out re-run (`benchmark_after.json`, seed 42, full, `app.run_duplicate_detection`), recomputed from the row data:
  - Pooled edited: 2475/2809 = 88.1 %.
  - The `correct` flag matches `matched && matched_filename == true_filename` on all rows.
  - FP is 0/118, on the same 118 negative files as BEFORE.
  - The (condition, track) keys are identical to BEFORE, and 0 rows went from correct to wrong.
  - Wrong matches: 5, all `jazz.00046 → jazz.00072`, which is a known GTZAN duplicate.
  - Latency over all 3,030 queries: mean 73.2 ms, p95 223.6 ms.
  - All of this matches verification.md and the report.
- Cost of the floor: 89.9 → 88.1 % pooled, on four steady ESC-50 tracks. The report states this honestly and does not present it as free. The fine-BER exemption alternative was rejected because choosing it would have meant tuning on the held-out set. That reasoning is sound.

## Paths affected by the fix and the remaining scope

- HTTP path (`POST /api/check`, quick run): 90.3 %, 0/28 FP.
- The user's three files:
  - original → Duplicate
  - +2 st → Edited Audio
  - +5 % → Edited Audio
  - All three resolve to "Chaleya Jawan 320 Kbps.mp3".
- Requirements:
  - Overall top-1 of 88.1 % meets the ≥85–90 % target.
  - There is no rise in FP.
  - Some per-condition results fall below 85 %: noise 10 dB at 41.7 %, noise 20 dB at 77.7 %, pitch ±4 st at 81.6–82.5 %, speed +20 % at 80.6 %. These are documented with cause as limitations. That follows the approved "report honestly, don't loosen" trade-off and is not a blocker.
- Constraints:
  - Manifest and archive shasums equal `preflight.txt`, so the store was not written.
  - The PORT env line is intact (`app.py:1514`).
  - HEAD = origin/main 057c0e19, so nothing was committed.
  - No file in `Prism-main 2` is newer than plan.md.
- Report (`sonic-robustness-report.md`): it covers root causes (including the steady-passage mechanism), changes, the before/after table, FP rate, latency, limitations, rerun instructions and the iteration-2 trade-off.

No new blockers found.
