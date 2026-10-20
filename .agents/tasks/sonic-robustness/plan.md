# Module 1 (SONIC) robustness fix plan

Inputs: `diagnosis.md` and `benchmark_before.json` in this folder. BEFORE (seed 42, full run): pooled edited top-1 was 19.4%, duplicate 57.3%, FPR 0/118, mean latency 11.9 ms.

All paths below are relative to `/Users/user84/Downloads/prism_v2`. Use absolute paths in commands if your shell cwd is anywhere else. Never touch `/Users/user84/Downloads/Prism-main 2`.

## Ground rules (apply to every item)

- Python: `/Users/user84/Downloads/prism_v2/.venv/bin/python` only. Add no new dependencies. Tests use stdlib `unittest`; the venv has no pytest.
- Do not commit or push. Leave every change uncommitted.
- `app.py`: keep the pre-existing PORT env-var line (`port=int(os.environ.get("PORT", "5000"))`) byte-for-byte.
- `sonic_db_store/` stays read-only. No rebuild is needed, for three reasons:
  - 9,606 "ai_generated" entries plus all the Windows-path sources can't be re-extracted on this Mac.
  - The sub-fingerprint codes the fix relies on are window-local and reproducible.
  - The primary-landmark drift (diagnosis H2/H5) only affects the fast path, which stays as it is.
  
  The plan therefore never writes into `sonic_db_store/`. If a later item finds it truly needs to write there, it must first `cp -R sonic_db_store sonic_db_store.backup-$(date +%Y%m%d-%H%M)` and record why in `results.md`. Check this with `git status --short sonic_db_store` before and after: it must show exactly the two pre-existing entries (` M sonic_db_store/manifest.json`, `?? sonic_db_store/fingerprints/fp_28408.npz`).
- Any HTTP-path run (`--via-http`, `repro_web_path.py`) overwrites `uploads/latest_check_result.json`. Copy it aside first (`cp uploads/latest_check_result.json /tmp/m1_latest_backup.json`) and restore it afterwards.
- Run every benchmark/diagnostic with `< /dev/null`. A backgrounded run that inherits the TTY gets SIGTTOU-suspended.
- Thread safety: `run_duplicate_detection` queries with up to 4 threads. Query code must not mutate shared index state. In particular, set faiss `efSearch` once at build time, never per query.
- Public signatures stay compatible: `query(path, idx, top_k=5)`, `query_array(y, idx, top_k, load_ms, sr)`, and `MatchResult.file_id` (= accepted match or None). Research scripts keep working, and `res.file_id == true_id` keeps meaning "accepted and correct".
- UI contract: keep `matched`, `matched_filename`, `match_type ∈ {"Duplicate","Edited Audio",None}`, `score`, `stage_used` (values `primary` / `primary+secondary`), `confident`, `escalated`. New response fields are additive only.

## Root causes → fix map

| RC | Root cause (diagnosis §4) | Fix item(s) | Benchmark conditions expected to move |
|---|---|---|---|
| RC1 | `_classify_match` needs a local source path (`res.confident` only) or 5 s waveform correlation | 7 | all: duplicate 57→~97, mp3/gain/HPF/trim_end/noise to their retrieval ceilings, trims/cut 0→~85 |
| RC2 | `query_array` absolute 600/900 gate rejects correct escalated #1 results | 6 | speed (0→80+), HPF 200, trim_start, shift_cut |
| RC3 | `run_pitch_rescue = False` | 4, 6 | pitch ±1/±2/±4 (0→75–90) |
| RC4 | query window = first 5 s from t=0 | 5 | timeshift_pad 5/20 s (0→90+); possibly noise via window length (E2) |
| RC5 | scale drift: no sub-fp IDF >20k codes, hub files 9724/212/488, default HNSW efSearch at 1.9M codes | 3, E3 | noise 10/20 dB, pitch ±4, speed ±20 (recall of true file in candidate lists) |
| RC6 | manifest 28,409 ≠ archive 28,408, so a full `.npz` rebuild runs on every start (~4 min) | 2 | none (startup only); needed to make the experiment loops affordable |

## Design decisions (decided here; experiments only tune parameters)

1. **Acceptance moves from the source path / waveform to fingerprint evidence: Haitsma–Kalker aligned bit-error-rate (BER) verification.**
   - Each window in the sub-fp index gets a known in-file time: windows are stored contiguously per file at a fixed hop. That was checked: archive `subfp_fids` is monotonic, and the coarse and fine arrays are both 1,911,089 long, so the same label means the same window in both.
   - For a candidate file and edit hypothesis, the query windows' best-match times give an offset. A Hough-style histogram of `t_db − r·t_query` finds it.
   - The candidate's stored codes at that offset are then compared directly with the query codes: coarse and fine BER over every aligned window, not only top-K neighbours.
   - Why: BER-at-alignment is the classic acceptance statistic from the paper the code already cites. It doesn't depend on whether the source file exists, it works for 2-window (2 s) queries, and random or unrelated audio sits near 0.5 BER. That makes a calibrated threshold FP-safe.
   - The research ranking signals (fused score, margin) stay as candidate generators, and as the E1 baseline.
2. **Acceptance lives in `sonic.retrieval.pipeline`, not in the app.** `MatchResult.file_id` becomes the verified decision, and `sonic_db_manager._classify_match` only maps evidence to `(matched, match_type)`. The research scripts, the benchmark and the app then measure the same thing, which removes the "pipeline accepted / app rejected" split behind RC1.
3. **Multi-hypothesis re-query replaces the bonus-only pitch rescue.**
   - Hypothesis set H = identity (0 st, tempo 1.0), the existing `PITCH_HYPOTHESIS_SEMITONES` (±1..4 st, frequency-warp of the query STFT), and tempo hypotheses r ∈ {0.8, 0.9, 1.1, 1.2}. A tempo hypothesis scales the query sub-fp window/hop by 1/r, so a DB 1.5 s window maps to 1.5/r s of query.
   - Each hypothesis is a full coarse+fine vote plus verification. A candidate keeps its best (lowest-BER) hypothesis.
   - Why: per-hypothesis verification can't be hijacked by a hub file's unwarped base score. That's the failure mode of the current additive bonus, which only reached 65–74% at ±4 st.
   - The integer-semitone grid covers fractional shifts because the coarse code tolerates ±0.5 st (documented in `binary_embed.py`). Tempo hypotheses reuse `sub_fingerprints_from_magnitude(window_sec=, hop_sec=)`, which already exists. No index change.
4. **Early exit for latency.** Identity runs first. If its best candidate passes the strong-accept threshold, skip the other hypotheses. Negatives always run the full grid, so worst-case latency is measured on negatives.
5. **Leading-silence skip at -45 dB relative, not the -16 dB (0.15) heuristic** that config.py says hurt speech. Pads are digital silence, and -45 dB is far below soft speech onsets. The longer decode only happens when the first window starts silent.
6. **Primary landmark fast path is kept unchanged** (`confident=True` → accept, label "Duplicate"). BEFORE showed 0 wrong matches and 0 FPs on that path.
7. **No pytest.** Tests go in `SONIC-fixed_1/tests/test_*.py` with `unittest` and synthetic audio. They need no DB and run in seconds.
8. **Varispeed** (speed with pitch, i.e. resampling) is added to the benchmark behind `--with-varispeed`, reported separately and not in the pooled number, so the BEFORE comparison stays valid. The user's file is tempo-only, but "speed change" tools often resample. E4 tests 4 joint hypotheses (semi = 12·log2 r, tempo r) for it.

## Common commands

```bash
cd /Users/user84/Downloads/prism_v2
PY=/Users/user84/Downloads/prism_v2/.venv/bin/python
# compile check
$PY -m py_compile app.py sonic_db_manager.py SONIC-fixed_1/sonic/io.py SONIC-fixed_1/sonic/config.py \
  SONIC-fixed_1/sonic/retrieval/pipeline.py SONIC-fixed_1/sonic/fingerprint/binary_embed.py \
  SONIC-fixed_1/sonic/index/subfingerprint_index.py SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py
# unit tests
$PY -m unittest discover -s SONIC-fixed_1/tests -p 'test_*.py' -v
# benchmark (from diagnosis.md §5)
$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py \
  --out-json .agents/tasks/sonic-robustness/benchmark_after.json \
  --out-md   .agents/tasks/sonic-robustness/benchmark_after.md < /dev/null          # full, held-out seed 42
$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --out-json /tmp/m1_quick.json < /dev/null
$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --conditions pitch_+2st,speed_+5pct,timeshift_pad_5s < /dev/null
$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --via-http < /dev/null
$PY .agents/tasks/sonic-robustness/repro_web_path.py < /dev/null
$PY .agents/tasks/sonic-robustness/diag_layer_attribution.py 10 < /dev/null
```

Seed policy (anti-overfitting): tune only on dev seed 7 (`--seed 7`, plus the calibration script). Seed 42 is the held-out set: use it only for the final full run in item 10 and for sanity spot-checks with `--conditions`. Never sweep thresholds on seed-42 results.

# Implementation Plan

- [ ] 1. Preflight: snapshot sources and record a dev-seed BEFORE.
      Copy the files this plan changes to `.agents/tasks/sonic-robustness/backup/src/`, preserving relative paths: `app.py`, `sonic_db_manager.py`, and `SONIC-fixed_1/sonic/{io.py,config.py,retrieval/pipeline.py,fingerprint/binary_embed.py,index/subfingerprint_index.py}`. Also copy `SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py`. Record `git status --short` and `shasum sonic_db_store/manifest.json sonic_db_store/unified_archive.npz` in `.agents/tasks/sonic-robustness/preflight.txt`. Run the quick benchmark on the dev seed with the unmodified code so later dev numbers have a baseline.
      Files: `.agents/tasks/sonic-robustness/backup/src/**`, `.agents/tasks/sonic-robustness/preflight.txt`, `.agents/tasks/sonic-robustness/dev_before_seed7.json`
      Verify: `$PY SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py --quick --seed 7 --out-json .agents/tasks/sonic-robustness/dev_before_seed7.json < /dev/null` completes. Pooled edited accuracy should be low (~20%), similar to BEFORE.

- [ ] 2. Fast DB attach via archive + delta (RC6). In `SONICDatabaseManager._rebuild_index`:
      - When the archive's `file_ids` are a strict subset of the manifest ids, load the archive as today, then load only the missing manifest entries' `.npz` files (currently fid 28408) and append them.
        - Landmark arrays: concatenate, with delta local ids = `n_archive + i`. Update `n_hashes_per_file`.
        - `raw_vecs`: vstack.
        - Global binary HNSW: `faiss.read_index_binary(...)`, then `index.add(delta_codes)` (in memory only, never written back). `file_ids` = archive ids + delta ids.
        - Coarse and fine sub-fp: concatenate fids/codes after the archive's, which keeps per-file contiguity.
      - Keep the exact-equality fast path. Fall back to the existing full `.npz` path when the archive contains ids that aren't in the manifest.
      - After any successful build, refresh `self._manifest_by_id`. It is currently only set in `_load_and_init`, which goes stale after append/clean.
      - Factor the "assemble DomainIndex from arrays" code into one private helper used by both paths, so item 3's index fields are set in one place.
      - Add `SONIC-fixed_1/tests/test_db_attach.py`. It builds a temp store (`tempfile.mkdtemp`) with 4 synthetic WAVs (seeded chirp/noise mixtures, 3–6 s) fingerprinted via `extract_fingerprint`. It writes an archive npz with the same keys as the real one (`file_ids, n_hashes, hashes, lids, times, raw_vecs, subfp_fids, subfp_codes, subfp_fine_fids, subfp_fine_codes`) plus a faiss file for 3 of them, and a manifest listing all 4. It asserts that the delta path produces 4 files, every file self-queries to itself, and the full-rebuild path gives identical `ranked_ids[0]`. Instantiate `SONICDatabaseManager(store_dir=tmp)` and wait for `domain_index`. Delete the temp dir in `tearDown`.
      Files: `sonic_db_manager.py`, `SONIC-fixed_1/tests/test_db_attach.py`
      Verify: unit tests pass. `$PY -c "import time,sys;sys.path[:0]=['.','SONIC-fixed_1'];import sonic_db_manager as m;t=time.time();d=m.SONICDatabaseManager();[time.sleep(0.5) for _ in iter(lambda: d.domain_index is None, False)];print(round(time.time()-t),'s',d.domain_index.landmark_index.n_files)" < /dev/null` prints 28409 files and an attach time below the ~75 s standalone full rebuild. Record both numbers in `preflight.txt`. `git status --short sonic_db_store` is unchanged.

- [ ] 3. Sub-fingerprint index: aligned lookup support and search-breadth knob (enables RC1/RC3/RC5 fixes; no ranking change yet). In `SubFingerprintIndex`:
      - Keep a reference to `codes` (the archive arrays already live in memory, so this adds no RAM).
      - Compute `file_start: dict[fid] -> first label` and `file_count`. Assert per-file contiguity at build; if a file's windows aren't contiguous, raise `ValueError`, since every builder in this repo produces contiguous blocks.
      - Add `ef_search: int` applied once in `build()` via `self.index.hnsw.efSearch = max(ef_search, k)`. Default `SUBFP_EF_SEARCH = 64`, tuned in E3.
      - Add `vote_detailed(query_codes, top_k_per_window) -> dict[fid, list[(window_idx, label, dist)]]`, the best label per (query window, file).
      - Add `aligned_distances(fid, db_window_idx: np.ndarray) -> np.ndarray`, Hamming distances computed with `np.unpackbits` / popcount on the stored codes, `-1` where out of range.
      - Add optional BER-threshold voting `vote(..., max_dist=None)` so only neighbours with `dist <= max_dist` count. Used in E3; `None` keeps today's behaviour.
      - Add `SUBFP_HOP_SEC_EXACT = round(SUBFP_HOP_SEC/frame_dt)*frame_dt` (≈0.3018 s) in `binary_embed.py` so DB window times are exact.
      - Existing `vote()` output must be bit-identical when `max_dist=None`.
      - Tests (`SONIC-fixed_1/tests/test_subfp_index.py`): synthetic 3-file index (built with `build_domain_index` on temp WAVs).
        - A query cut from file B starting at 2.1 s returns B windows whose `label - file_start[B]` advance by 1 per query window.
        - `aligned_distances` at the true offset gives mean BER < 0.1, at a wrong offset > 0.3.
        - `vote()` output is unchanged against a copy computed before the change; hard-code a small expected list from the current code during item 1.
      Files: `SONIC-fixed_1/sonic/index/subfingerprint_index.py`, `SONIC-fixed_1/sonic/fingerprint/binary_embed.py`, `SONIC-fixed_1/tests/test_subfp_index.py`
      Verify: unit tests pass. `--quick --seed 7 --conditions duplicate,mp3_64kbps` gives the same accuracy as `dev_before_seed7.json` for those conditions (ranking untouched).

- [ ] 4. Hypothesis code generation (RC3, plus tempo for RC2/speed). In `binary_embed.py`:
      - Add `TEMPO_HYPOTHESES = (0.8, 0.9, 1.1, 1.2)` and `JOINT_VARISPEED_HYPOTHESES` (pairs `(12*log2(r), r)` for the same r values; used only if E4 enables them).
      - Add `hypothesis_sub_fingerprints(mag, frame_times, sr, semitone, tempo, n_bands, query_hop_sec)`. It returns `[(t_query_start, raw_vec)]`: frequency-warp via the existing `warp_magnitude_freq` when `semitone != 0`, then `sub_fingerprints_from_magnitude(..., window_sec=SUBFP_WINDOW_SEC/tempo, hop_sec=query_hop_sec/tempo, n_bands=n_bands)`.
      - Keep `pitch_hypothesis_sub_fingerprints` as a thin wrapper for compatibility.
      - Tests (`SONIC-fixed_1/tests/test_hypotheses.py`): a synthetic harmonic signal (sum of 5 partials with an AM envelope, 6 s), pitch-shifted +2 st with `librosa.effects.pitch_shift`, and time-stretched with rate 1.1. The matching hypothesis's codes have lower mean Hamming distance to the original's windows than the identity hypothesis. `librosa` exists in the venv and is only used inside the tests.
      Files: `SONIC-fixed_1/sonic/fingerprint/binary_embed.py`, `SONIC-fixed_1/tests/test_hypotheses.py`
      Verify: unit tests pass.

- [ ] 5. Query window with leading-silence skip (RC4). In `sonic/io.py`, change `load_query_window(path, target_duration, sr, lookahead_sec=30.0, silence_rel_db=-45.0, min_skip_sec=0.3)`:
      - Decode `target_duration` first.
      - If its first 100 ms frame is below `10**(silence_rel_db/20) * max_frame_rms` (or the whole window is silent), decode `target_duration + lookahead_sec` and find the first frame above the threshold. Use `trim_leading_silence(..., rel_threshold=10**(silence_rel_db/20))`; the threshold is relative to the longer span's max frame RMS.
      - Return `y[start:start+target]`, re-peak-normalised. Skip only when `start >= min_skip_sec*sr`.
      - Add `query_array_from_window`-style use in `pipeline.query()`: when `idx.skip_leading_silence` (new `DomainIndex` field, default True), use `load_query_window(path, idx.query_max_duration_sec)` instead of `load_audio(..., max_duration=...)`.
      - `query_array()` gets the same skip applied to the in-memory array via a shared helper `skip_leading_silence(y, sr, ...)`, so array-based research benchmarks behave the same.
      - Tests (`SONIC-fixed_1/tests/test_query_window.py`):
        - 5 s and 20 s zero pads, and 2 s of −60 dB noise before content, are skipped.
        - Content starting at t=0 is untouched (identical array).
        - Speech-like signal with a soft −30 dB onset is not cut.
      Files: `SONIC-fixed_1/sonic/io.py`, `SONIC-fixed_1/sonic/retrieval/pipeline.py`, `SONIC-fixed_1/tests/test_query_window.py`
      Verify: unit tests pass. `--quick --seed 7 --conditions timeshift_pad_5s,timeshift_pad_20s,duplicate` shows retrieval improving: pad conditions should now behave like `duplicate`. App acceptance is still limited until item 7. Duplicate on speech must not drop versus `dev_before_seed7.json`.

- [ ] 6. Pipeline: multi-hypothesis verification and evidence-based acceptance (RC2, RC3, RC5). In `sonic/retrieval/pipeline.py`:
      - Add `@dataclass RetrievalParams`, attached as `DomainIndex.params`, with every tunable:
        - `query_hop_sec=0.3`, `subfp_top_k=20`
        - `pitch_hyps=PITCH_HYPOTHESIS_SEMITONES`, `tempo_hyps=TEMPO_HYPOTHESES`, `joint_hyps=()`
        - `vote_max_dist=None`
        - `verify_top_n=5`, `offset_bin_sec=0.3`
        - `accept_coarse_ber=0.25`, `accept_fine_ber=0.35`, `strong_fine_ber=0.20`, `min_aligned_windows=2`, `margin_ber=0.03`
        - `dup_fine_ber=0.12`
        - `decision_mode="verify"` (alternatives `"legacy"` = today's 600/900 gate for comparison, `"recal"` = E1 baseline B)
        
        Starting values are placeholders, finalised in item 9.
      - Escalation flow after the unchanged primary stage:
        - Keep the fused candidate scores, which give the legacy ranking (`ranked_ids`).
        - For hypothesis h0 (identity) then the rest of H: compute coarse+fine codes (item 4), run `vote_detailed` on both indices, and take the candidate set = top `verify_top_n` by fused vote for that h, ∪ top-3 legacy fused candidates for h0.
        - For each candidate: estimate the offset by histogramming `t_db - tempo*t_q` over the coarse matches (bin `offset_bin_sec`, merge ±1 bin), map each query window to its DB window index, and get coarse and fine BER from `aligned_distances` over the in-range windows. Also try offset ±1 window and keep the best.
        - Keep each candidate's best hypothesis by fine BER (coarse BER as tie-break).
        - Early exit after h0 when the best candidate has `fine_ber <= strong_fine_ber` and `n_aligned >= min_aligned_windows`.
      - Accept rule (`decision_mode="verify"`): top candidate by fine BER with `n_aligned >= min_aligned_windows` and `coarse_ber <= accept_coarse_ber` and `fine_ber <= accept_fine_ber`. When the second-best distinct file also passes, also require `second.fine_ber - best.fine_ber >= margin_ber`; otherwise reject as ambiguous.
      - `file_id` = accepted fid or None. Set `ranked_ids` from verified order (accepted first), falling back to legacy order.
      - Remove the hard-coded `run_pitch_rescue = False` and the bonus code. Hypotheses now live in the verify stage. Keep the old fusion only under `decision_mode="legacy"` so E1 can reproduce BEFORE behaviour.
      - Extend `MatchResult` with defaults so it stays constructible positionally:
        - `hypothesis: dict` (`{"semitones":s,"tempo":r}`)
        - `evidence: dict` (`coarse_ber, fine_ber, n_aligned, offset_sec, second_fine_ber`)
        - `candidate_evidence: dict[int, dict]` (per verified fid, including `passes` = absolute thresholds met)
        - `match_kind: str|None` (`"duplicate"` if primary-confident, or identity hypothesis and `fine_ber <= dup_fine_ber`; else `"edited"`)
        - `timing_ms["verify_ms"]`, `timing_ms["n_hypotheses"]`
      - The primary-confident fast path also fills `candidate_evidence` for its top fid, with `passes=True` and `match_kind="duplicate"`.
      - Apply the same flow to small batch indexes (`build_domain_index`); nothing in it depends on corpus size.
      - Tests (`SONIC-fixed_1/tests/test_pipeline_verify.py`, synthetic 6-file index built with `build_domain_index` on temp WAVs):
        - Exact copy, +2 st pitch, tempo ×1.1, 5 s zero-pad, and 3 s start trim of file 2 are all accepted as file 2, with `match_kind` duplicate/edited as appropriate.
        - An unrelated synthetic file and a time-reversed file 2 are rejected (`file_id is None`).
        - `decision_mode="legacy"` reproduces the old gate (`file_id` None for the tempo case).
      Files: `SONIC-fixed_1/sonic/retrieval/pipeline.py`, `SONIC-fixed_1/tests/test_pipeline_verify.py`
      Verify: unit tests pass and the compile check passes. `$PY .agents/tasks/sonic-robustness/diag_stage_trace.py 28408 ~/Downloads/Chaleya_Jawan_pitch_shift_plus_2_semitones.mp3 < /dev/null` (or a 5-line ad-hoc call of `query()` on the 3 user files against `SONICDatabaseManager().domain_index`) shows fid 28408 accepted for all three files.

- [ ] 7. Decision layer and callers (RC1). In `sonic_db_manager.py`:
      - Rewrite `_classify_match(cls, qpath, mpath, res, cand_fid=None)`:
        - `fid = cand_fid if cand_fid is not None else res.file_id`, and `ev = res.candidate_evidence.get(fid)`.
        - If `ev` exists and `ev["passes"]` (and, when `fid == res.file_id`, the pipeline accepted it), return `(True, "Duplicate" if kind == "duplicate" else "Edited Audio")`.
        - Otherwise return `(False, None)`.
        - Keep the existing same-filename shortcut only when `fid == res.file_id` and `res.file_id is not None`.
        - The result must not depend on `Path(mpath).exists()`, and there is no waveform load.
        - `_classify_match_type` delegates to the new logic when `res` is given.
      - `query_file`: classify with `cand_fid=res.file_id`. Add additive keys `edit_hypothesis` (e.g. `"pitch +2 st"`, `"tempo x1.10"`, `"none"`), `similarity` (`round(1 - fine_ber, 4)`), and `verification` (the evidence dict). Fill `timing_ms` keys used by the UI as before.
      - Ingestion intra-batch dedup loop and `audit_database`: pass `cand_fid=cand_fid`. This fixes a latent hazard: with evidence-based classification, every ranked candidate would otherwise inherit the top-1's evidence, and audit's "Remove all duplicates" could delete unrelated entries. Only candidates with their own passing evidence count. In audit, skip `cand_fid == fid` (self).
      - `app.py` `_query_worker`: pass the new additive keys through in the DB branch. In the batch branch (no DB), call `_classify_match(..., cand_fid=res_b.file_id)`. No other `app.py` changes. PORT line untouched.
      - Tests (`SONIC-fixed_1/tests/test_classify_match.py`):
        - A fake `res` with candidate evidence gives the same verdict for an existing and a non-existing `mpath`.
        - `cand_fid` without evidence is rejected.
        - The audit-style loop over `ranked_ids` flags only candidates with passing evidence.
      Files: `sonic_db_manager.py`, `app.py`, `SONIC-fixed_1/tests/test_classify_match.py`
      Verify: unit tests pass. `$PY .agents/tasks/sonic-robustness/repro_web_path.py < /dev/null` (back up and restore `uploads/latest_check_result.json`) shows:
      - Original: matched Duplicate → "Chaleya Jawan 320 Kbps.mp3".
      - Pitch +2 st and speed +5 % files: matched "Edited Audio" → same file, in single and 2-file requests and the direct call.
      
      `--quick --seed 7` pooled edited accuracy is well above `dev_before_seed7.json`, with 0 false positives in quick negatives. If FPs appear, don't loosen anything; continue to items 8–9 calibration.

- [ ] 8. Calibration tooling and benchmark extensions.
      (a) `SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py`:
        - Add `--with-varispeed`, which adds `varispeed_{-20,-10,-5,+5,+10,+20}pct` (`librosa.resample(y, orig_sr=SR, target_sr=round(SR/(1+p/100)))`, written at SR, so pitch and tempo both change). Store these under `overall["extra"]`; they are excluded from `edited_accuracy` and still shown in the markdown table.
        - Store `edit_hypothesis`, `similarity`, `verification` in each row when present.
        - Default behaviour and condition set otherwise unchanged.
      (b) New `.agents/tasks/sonic-robustness/calib_evidence.py`. One process, one DB attach (`SONICDatabaseManager()`):
        - Generate dev positives with `B.select_tracks(manifest, echoes, 12, seed=7)` × all 30 conditions (+ varispeed), using `B.make_edit`, cached under `tempfile.mkdtemp(prefix="m1_calib_")`.
        - Generate ≥ 500 calibration negatives, disjoint from the benchmark's seed-42 negatives (re-derive those by replaying `B.build_negatives`'s RNG `random.Random("42|neg")` and exclude by filename):
          - 200 LibriSpeech test-clean from `~/Downloads/Echoes/_negatives`.
          - 80 time-reversed GTZAN 30 s clips.
          - 80 time-reversed ESC-50 clips.
          - 40 reversed ESC clips trimmed to 2 s (short-query hard negatives).
          - Pitch ±2/±4 st and tempo ×0.8/×1.2 versions of 30 of the test-clean and 30 of the reversed-GTZAN negatives. These matter because the hypothesis grid itself raises FP opportunity.
        - For each named config (a dict of `RetrievalParams` overrides + `efSearch` + window length), mutate `idx.params` in place (single-threaded), set efSearch on both sub-fp indexes, and run `pipeline.query` on every file with `decision_mode="verify"` and thresholds wide open (accept everything), so raw evidence is dumped. Write `.agents/tasks/sonic-robustness/experiments/<config>.json` (rows: condition, domain, true fid, top fid, evidence, best hypothesis, latency).
        - `--sweep` mode reads a dump and grid-searches `(accept_coarse_ber, accept_fine_ber, min_aligned_windows, margin_ber)`. Report dev pooled accuracy and per-condition accuracy at **0 calibration FPs**, using the threshold one grid step stricter than the loosest 0-FP point as a safety margin, plus p50/p95 latency. Print a markdown table.
        - Clean up the temp dir in `finally`.
      Files: `SONIC-fixed_1/scripts/20_module1_robustness_benchmark.py`, `.agents/tasks/sonic-robustness/calib_evidence.py`
      Verify: `$PY .agents/tasks/sonic-robustness/calib_evidence.py --configs baseline --per-domain 4 --negatives 60 < /dev/null` runs end to end and writes `experiments/baseline.json`. `--sweep experiments/baseline.json` prints a table. `--quick --seed 7 --with-varispeed --conditions varispeed_+5pct,duplicate` runs. No `m1_calib_*` / `module1_bench_*` left in `$TMPDIR`.

- [ ] 9. Run experiments E1–E6 on dev + calibration negatives, then lock parameters. Each experiment is one `calib_evidence.py` config set followed by `--sweep`. Record every table and decision in `.agents/tasks/sonic-robustness/experiments.md`. Decision rule for all: pick the config with the highest dev pooled edited accuracy at 0 calibration FPs and p95 latency ≤ 300 ms. If two are within 1.5 pts, pick the cheaper or simpler one. If an alternative doesn't beat the default by ≥ 1.5 pts, keep the default.

      | Exp | Question | Options | Default if inconclusive |
      |---|---|---|---|
      | E1 | Does BER verification beat recalibrated research signals? | A `verify` (aligned BER); B `recal` (research fused `score_top`/`sec_margin` with thresholds swept to 0 FP, path-independent, pitch hypotheses as the old bonus); C `legacy` (sanity = BEFORE) | A |
      | E2 | Query window and query hop | window 5 / 8 / 10 s × `query_hop_sec` 0.3 / 0.15 | 5 s, 0.15 (watch noise_10dB, trims, ESC short clips, latency) |
      | E3 | Candidate recall at 1.9M codes (hubs, HNSW) | `efSearch` 16 / 64 / 128 / 256; `subfp_top_k` 20 / 50; `vote_max_dist` None / 20 / 24 bits (coarse 64-bit) | efSearch 64, top_k 20, max_dist None. Persisting an IDF cache is out of scope unless every E3 option leaves noise_snr10db and pitch ±4 candidate recall (true fid in verified set) < 70%; then note it as future work in `results.md`, don't implement. |
      | E4 | Hypothesis grid | pitch ±1..4 integer (current) vs adding ±3; tempo {0.8, 0.9, 1.1, 1.2} vs {0.85, 0.92, 1.08, 1.17} vs none; joint varispeed 4 on/off | current pitch grid; tempo {0.8, 0.9, 1.1, 1.2}; joint off unless varispeed improves ≥ 10 pts at ≤ +30 ms p95 |
      | E5 | Leading-silence skip threshold | −35 / −45 / −55 dB; plus a second-window re-query (query again from content start + 5 s when not accepted, accept only with the same thresholds) on/off | −45 dB, second-window off. Turn it on only if it adds ≥ 2 pts at 0 calibration FPs. |
      | E6 | Duplicate vs Edited labelling | `dup_fine_ber` 0.08 / 0.12 / 0.16 | the value that gives duplicate→"Duplicate" ≥ 90% and pitch/speed→"Edited Audio" ≥ 95% |

      Write the chosen values as the defaults in `RetrievalParams` / `SUBFP_EF_SEARCH`, each with a short comment citing the experiment and its dev numbers, matching the style of the existing measured-rationale comments.
      Files: `SONIC-fixed_1/sonic/retrieval/pipeline.py`, `SONIC-fixed_1/sonic/fingerprint/binary_embed.py`, `SONIC-fixed_1/sonic/index/subfingerprint_index.py`, `SONIC-fixed_1/sonic/config.py` (if the window length changes, update `QUERY_MAX_DURATION_SEC` and the comment), `.agents/tasks/sonic-robustness/experiments.md`, `.agents/tasks/sonic-robustness/experiments/*.json`
      Verify: unit tests pass. A final `calib_evidence.py --configs final` dump with locked thresholds (no sweep) shows 0 calibration FPs. `--quick --seed 7 --with-varispeed` meets the acceptance floors below, within ~5 pts given the quick sample size.

- [ ] 10. Held-out evaluation and report. Run the full benchmark on seed 42 (`benchmark_after.json/.md`, plus `--with-varispeed` as a separate `benchmark_after_varispeed.json`), `--quick --via-http`, `repro_web_path.py`, and `diag_layer_attribution.py 10`. Back up and restore `uploads/latest_check_result.json` around the HTTP runs. Write `.agents/tasks/sonic-robustness/results.md` with:
      - BEFORE vs AFTER per-condition table, with deltas.
      - FPR per negative source.
      - Latency mean/p95 for positives and negatives.
      - Attach time.
      - The user's 3 files.
      - Which RC each change fixed.
      - Remaining shortfalls with their cause (expected candidates: noise 10 dB, pitch ±4 on environmental clips).
      - The dev (seed 7) vs held-out (seed 42) gap.
      
      Never retune on seed-42 results. If a floor is missed, report it honestly rather than loosening thresholds. Confirm `git status --short sonic_db_store` is unchanged and no `module1_bench_*` / `m1_calib_*` temp dirs remain.
      Files: `.agents/tasks/sonic-robustness/benchmark_after*.{json,md}`, `.agents/tasks/sonic-robustness/results.md`
      Verify: the commands above complete. The acceptance criteria below hold, or each miss is documented in `results.md`.

## Acceptance criteria (held-out seed 42 full run, `app.run_duplicate_detection` path)

| Metric | Target |
|---|---|
| Pooled edited top-1 (30 edited conditions, same definition as BEFORE) | ≥ 85% (goal 90%) |
| duplicate | ≥ 97% |
| pitch ±1, ±2 st | ≥ 85% each |
| pitch ±4 st | ≥ 75% each |
| speed ±5, ±10 % | ≥ 85% each |
| speed ±20 % | ≥ 80% each |
| mp3 64 / 128 kbps | ≥ 92% |
| gain ±3/±6 dB | ≥ 95% |
| high-pass 150 / 200 Hz | ≥ 88% |
| noise SNR 20 dB | ≥ 80% |
| noise SNR 10 dB | ≥ 50% (report; stretch 70%) |
| trim start/end 3/10 s | ≥ 85% |
| time-shift cut 5/20 s | ≥ 85% |
| time-shift pad 5/20 s | ≥ 90% |
| wrong-match (matched to the wrong DB file) | ≤ 1 per condition, ≤ 0.5% pooled |
| False positives | 0/118 benchmark negatives; 0 on the ≥ 500 calibration negatives with locked thresholds (≤ 1 tolerated only with a written cause) |
| Labels | duplicate condition → "Duplicate" ≥ 90%; pitch/speed → "Edited Audio" ≥ 95% (correct rows) |
| User files (`repro_web_path.py`) | original → Duplicate; pitch +2 st and speed +5 % → "Edited Audio", matched_filename "Chaleya Jawan 320 Kbps.mp3", via HTTP single, HTTP 2-file batch, and direct call |
| Latency (in-process wall per query, positives + negatives) | mean ≤ 120 ms, p95 ≤ 300 ms; negatives p95 ≤ 300 ms |
| HTTP quick run | same accuracy as in-process within quick-sample noise; 0 FP |
| Dev/held-out gap | pooled seed 7 vs seed 42 within ~5 pts (no overfitting) |
| Startup | DB attaches with 28,409 files; time recorded (expected well under the old ~245 s in-app) |
| Hygiene | unit tests pass; `sonic_db_store` unchanged; PORT line intact; no commits; temp dirs removed; `uploads/latest_check_result.json` restored |

## Gaps and assumptions

- Varispeed (resample-speed) isn't in the user's file or the BEFORE set. It's reported but doesn't gate.
- Thresholds come from synthetic edits (librosa phase-vocoder, white noise). Real-world tools (Audacity, DAWs) can differ slightly. The integer pitch grid and BER tolerance cover fractional settings, and the user's own two files act as the real-tool check.
- If E3 shows candidate recall, not verification, limiting noise 10 dB, a persisted sub-fp IDF cache would need a new derived file in the store. That is deferred; see E3.
