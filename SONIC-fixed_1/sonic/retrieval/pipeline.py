"""Confidence-guided adaptive retrieval pipeline - the core novelty of SONIC.

Flow:
  1. Load query audio once (capped to QUERY_MAX_DURATION_SEC), used for both stages.
  2. Primary stage: log-frequency landmark hashing + inverted-index voting
     (pitch/tempo invariant, near-zero cost). Compute a confidence score from
     the top candidate's temporal-offset consistency and its score margin over
     the runner-up.
  3. If confident: return the primary top candidate. This is the fast path and
     is expected to cover the large majority of queries (clean duplicates and
     mildly-edited audio), keeping average latency and RAM low.
  4. If not confident (heavy modification, sparse-landmark environmental audio,
     noise): compute the secondary binary embedding (Haitsma-Kalker style
     band-delta code) from the SAME loaded audio and search the binary-HNSW
     index for a shortlist. Candidates found only via the secondary shortlist
     (i.e. missed by the primary inverted-index lookup entirely) are verified
     with a direct pairwise landmark-hash overlap against the query, and the
     final answer is the candidate with the best available evidence across
     both stages.

  5. Acceptance of escalated queries (decision_mode="verify", the default):
     candidates from the sub-fingerprint votes are re-searched under pitch
     (+-1..4 st) and tempo (x0.8..1.2) hypotheses and each is verified by the
     bit error rate between the query's codes and the candidate's STORED codes
     at one consistent time alignment (Haitsma & Kalker 2002), plus a
     time-reversal check. See _verify_escalation and RetrievalParams.

Every result carries its evidence (confidence, scores, which stage produced it)
for explainability.
"""

from dataclasses import dataclass, field

import numpy as np

from sonic.config import ANALYSIS_SR, QUERY_MAX_DURATION_SEC, QUERY_MAX_DURATION_SEC_BY_DOMAIN
from sonic.fingerprint.binary_embed import (
    band_energy_vector_from_magnitude, fine_sub_fingerprints_from_magnitude,
    sub_fingerprints_from_magnitude, pitch_hypothesis_sub_fingerprints,
    pitch_hypothesis_fine_sub_fingerprints, BinaryEmbedIndex,
    SUBFP_N_BANDS, SUBFP_FINE_N_BANDS, PITCH_HYPOTHESIS_SEMITONES, TEMPO_HYPOTHESES,
    SUBFP_HOP_FRAMES, SUBFP_WINDOW_FRAMES, SUBFP_N_SEG, log_mel_bands, warp_magnitude_freq, windowed_codes,
    windowed_codes_at,
)
from sonic.fingerprint.landmarks import build_landmarks_from_magnitude, stft_magnitude, Landmark, HOP_LENGTH
from sonic.index.binary_hnsw_index import BinaryHNSWIndex
from sonic.index.landmark_index import LandmarkIndex, OFFSET_BIN_SEC as LANDMARK_OFFSET_BIN_SEC
from sonic.index.subfingerprint_index import SubFingerprintIndex, _POPCOUNT8 as _POP
from sonic.io import load_audio, load_query_window, skip_leading_silence

# Confidence-gate defaults. Calibrated empirically per domain in
# scripts/03_run_benchmark.py's calibration sweep (see REPORT.md Sec. "adaptive
# threshold calibration"); these are the pre-calibration starting values.
DEFAULT_OFFSET_CONF_TAU = 0.15
DEFAULT_MARGIN_TAU = 0.20
# Measured bug: under pitch/speed-shifted queries, primary weighted_score
# collapses to noise level (2-17) but the margin between a noisy top1/top2 can
# still easily exceed 20% by chance, so a low min_score_tau let noise pass as
# "confident" and skip the secondary stage entirely (in one 20-query pitch
# test, 11/20 wrongly short-circuited this way; the 9/20 that DID escalate
# were 9/9 correct via the multi-window sub-fingerprint vote). Clean/duplicate
# true matches score in the hundreds-to-thousands, so 50 cleanly separates
# "genuinely confident" from "noise that happens to have some margin."
DEFAULT_MIN_SCORE_TAU = 50.0
DEFAULT_MIN_ACCEPT_SCORE = 0.5  # absolute floor below which we report "no match" rather than a weak guess

# Gate for the +-1 semitone pitch-hypothesis rescue (see its use in
# query_array below) - roughly half of DEFAULT_MARGIN_TAU (0.20), so this only
# triggers for genuine razor-thin ties (< 10% margin).
PITCH_RESCUE_MARGIN_TAU = 0.10

PRIMARY_POOL_K = 20
SECONDARY_SHORTLIST_K = 10
# Sub-fingerprint vote search breadth. Measured: at the default-scale K=5/10,
# the true match was often missed by the initial FAISS shortlist before voting
# even had a chance to aggregate it (rank@20-with-K=20 found it in 20/20 test
# cases at rank 0-5, vs the pipeline's end-to-end accuracy being far lower at
# K=5/10) - this corpus is small (a few thousand window-codes for 200 files),
# so a wider per-window search is still cheap.
SUBFP_TOP_K_PER_WINDOW = 20
SUBFP_FINAL_TOP_K = 20


@dataclass
class RetrievalParams:
    """Every tunable of the escalation stage in one place (attached to each
    DomainIndex as .params). decision_mode:
      "verify" - multi-hypothesis candidate search + aligned bit-error-rate
                 verification (Haitsma & Kalker 2002): a candidate is accepted
                 only when the query's sub-fingerprint codes agree with the
                 candidate's STORED codes at one consistent time alignment.
      "legacy" - the previous fused-vote ranking + absolute 600/900 gate, kept
                 only to reproduce the BEFORE behaviour for comparison.
      "recal"  - legacy fused ranking with the pitch bonus on and the gate
                 replaced by recal_score / recal_margin (comparison baseline).
    """
    query_hop_frames: int = 14          # search-grid hop of query windows (~0.16 s)
    verify_hop_frames: int = 2          # dense grid used to line query windows up with stored ones
    subfp_top_k: int = SUBFP_TOP_K_PER_WINDOW
    pitch_hyps: tuple = PITCH_HYPOTHESIS_SEMITONES   # compensating warps, see binary_embed.py
    tempo_hyps: tuple = TEMPO_HYPOTHESES
    joint_hyps: tuple = ()              # ((semitone, tempo), ...) varispeed pairs, off by default
    fine_search: str = "all"            # search the fine index too: "all" | "identity" | "none"
    vote_max_dist: int | None = None
    verify_top_n: int = 5
    offset_search_frames: int = 16      # +- refinement around each vote-derived offset
    # Acceptance (decision_mode="verify"). Locked on the dev seed (7) positives
    # + 784 calibration negatives that are disjoint from the benchmark's held-out
    # seed-42 negatives (.agents/tasks/sonic-robustness/experiments.md): the best
    # 0-FP point was (0.25, 0.26, rev 0.0, contrast 0.08 if n<10, 1-window 0.06)
    # at 93.3 % dev edited top-1; every BER threshold is set one grid step
    # stricter as a safety margin, and reverse_margin raised to 0.01 after a
    # time-reversed steady-noise clip in the dev benchmark negatives passed at
    # +0.0007 (dev edited top-1 at the locked values: 91.9 %, 0/784 calibration FPs).
    accept_coarse_ber: float = 0.24
    accept_fine_ber: float = 0.25
    min_aligned_windows: int = 2
    single_window_ber: float | None = 0.05   # 1 aligned window (~2 s queries): coarse <= x and fine <= 2x
    short_alignment_windows: int = 4         # alignments over fewer windows than this ...
    short_alignment_fine_ber: float = 0.15   # ... need fine BER <= this (final calibration run, E7)
    min_contrast: float | None = 0.08        # misaligned-minus-aligned coarse BER ...
    contrast_below_windows: int = 10         # ... required only with fewer aligned windows than this
    # Contrast floor for alignments over >= contrast_below_windows windows. Added
    # after the held-out seed-42 benchmark accepted one non-DB music segment
    # (11 windows, fine BER 0.235, contrast 0.0085: a steady passage matching a
    # steady passage of another file at any offset). Value chosen on dev seed 7
    # + the 784 calibration negatives only: 0.04 is the largest grid point with
    # no dev cost (0.05 costs -0.4 pts); experiments.md E8.
    min_contrast_long: float | None = 0.04
    reverse_margin: float | None = 0.01      # forward score must beat the time-reversed hypothesis by this
    reverse_check_top: int = 3               # candidates (by score) that get the time-reversal check
    margin_ber: float = 0.0                  # min score gap to a second passing file (0 = off)
    strong_fine_ber: float = 0.20            # early exit: stop trying hypotheses once a pass is this strong
    dup_fine_ber: float = 0.12
    silent_window_rel_db: float = -40.0      # query windows this far below the loudest are ignored
    decision_mode: str = "verify"
    recal_score: float = 600.0
    recal_margin: float = 0.20


@dataclass
class MatchResult:
    file_id: int | None
    confident: bool
    stage_used: str          # "primary" or "primary+secondary"
    primary_confidence: float
    offset_confidence: float
    weighted_score: float
    margin: float
    escalated: bool
    ranked_ids: list = field(default_factory=list)   # top-K file_ids, for recall@K
    secondary_shortlist: list = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)
    # Verification evidence (decision_mode="verify"). file_id is the accepted
    # match (or None); candidate_evidence holds per-candidate aligned BER
    # evidence with "passes" = absolute acceptance thresholds met.
    hypothesis: dict = field(default_factory=dict)          # {"semitones": s, "tempo": r} of the accepted match
    evidence: dict = field(default_factory=dict)            # evidence of the accepted (or best) candidate
    candidate_evidence: dict = field(default_factory=dict)  # {file_id: evidence}
    match_kind: str | None = None                           # "duplicate" | "edited" | None


@dataclass
class DomainIndex:
    category: str
    landmark_index: LandmarkIndex
    binary_embed: BinaryEmbedIndex
    binary_hnsw: BinaryHNSWIndex
    subfp_index: SubFingerprintIndex
    subfp_fine_index: SubFingerprintIndex
    offset_conf_tau: float = DEFAULT_OFFSET_CONF_TAU
    margin_tau: float = DEFAULT_MARGIN_TAU
    min_score_tau: float = DEFAULT_MIN_SCORE_TAU
    min_accept_score: float = DEFAULT_MIN_ACCEPT_SCORE
    query_max_duration_sec: float = QUERY_MAX_DURATION_SEC
    skip_leading_silence: bool = True
    params: RetrievalParams = field(default_factory=RetrievalParams)

    def nbytes(self) -> int:
        return (self.landmark_index.nbytes() + self.binary_embed.nbytes()
                + self.binary_hnsw.nbytes() + self.subfp_index.nbytes()
                + self.subfp_fine_index.nbytes())



# RAM optimization: build_domain_index() used to accumulate every file's
# Landmark object, raw band-energy vector, and every window's sub-fingerprint
# code in one big Python list each, held for the ENTIRE corpus until the very
# end (measured: ~55MB of pure list/object overhead beyond what the final
# index structures need, at N=2000 files - see the RAM-scaling investigation
# for the measured breakdown). Processing the corpus in fixed-size CHUNKS and
# flattening/compacting each chunk into plain arrays immediately - discarding
# that chunk's raw per-file objects before moving to the next chunk - caps
# the "raw per-file object" memory at chunk size instead of corpus size,
# while producing byte-for-byte identical final index contents (chunking
# changes nothing about what's computed, only when intermediate objects are
# freed). Explicit `del` was verified NOT to shrink already-reached peak RSS
# (CPython/glibc don't return freed heap pages to the OS) - the only way to
# actually lower the peak is to never let it get that high in the first
# place, which is what per-chunk compaction does.
DOMAIN_INDEX_BUILD_CHUNK_SIZE = 64


def build_domain_index(category: str, entries: list[dict]) -> DomainIndex:
    file_ids = [e["file_id"] for e in entries]
    embed_idx = BinaryEmbedIndex()  # only used for its .binarize() helper here

    lidx = LandmarkIndex()
    lidx.file_id_to_local = {fid: i for i, fid in enumerate(file_ids)}
    lidx.local_to_file_id = np.array(file_ids, dtype=np.int64)
    lidx.n_files = len(file_ids)

    hash_chunks, lid_chunks, time_chunks = [], [], []
    raw_vec_chunks = []
    subfp_file_ids, subfp_code_chunks = [], []
    subfp_fine_file_ids, subfp_fine_code_chunks = [], []

    for chunk_start in range(0, len(entries), DOMAIN_INDEX_BUILD_CHUNK_SIZE):
        chunk = entries[chunk_start:chunk_start + DOMAIN_INDEX_BUILD_CHUNK_SIZE]
        chunk_file_ids = [e["file_id"] for e in chunk]
        chunk_landmarks, chunk_raw_vecs = [], []
        chunk_subfp_codes, chunk_subfp_fine_codes = [], []

        for e in chunk:
            y = load_audio(e["path"])
            mag, frame_times = stft_magnitude(y)
            chunk_landmarks.append(build_landmarks_from_magnitude(mag, frame_times))
            chunk_raw_vecs.append(band_energy_vector_from_magnitude(mag))

            for _start_time, raw_vec in sub_fingerprints_from_magnitude(mag, frame_times):
                subfp_file_ids.append(e["file_id"])
                chunk_subfp_codes.append(embed_idx.binarize(raw_vec))

            for _start_time, raw_vec in fine_sub_fingerprints_from_magnitude(mag, frame_times):
                subfp_fine_file_ids.append(e["file_id"])
                chunk_subfp_fine_codes.append(embed_idx.binarize(raw_vec))

        # Compact this chunk into flat/stacked arrays now, then let
        # chunk_landmarks/chunk_raw_vecs/chunk_subfp_codes go out of scope at
        # the top of the next loop iteration - only the compact per-chunk
        # arrays below survive into the accumulator lists.
        c_hash, c_lid, c_time = lidx.flatten_chunk(chunk_file_ids, chunk_landmarks)
        hash_chunks.append(c_hash)
        lid_chunks.append(c_lid)
        time_chunks.append(c_time)
        raw_vec_chunks.append(np.stack(chunk_raw_vecs))
        if chunk_subfp_codes:
            subfp_code_chunks.append(np.stack(chunk_subfp_codes))
        if chunk_subfp_fine_codes:
            subfp_fine_code_chunks.append(np.stack(chunk_subfp_fine_codes))

    lidx.build_from_flat(np.concatenate(hash_chunks), np.concatenate(lid_chunks), np.concatenate(time_chunks))
    del hash_chunks, lid_chunks, time_chunks

    raw_vecs = np.concatenate(raw_vec_chunks)
    del raw_vec_chunks

    bidx = BinaryEmbedIndex()
    bidx.build(file_ids, raw_vecs)

    hidx = BinaryHNSWIndex()
    hidx.build(file_ids, bidx.codes)

    sidx = SubFingerprintIndex()
    if subfp_code_chunks:
        sidx.build(subfp_file_ids, np.concatenate(subfp_code_chunks))

    sidx_fine = SubFingerprintIndex()
    if subfp_fine_code_chunks:
        sidx_fine.build(subfp_fine_file_ids, np.concatenate(subfp_fine_code_chunks))

    return DomainIndex(
        category=category,
        landmark_index=lidx,
        binary_embed=bidx,
        binary_hnsw=hidx,
        subfp_index=sidx,
        subfp_fine_index=sidx_fine,
        query_max_duration_sec=QUERY_MAX_DURATION_SEC_BY_DOMAIN.get(category, QUERY_MAX_DURATION_SEC),
    )


def _pairwise_overlap_score(query_lm: Landmark, cand_hashes: np.ndarray) -> float:
    if query_lm.hash.size == 0 or cand_hashes.size == 0:
        return 0.0
    cand_set = np.unique(cand_hashes)
    hits = np.isin(query_lm.hash, cand_set)
    return float(hits.sum()) / query_lm.hash.size


def _is_confident(top, second, idx: DomainIndex) -> tuple[bool, float]:
    if top is None:
        return False, 0.0
    margin = 1.0
    if second is not None and top.weighted_score > 0:
        margin = (top.weighted_score - second.weighted_score) / top.weighted_score
    # Both the offset-consistency path and the margin path used to accept a
    # candidate on RATIO evidence alone (e.g. offset_confidence=0.8 from just
    # 4-5 total hash matches). With so few matches the ratio is statistically
    # meaningless - measured: 23/100 pitch-shifted speech queries wrongly
    # short-circuited to the (wrong) primary answer this way, all with
    # weighted_score in the 2-15 range. Both paths now also require the
    # absolute min_score_tau floor, so a handful of coincidental matches can
    # no longer look "confident" regardless of what ratio they happen to form.
    confident = top.weighted_score >= idx.min_score_tau and (
        top.offset_confidence >= idx.offset_conf_tau or margin >= idx.margin_tau
    )
    return confident, margin


def query(y_full_path: str, idx: DomainIndex, top_k: int = 5) -> MatchResult:
    """Query from a file on disk (used for duplicate-audio benchmarking)."""
    import time

    t0 = time.perf_counter()
    if idx.skip_leading_silence:
        # Leading silence / offset (time-shift "pad") used to leave the fixed
        # t=0 window empty -> 0% retrieval. Only decodes more when the first
        # window starts silent (see io.load_query_window).
        y = load_query_window(y_full_path, idx.query_max_duration_sec)
    else:
        y = load_audio(y_full_path, max_duration=idx.query_max_duration_sec)
    load_ms = (time.perf_counter() - t0) * 1000
    return query_array(y, idx, top_k=top_k, load_ms=load_ms)


def query_array(y: np.ndarray, idx: DomainIndex, top_k: int = 5, load_ms: float = 0.0,
                 sr: int = None) -> MatchResult:
    """Query from an already-loaded/modified in-memory audio array (used for
    edited-audio benchmarking, where the modification is generated in memory).
    The domain's query_max_duration_sec cap is applied here too, so edited-query
    latency stays comparable to file-path queries regardless of how long the
    modified array is.
    """
    import time
    from sonic.config import ANALYSIS_SR

    sr = sr or ANALYSIS_SR
    max_samples = int(idx.query_max_duration_sec * sr)
    if idx.skip_leading_silence:
        y = skip_leading_silence(y, sr, max_samples)
    if y.size > max_samples:
        y = y[:max_samples]

    timing = {"load_ms": load_ms}

    t0 = time.perf_counter()
    mag, frame_times = stft_magnitude(y, sr)
    q_lm = build_landmarks_from_magnitude(mag, frame_times, sr)
    timing["primary_fingerprint_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cands = idx.landmark_index.query(q_lm, top_k=PRIMARY_POOL_K)
    timing["primary_search_ms"] = (time.perf_counter() - t0) * 1000

    top = cands[0] if cands else None
    second = cands[1] if len(cands) > 1 else None
    confident, margin = _is_confident(top, second, idx)

    params = idx.params
    if confident:
        accepted = top is not None and top.weighted_score >= idx.min_accept_score
        res = MatchResult(
            file_id=(top.file_id if accepted else None),
            confident=True,
            stage_used="primary",
            primary_confidence=top.query_confidence if top else 0.0,
            offset_confidence=top.offset_confidence if top else 0.0,
            weighted_score=top.weighted_score if top else 0.0,
            margin=margin,
            escalated=False,
            ranked_ids=[c.file_id for c in cands[:top_k]],
            timing_ms=timing,
        )
        if accepted and params.decision_mode == "verify":
            # Acceptance is unchanged (fast path: 0 wrong matches / 0 FPs in
            # the BEFORE benchmark). The aligned-code check here only decides
            # the "duplicate" vs "edited" label; it needs no index search.
            t0 = time.perf_counter()
            ev = _label_evidence(mag, sr, idx, top, frame_times.size)
            ev["passes"] = True
            ev["stage"] = "primary"
            kind = _kind(ev, params)
            ev["kind"] = kind
            res.match_kind = kind
            res.evidence = ev
            res.hypothesis = {"semitones": 0.0, "tempo": 1.0}
            res.candidate_evidence = {top.file_id: ev}
            timing["verify_ms"] = (time.perf_counter() - t0) * 1000
            timing["n_hypotheses"] = 1
        return res

    if params.decision_mode == "verify":
        return _verify_escalation(mag, frame_times, sr, idx, cands, margin, top_k, timing)

    # --- Escalate to secondary stage(s) ---
    t0 = time.perf_counter()
    embed_vec = band_energy_vector_from_magnitude(mag, sr)
    code = idx.binary_embed.binarize(embed_vec)
    shortlist = idx.binary_hnsw.search(code, top_k=SECONDARY_SHORTLIST_K)


    # Multi-window sub-fingerprint majority vote (Haitsma-Kalker style - see
    # fingerprint/binary_embed.py and index/subfingerprint_index.py). This is
    # what actually recovers pitch/tempo-shifted queries: the single global
    # secondary code above and the primary landmark hash both collapse under
    # phase-vocoder pitch/speed modification, but overlapping short windows
    # retain redundancy - some windows usually still match even when the
    # aggregate doesn't.
    subfp_windows = sub_fingerprints_from_magnitude(mag, frame_times, sr)
    subfp_votes = []
    if subfp_windows:
        query_codes = np.stack([idx.binary_embed.binarize(v) for _t, v in subfp_windows])
        subfp_votes = idx.subfp_index.vote(
            query_codes, top_k_per_window=SUBFP_TOP_K_PER_WINDOW, final_top_k=SUBFP_FINAL_TOP_K,
        )

    # Fine-resolution sub-fingerprint vote (see fingerprint/binary_embed.py's
    # SUBFP_FINE_N_BANDS comment): more discriminative than the coarse code
    # above, at the cost of being less pitch/tempo tolerant. Diagnosed on real
    # music trim_start/time_shift failures where the coarse code let an
    # unrelated file win by a razor-thin margin even though the true file also
    # matched closely - the fine code breaks that kind of near-tie decisively
    # for modifications that don't actually change pitch/tempo.
    fine_windows = fine_sub_fingerprints_from_magnitude(mag, frame_times, sr)
    subfp_fine_votes = []
    if fine_windows:
        fine_codes = np.stack([idx.binary_embed.binarize(v) for _t, v in fine_windows])
        subfp_fine_votes = idx.subfp_fine_index.vote(
            fine_codes, top_k_per_window=SUBFP_TOP_K_PER_WINDOW, final_top_k=SUBFP_FINAL_TOP_K,
        )

    timing["secondary_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    primary_ids = {c.file_id: c for c in cands}
    # Fuse primary landmark evidence with BOTH secondary evidence sources,
    # ADDITIVELY for every candidate that appears in any pool - not "primary
    # score if available, else re-derive from primary hashes again". An
    # earlier version fell back to a primary-hash overlap score for
    # secondary-only candidates, which is exactly the signal that's near-zero
    # for the queries this stage exists to rescue (heavy pitch/tempo shift):
    # landmark hashes barely survive a phase-vocoder pitch/tempo change (see
    # REPORT.md's pitch/tempo robustness analysis), so that fallback almost
    # never found the true candidate.
    combined_scores = {c.file_id: c.weighted_score for c in cands}

    n_bits = idx.binary_hnsw.n_bits
    for fid, hamming in shortlist:
        combined_scores[fid] = combined_scores.get(fid, 0.0) + float(n_bits - hamming)

    # Sub-fingerprint votes are summed over multiple windows, so they can
    # reach a much larger scale than a single-window Hamming score (up to
    # n_bits * n_query_windows) precisely when many windows agree - that
    # larger scale is intentional: strong multi-window agreement should be
    # able to override weak/absent primary and single-code evidence.
    for fid, vote_score in subfp_votes:
        combined_scores[fid] = combined_scores.get(fid, 0.0) + vote_score

    # Fine-code votes are scaled down to the coarse code's per-bit scale
    # (n_bits ratio) before fusing, rather than added raw. The fine code has
    # 3x the bits of the coarse one, so unscaled it would dominate the fused
    # score whenever it has ANY signal - including the noisy signal it
    # produces under pitch/speed shift (measured earlier: a 64-band code's
    # Hamming distance nearly doubled under a mere +1 semitone). Scaling keeps
    # its role as a tie-breaker on top of the coarse/pitch-tolerant code
    # rather than letting it override that code's judgment under pitch/tempo
    # modification, where it is the less trustworthy of the two.
    fine_scale = SUBFP_N_BANDS / SUBFP_FINE_N_BANDS
    for fid, vote_score in subfp_fine_votes:
        combined_scores[fid] = combined_scores.get(fid, 0.0) + vote_score * fine_scale

    # --- Targeted +-1 semitone pitch-hypothesis rescue (see
    # fingerprint/binary_embed.py's pitch_hypothesis_sub_fingerprints
    # docstring for the measured motivation/validation). Gated to only run
    # when the UNWARPED fusion above is already ambiguous (top candidate's
    # lead over the runner-up is thin) - i.e. this is a tie-breaker rescue,
    # not something that runs on every escalated query. Two reasons, both
    # measured:
    #   1. Latency: most escalated queries (background_noise, mp3, filtering,
    #      etc. that still needed to escalate) are already resolved
    #      correctly and decisively by the unwarped fusion above: recomputing
    #      two extra coarse + two extra fine sub-fingerprint codes and voting
    #      each against the full index for every one of them was measured to
    #      roughly double average escalated-query latency for no accuracy
    #      benefit on those conditions.
    #   2. Precision: gating on an existing decisive lead is what keeps this
    #      change a true no-op on already-correct calls. Without the gate, a
    #      small number of ALREADY-CORRECT +-0.5-semitone (and, rarely, other-
    #      modification) queries flipped to wrong: a competing candidate could
    #      occasionally score higher under an (incorrect, since the true shift
    #      wasn't really +-1 semitone) warp hypothesis than the true
    #      candidate's own already-strong, already-decisive unwarped lead,
    #      because the bonus rule only compares each candidate's hypothesis
    #      score against ITS OWN base score, not against the current leader.
    #      Skipping the whole rescue whenever the unwarped call is already
    #      decisive removes that failure mode entirely, since a decisive
    #      unwarped leader is never even put up for reconsideration.
    # PITCH_RESCUE_MARGIN_TAU chosen as roughly half of DEFAULT_MARGIN_TAU
    # (the primary stage's own "confident" margin) - deliberately more
    # permissive than the primary gate (this is a cheap secondary-stage
    # re-rank, not a full stage skip) while still excluding the clearly-
    # decisive majority of escalated calls from the extra computation.
    top_fid_prebonus = max(combined_scores, key=combined_scores.get) if combined_scores else None
    prebonus_top = combined_scores.get(top_fid_prebonus, 0.0) if top_fid_prebonus is not None else 0.0
    prebonus_second = max(
        (v for fid, v in combined_scores.items() if fid != top_fid_prebonus), default=0.0,
    )
    prebonus_margin = ((prebonus_top - prebonus_second) / prebonus_top) if prebonus_top > 0 else 0.0
    # Only reached in decision_mode "legacy" (reproduces the BEFORE behaviour,
    # where the rescue was hard-disabled) or "recal" (rescue on, comparison
    # baseline). The default "verify" mode handles pitch via per-hypothesis
    # verification in _verify_escalation instead of this additive bonus.
    run_pitch_rescue = params.decision_mode == "recal"

    pitch_hyp_votes_by_shift = {}
    pitch_hyp_fine_votes_by_shift = {}
    if run_pitch_rescue:
        hyp_windows_by_shift = pitch_hypothesis_sub_fingerprints(mag, frame_times, sr)
        for semi, hyp_windows in hyp_windows_by_shift.items():
            if not hyp_windows:
                continue
            hyp_codes = np.stack([idx.binary_embed.binarize(v) for _t, v in hyp_windows])
            pitch_hyp_votes_by_shift[semi] = dict(idx.subfp_index.vote(
                hyp_codes, top_k_per_window=SUBFP_TOP_K_PER_WINDOW, final_top_k=SUBFP_FINAL_TOP_K,
            ))
        if fine_windows:
            hyp_fine_windows_by_shift = pitch_hypothesis_fine_sub_fingerprints(mag, frame_times, sr)
            for semi, hyp_windows in hyp_fine_windows_by_shift.items():
                if not hyp_windows:
                    continue
                hyp_codes = np.stack([idx.binary_embed.binarize(v) for _t, v in hyp_windows])
                pitch_hyp_fine_votes_by_shift[semi] = dict(idx.subfp_fine_index.vote(
                    hyp_codes, top_k_per_window=SUBFP_TOP_K_PER_WINDOW, final_top_k=SUBFP_FINAL_TOP_K,
                ))

    # Pitch-hypothesis bonus: for each candidate, only the EXCESS of the best
    # compensated-shift vote over the already-fused unwarped coarse vote is
    # added - never the raw hypothesis vote itself. This makes the bonus
    # strictly additive evidence for candidates whose match genuinely improves
    # under a +-1 semitone compensation (the true match, when the query really
    # was shifted by about that much) while leaving every other candidate's
    # score (and therefore every already-passing, non-pitch-shifted condition)
    # unchanged whenever the hypotheses don't outperform the unwarped code -
    # measured to be the case for +-0.5 semitone and all other modification
    # types (see fingerprint/binary_embed.py's PITCH_HYPOTHESIS_SEMITONES
    # docstring).
    # Restricted to candidates that ALREADY appear in the unwarped coarse
    # shortlist (base_coarse_votes) - i.e. this can only strengthen a
    # candidate the unwarped evidence already considers plausible, never
    # introduce a brand-new candidate purely off hypothesis-vote noise.
    # Measured: without this restriction, a handful of candidates that show
    # up ONLY under a warped hypothesis (never in the unwarped vote at all -
    # a sign the "match" is coincidental to that specific compensation, not a
    # real one) picked up spurious bonuses and net accuracy went DOWN
    # slightly on music +-1 semitone despite helping several individual
    # cases; restricting to already-plausible candidates removed that
    # regression.
    if pitch_hyp_votes_by_shift:
        base_coarse_votes = dict(subfp_votes)
        all_hyp_fids = set(base_coarse_votes.keys())
        for votes in pitch_hyp_votes_by_shift.values():
            all_hyp_fids.update(votes.keys())
            
        for fid in all_hyp_fids:
            base = base_coarse_votes.get(fid, 0.0)
            best_hyp = max((votes.get(fid, 0.0) for votes in pitch_hyp_votes_by_shift.values()), default=0.0)
            bonus = max(0.0, best_hyp - base)
            if bonus > 0.0:
                combined_scores[fid] = combined_scores.get(fid, 0.0) + bonus

    # Same bonus treatment for the FINE code, scaled down by the same
    # fine_scale used for its base vote above (so a fine-code bonus can never
    # outweigh what an equivalently-strong coarse-code bonus would). This is
    # the component that turned out to matter most in practice: diagnosed on
    # real music +-1 semitone failures, the fine code (not the coarse one) was
    # the dominant source of wrong answers (favored the wrong file in 92% of
    # 39 diagnosed failures - see pitch_hypothesis_sub_fingerprints'
    # docstring). Same strict-improvement-only, already-plausible-candidate
    # restriction as the coarse bonus above, for the same reason (avoids
    # rewarding a candidate that only "matches" a specific warped hypothesis
    # by coincidence).
    if pitch_hyp_fine_votes_by_shift:
        base_fine_votes = dict(subfp_fine_votes)
        all_hyp_fine_fids = set(base_fine_votes.keys())
        for votes in pitch_hyp_fine_votes_by_shift.values():
            all_hyp_fine_fids.update(votes.keys())
            
        for fid in all_hyp_fine_fids:
            base = base_fine_votes.get(fid, 0.0)
            best_hyp = max((votes.get(fid, 0.0) for votes in pitch_hyp_fine_votes_by_shift.values()), default=0.0)
            bonus = max(0.0, best_hyp - base)
            if bonus > 0.0:
                combined_scores[fid] = combined_scores.get(fid, 0.0) + bonus * fine_scale

    timing["rerank_ms"] = (time.perf_counter() - t0) * 1000

    if not combined_scores:
        return MatchResult(
            file_id=None, confident=False, stage_used="primary+secondary",
            primary_confidence=0.0, offset_confidence=0.0, weighted_score=0.0,
            margin=margin, escalated=True,
            secondary_shortlist=shortlist, timing_ms=timing,
        )

    ranked = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
    best_fid = ranked[0]
    best_c = primary_ids.get(best_fid)
    score_top = combined_scores[best_fid]
    score_second = combined_scores[ranked[1]] if len(ranked) > 1 else 0.0
    sec_margin = ((score_top - score_second) / score_top) if score_top > 0 else 0.0

    if params.decision_mode == "recal":
        accepted = score_top >= params.recal_score and sec_margin >= params.recal_margin
    else:
        has_primary = best_c is not None and (best_c.weighted_score >= 12.0 or (best_c.offset_confidence >= 0.30 and best_c.weighted_score >= 4.0))
        has_secondary = (score_top >= 600.0 and sec_margin >= 0.20 and best_c is not None) or (score_top >= 900.0 and sec_margin >= 0.40)
        accepted = has_primary or has_secondary

    return MatchResult(
        file_id=(best_fid if accepted else None),
        confident=False,
        stage_used="primary+secondary",
        primary_confidence=(best_c.query_confidence if best_c else 0.0),
        offset_confidence=(best_c.offset_confidence if best_c else 0.0),
        weighted_score=(best_c.weighted_score if best_c else combined_scores[best_fid]),
        margin=sec_margin if best_c is None else margin,
        escalated=True,
        ranked_ids=ranked,
        secondary_shortlist=shortlist,
        timing_ms=timing,
    )


# ----------------------------------------------------------------------------
# Verification stage (decision_mode="verify")
#
# Why: the fused vote scores above are relative evidence whose scale depends on
# corpus size, query length and edit type, so no fixed gate on them works at
# 28k files (the 600/900 gate tuned on 200-file indexes rejected ~80% of
# correctly ranked speed edits). The acceptance statistic used here is the one
# from the Haitsma & Kalker (ISMIR 2002) paper the sub-fingerprint code is
# built on: the bit error rate between the query's codes and a candidate's
# STORED codes at one consistent time alignment. Unrelated audio sits far from
# 0 BER at every alignment; a true copy (even edited, once the matching
# pitch/tempo hypothesis compensates the edit) sits close to it. Candidates
# come from the same sub-fingerprint HNSW votes as before, per hypothesis.
# ----------------------------------------------------------------------------

_FRAME_DT = HOP_LENGTH / ANALYSIS_SR


def _hypotheses(params: RetrievalParams) -> list[tuple[float, float]]:
    """Identity first, then milder hypotheses before stronger ones (early exit
    usually stops at the matching one). Pitch values are compensating warps
    (see binary_embed.warp_magnitude_freq): +2 compensates a query pitched DOWN 2 st."""
    hyps = [(float(s), 1.0) for s in params.pitch_hyps if s]
    hyps += [(0.0, float(r)) for r in params.tempo_hyps if r != 1.0]
    hyps += [(float(s), float(r)) for s, r in params.joint_hyps]

    def severity(h):
        s, r = h
        return abs(s) + abs(12.0 * np.log2(r))

    return [(0.0, 1.0)] + sorted(hyps, key=severity)


class _QueryCodes:
    """Per-query cache of dense hypothesis codes (local to one query call)."""

    def __init__(self, mag, sr, params):
        self.mag, self.sr, self.params = mag, sr, params
        self.n_frames = mag.shape[1]
        self._lm = {}
        # per-frame energy, used to keep (near-)silent windows out of search
        # and verification: silent windows encode to all-zero codes that
        # "match" the silent parts of any unrelated file.
        self.frame_energy = np.concatenate([[0.0], np.cumsum((mag.astype(np.float64) ** 2).sum(axis=0))])

    def _log_mels(self, semi):
        if semi not in self._lm:
            m = warp_magnitude_freq(self.mag, semi) if semi else self.mag
            self._lm[semi] = (log_mel_bands(m, self.sr, SUBFP_N_BANDS), log_mel_bands(m, self.sr, SUBFP_FINE_N_BANDS))
        return self._lm[semi]

    def get(self, semi, tempo):
        return _HypCodes(self, semi, tempo)


class _HypCodes:
    """Codes of one (pitch, tempo) hypothesis: coarse codes on the dense
    verify grid (cheap, 17 bands), fine codes (49 bands, ~3x the cost) only
    where they are actually needed - the sparse search grid and the windows a
    candidate's best alignments use - computed on demand and cached."""

    def __init__(self, qc: _QueryCodes, semi, tempo):
        self.semi, self.tempo = semi, tempo
        self.lm_c, self.lm_f = qc._log_mels(semi)
        self.n_q = qc.n_frames
        self.w = max(2, int(round(SUBFP_WINDOW_FRAMES / tempo)))
        self.w_q = min(self.w, self.n_q)
        self.g = qc.params.verify_hop_frames
        self.starts, self.cc = windowed_codes(self.lm_c, self.w, self.g)
        ce = qc.frame_energy
        win_e = (ce[np.minimum(self.starts + self.w_q, self.n_q)] - ce[self.starts]) / max(self.w_q, 1)
        peak = float(win_e.max()) if win_e.size else 0.0
        self.active = (win_e > peak * 10.0 ** (qc.params.silent_window_rel_db / 10.0)) & (win_e > 1e-10)
        self._fine = {}

    def fine(self, j: np.ndarray) -> np.ndarray:
        j = np.asarray(j, dtype=np.int64)
        missing = [int(x) for x in np.unique(j) if int(x) not in self._fine]
        if missing:
            codes = windowed_codes_at(self.lm_f, self.w, self.starts[missing])
            for x, c in zip(missing, codes):
                self._fine[x] = c
        if j.size == 0:
            return np.empty((0, SUBFP_N_SEG * SUBFP_FINE_N_BANDS // 8), np.uint8)
        return np.stack([self._fine[int(x)] for x in j.reshape(-1)]).reshape(j.shape + (-1,))


def _align_eval(idx: DomainIndex, fid: int, centers, radius: int, hc: _HypCodes, params, n_refine: int = 2):
    """Best aligned BER evidence of `fid` (see _coarse_align / _finish_align)."""
    rows = _coarse_align(idx, fid, centers, radius, hc, params, n_refine)
    return _finish_align(idx, fid, rows, hc) if rows else None


def _coarse_scan(idx: DomainIndex, fid: int, offs: np.ndarray, hc: _HypCodes):
    """Coarse-code BER of candidate `fid` at each offset in `offs` (DB frames;
    DB frame = offset + tempo * query frame). Returns (ber, n, j, valid, ks) or
    None. Pairs involving a silent query window or a silent (all-zero) stored
    window are not counted."""
    s_c = idx.subfp_index
    count = s_c.file_count.get(int(fid))
    start_c = s_c.file_start.get(int(fid))
    if count is None or s_c.codes is None or offs.size == 0:
        return None
    g, tempo, cc = hc.g, hc.tempo, hc.cc
    span = max(0, hc.n_q - hc.w_q)
    j_max = len(hc.starts) - 1
    k_lo = max(0, int(np.floor(offs.min() / SUBFP_HOP_FRAMES)))
    k_hi = min(count - 1, int(np.ceil((offs.max() + tempo * span) / SUBFP_HOP_FRAMES)))
    if k_hi < k_lo:
        return None
    ks = np.arange(k_lo, k_hi + 1)
    q = (ks[None, :] * SUBFP_HOP_FRAMES - offs[:, None]) / tempo        # (M, K) query start frame
    j = np.clip(np.rint(q / g), 0, j_max).astype(np.int64)
    db_c = s_c.codes[start_c + ks]                                        # (K, 8)
    db_live = _POP[db_c].sum(axis=1) > 0
    valid = (q >= -0.5 * g) & (q <= span + 0.5 * g) & hc.active[j] & db_live[None, :]
    n = valid.sum(axis=1)
    d_c = _POP[np.bitwise_xor(cc[j], db_c[None, :, :])].sum(axis=2, dtype=np.int32)
    ber = np.where(valid, d_c, 0).sum(axis=1) / np.maximum(n, 1) / (cc.shape[1] * 8)
    return ber, n, j, valid, ks


def _coarse_align(idx: DomainIndex, fid: int, centers, radius: int, hc: _HypCodes, params, n_refine: int = 2):
    """Scans offsets centers +- radius with the coarse code and returns the
    best `n_refine` alignments as [(off, ber_c, n, j, ks)] for re-scoring with
    the fine code."""
    offs = np.unique(np.concatenate([np.round(c) + np.arange(-radius, radius + 1, hc.g) for c in centers])).astype(np.float64)
    scan = _coarse_scan(idx, fid, offs, hc)
    if scan is None:
        return []
    ber_c, n, j, valid, ks = scan
    if n.max() < 1:
        return []
    enough = n >= min(params.min_aligned_windows, int(n.max()))
    order = np.argsort(np.where(enough, ber_c, np.inf), kind="stable")[:n_refine]
    return [(float(offs[b]), float(ber_c[b]), int(n[b]), j[b][valid[b]], ks[valid[b]])
            for b in order.tolist() if enough[b]]


# Offsets (in stored windows, ~0.3 s each) used to measure how much better the
# best alignment is than misaligned ones. Genuine copies have a sharp BER
# minimum at the true offset; time-stationary content (steady noise, drones,
# their time-reversed versions) matches about equally well at any offset.
_CONTRAST_SHIFTS = np.array([-8, -6, -4, -3, 3, 4, 6, 8])


def _finish_align(idx: DomainIndex, fid: int, rows, hc: _HypCodes):
    """Re-scores coarse alignments with the fine code; returns the best evidence."""
    s_f = idx.subfp_fine_index
    start_f = s_f.file_start.get(int(fid)) if s_f.codes is not None else None
    count_f = s_f.file_count.get(int(fid), 0) if start_f is not None else 0
    best = None
    for off, ber_c, n, jv, kv in rows:
        ber_f = ber_c
        if start_f is not None:
            m = kv < count_f
            if m.any():
                fq = hc.fine(jv[m])
                fd = _POP[np.bitwise_xor(fq, s_f.codes[start_f + kv[m]])].sum(axis=1, dtype=np.int32)
                ber_f = float(fd.mean()) / (fq.shape[1] * 8)
        score = 0.5 * (ber_c + ber_f)
        if best is None or score < best["score"]:
            best = {"coarse_ber": round(ber_c, 4), "fine_ber": round(ber_f, 4), "score": round(score, 4),
                    "n_aligned": int(n), "offset_sec": round(off * _FRAME_DT, 3), "_off": off}
    if best is not None:
        scan = _coarse_scan(idx, fid, best["_off"] + _CONTRAST_SHIFTS * float(SUBFP_HOP_FRAMES), hc)
        contrast = None
        if scan is not None and (scan[1] >= 1).any():
            contrast = float(np.median(scan[0][scan[1] >= 1])) - best["coarse_ber"]
        best["contrast"] = round(contrast, 4) if contrast is not None else None
        del best["_off"]
    return best


def _reverse_segments(codes: np.ndarray, n_bands: int) -> np.ndarray:
    """Codes of the time-reversed windows: band-delta bits are per segment and
    the per-segment median ignores sample order, so reversing a window in time
    just reverses its segment order."""
    bits = np.unpackbits(codes, axis=1).reshape(codes.shape[0], SUBFP_N_SEG, n_bands)
    return np.packbits(bits[:, ::-1, :].reshape(codes.shape[0], -1), axis=1)


def _reverse_evidence(idx: DomainIndex, fid: int, hc: _HypCodes, params) -> float | None:
    """Best combined BER of `fid` under the hypothesis that the query is the
    candidate played BACKWARDS (DB frame = offset - tempo * query frame).
    Band-delta window codes are built from per-segment medians, so steady
    material (noise beds, drones, dense mixes) and its time-reversed version
    encode almost alike; a forward match that is not clearly better than the
    backward one is not evidence of a forward copy (measured: every
    calibration negative that passed the BER thresholds was a time-reversed
    copy of its own DB source)."""
    s_c, s_f = idx.subfp_index, idx.subfp_fine_index
    count = s_c.file_count.get(int(fid))
    if count is None or s_c.codes is None:
        return None
    start = s_c.file_start[int(fid)]
    step = max(1, int(round(params.query_hop_frames / hc.g)))
    sp = np.arange(0, len(hc.starts), step)
    sp = sp[hc.active[sp]]
    if sp.size == 0:
        return None
    rc = _reverse_segments(hc.cc, SUBFP_N_BANDS)
    lim = min(count, 20000)
    db = s_c.codes[start:start + lim]
    d = _POP[np.bitwise_xor(rc[sp][:, None, :], db[None, :, :])].sum(axis=2, dtype=np.int32)
    kbest = np.argmin(d, axis=1)
    offs_votes = kbest * float(SUBFP_HOP_FRAMES) + hc.tempo * (hc.starts[sp] + hc.w_q)
    centers = _offset_centers([(offs_votes, (64 - d[np.arange(sp.size), kbest]).astype(np.float64))])
    offs = np.unique(np.concatenate([np.round(c) + np.arange(-params.offset_search_frames,
                                                              params.offset_search_frames + 1, hc.g) for c in centers]))
    span = max(0, hc.n_q - hc.w_q)
    k_lo = max(0, int(np.floor((offs.min() - hc.tempo * (span + hc.w_q)) / SUBFP_HOP_FRAMES)))
    k_hi = min(lim - 1, int(np.ceil(offs.max() / SUBFP_HOP_FRAMES)))
    if k_hi < k_lo:
        return None
    ks = np.arange(k_lo, k_hi + 1)
    q = (offs[:, None] - ks[None, :] * SUBFP_HOP_FRAMES) / hc.tempo - hc.w_q
    j = np.clip(np.rint(q / hc.g), 0, len(hc.starts) - 1).astype(np.int64)
    dbk = s_c.codes[start + ks]
    valid = (q >= -0.5 * hc.g) & (q <= span + 0.5 * hc.g) & hc.active[j] & (_POP[dbk].sum(axis=1) > 0)[None, :]
    n = valid.sum(axis=1)
    if n.max() < 1:
        return None
    dc = _POP[np.bitwise_xor(rc[j], dbk[None, :, :])].sum(axis=2, dtype=np.int32)
    ber_c = np.where(valid, dc, 0).sum(axis=1) / np.maximum(n, 1) / (rc.shape[1] * 8)
    enough = n >= min(params.min_aligned_windows, int(n.max()))
    b = int(np.argmin(np.where(enough, ber_c, np.inf)))
    ber_f = float(ber_c[b])
    start_f = s_f.file_start.get(int(fid)) if s_f.codes is not None else None
    if start_f is not None:
        jv, kv = j[b][valid[b]], ks[valid[b]]
        m = kv < s_f.file_count[int(fid)]
        if m.any():
            fq = _reverse_segments(hc.fine(jv[m]), SUBFP_FINE_N_BANDS)
            fd = _POP[np.bitwise_xor(fq, s_f.codes[start_f + kv[m]])].sum(axis=1, dtype=np.int32)
            ber_f = float(fd.mean()) / (fq.shape[1] * 8)
    return 0.5 * (float(ber_c[b]) + ber_f)


def _kind(ev, params) -> str:
    """"duplicate" when the identity hypothesis matched with a near-zero fine
    BER (re-encodes, gain), otherwise "edited"."""
    is_identity = ev.get("semitones", 0.0) == 0.0 and ev.get("tempo", 1.0) == 1.0
    fine = ev.get("fine_ber")
    return "duplicate" if (is_identity and (fine is None or fine <= params.dup_fine_ber)) else "edited"


def _passes_abs(ev, params) -> bool:
    """Bit-error-rate acceptance on one candidate's aligned evidence."""
    if ev is None:
        return False
    n = ev["n_aligned"]
    if n >= params.min_aligned_windows:
        if ev["coarse_ber"] > params.accept_coarse_ber or ev["fine_ber"] > params.accept_fine_ber:
            return False
        if n < params.short_alignment_windows and ev["fine_ber"] > params.short_alignment_fine_ber:
            return False
        # a genuine copy has a sharp BER minimum at its offset; require it at every window count
        floor = params.min_contrast if n < params.contrast_below_windows else params.min_contrast_long
        if floor is not None and (ev.get("contrast") is None or ev["contrast"] < floor):
            return False
        return True
    if n >= 1 and params.single_window_ber is not None:
        return ev["coarse_ber"] <= params.single_window_ber and ev["fine_ber"] <= 2 * params.single_window_ber
    return False


def _passes(ev, params) -> bool:
    if not _passes_abs(ev, params):
        return False
    if (params.reverse_margin is not None and ev.get("reverse_score") is not None
            and ev["reverse_score"] - ev["score"] < params.reverse_margin):
        return False
    return True


def _label_evidence(mag, sr, idx: DomainIndex, top, n_q_frames) -> dict:
    params = idx.params
    hc = _QueryCodes(mag, sr, params).get(0.0, 1.0)
    if len(hc.starts) == 0:
        return {}
    ev = _align_eval(idx, top.file_id, [top.best_offset_sec / _FRAME_DT],
                     int(np.ceil(LANDMARK_OFFSET_BIN_SEC / _FRAME_DT)), hc, params)
    if ev is None:
        return {}
    ev.update({"semitones": 0.0, "tempo": 1.0})
    return ev


def _vote_offsets(sidx: SubFingerprintIndex, codes, starts_sparse, tempo, params, scale=1.0):
    """Candidate scores {fid: vote} and per-fid offset samples (DB frames)."""
    qw, lbl, fid, dist = sidx.search_matches(codes, params.subfp_top_k)
    if params.vote_max_dist is not None and fid.size:
        keep = dist <= params.vote_max_dist * (sidx.n_bits / 64.0)
        qw, lbl, fid, dist = qw[keep], lbl[keep], fid[keep], dist[keep]
    if fid.size == 0:
        return {}, {}
    off = sidx.window_pos[lbl].astype(np.float64) * SUBFP_HOP_FRAMES - tempo * starts_sparse[qw]
    sim = (sidx.n_bits - dist).astype(np.float64) * scale
    uf, inv = np.unique(fid, return_inverse=True)
    tot = np.bincount(inv, weights=sim)
    votes = dict(zip(uf.tolist(), tot.tolist()))
    order = np.argsort(inv, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(inv))])
    offsets = {int(uf[i]): (off[order[bounds[i]:bounds[i + 1]]], sim[order[bounds[i]:bounds[i + 1]]])
               for i in range(uf.size)}
    return votes, offsets


def _offset_centers(off_w_list) -> list[float]:
    offs = np.concatenate([o for o, _w in off_w_list])
    w = np.concatenate([w for _o, w in off_w_list])
    bins = np.floor(offs / SUBFP_HOP_FRAMES + 0.5).astype(np.int64)
    ub, inv = np.unique(bins, return_inverse=True)
    bw = np.bincount(inv, weights=w)
    order = np.argsort(-bw)
    centers = []
    for r, bi in enumerate(order[:2]):
        if r == 1 and bw[bi] < 0.5 * bw[order[0]]:
            break
        centers.append(float(np.median(offs[inv == bi])))
    return centers


def _verify_escalation(mag, frame_times, sr, idx: DomainIndex, cands, margin, top_k, timing) -> MatchResult:
    import time

    params = idx.params
    t0 = time.perf_counter()
    qc = _QueryCodes(mag, sr, params)
    step = max(1, int(round(params.query_hop_frames / params.verify_hop_frames)))
    fine_scale = SUBFP_N_BANDS / SUBFP_FINE_N_BANDS
    primary_by_fid = {c.file_id: c for c in cands}

    best_ev: dict[int, dict] = {}
    best_votes: dict[int, float] = {}
    vote_order: dict[int, float] = {}
    hcs = {}
    n_hyp = 0
    for semi, tempo in _hypotheses(params):
        hc = qc.get(semi, tempo)
        if len(hc.starts) == 0:
            break
        hcs[(semi, tempo)] = hc
        n_hyp += 1
        sp = np.arange(0, len(hc.starts), step)
        sp = sp[hc.active[sp]]
        if sp.size == 0:
            break
        votes_c, offs_c = _vote_offsets(idx.subfp_index, hc.cc[sp], hc.starts[sp], tempo, params)
        votes_f, offs_f = ({}, {})
        use_fine = params.fine_search == "all" or (params.fine_search == "identity" and semi == 0.0 and tempo == 1.0)
        if use_fine and idx.subfp_fine_index.index is not None:
            votes_f, offs_f = _vote_offsets(idx.subfp_fine_index, hc.fine(sp), hc.starts[sp], tempo, params, fine_scale)
        votes = dict(votes_c)
        for f, v in votes_f.items():
            votes[f] = votes.get(f, 0.0) + v
        cand_fids = sorted(votes, key=votes.get, reverse=True)[: params.verify_top_n]
        extra = []
        if semi == 0.0 and tempo == 1.0:
            extra = [c.file_id for c in cands[:3] if c.file_id not in cand_fids]
        for f, v in votes.items():
            vote_order[f] = max(vote_order.get(f, 0.0), v)

        coarse_rows = {}
        for f in cand_fids + extra:
            if f in votes:
                centers = _offset_centers([x for x in (offs_c.get(f), offs_f.get(f)) if x is not None])
                radius = params.offset_search_frames
            else:  # primary-only candidate: landmark offset histogram (0.5 s bins)
                centers = [primary_by_fid[f].best_offset_sec / _FRAME_DT]
                radius = int(np.ceil(LANDMARK_OFFSET_BIN_SEC / _FRAME_DT))
            rows = _coarse_align(idx, f, centers, radius, hc, params)
            # the fine code is only worth computing for alignments that can still pass
            rows = [r for r in rows if r[1] <= params.accept_coarse_ber]
            if rows:
                coarse_rows[f] = rows
        if coarse_rows:  # one batched fine-code computation for every alignment to re-score
            hc.fine(np.concatenate([r[3] for rows in coarse_rows.values() for r in rows]))
        for f, rows in coarse_rows.items():
            ev = _finish_align(idx, f, rows, hc)
            if ev is None:
                continue
            ev.update({"semitones": semi, "tempo": tempo})
            prev = best_ev.get(f)
            if prev is None or ev["score"] < prev["score"]:
                best_ev[f] = ev
                best_votes[f] = votes.get(f, 0.0)

        # early exit once a candidate is verified with a strong margin to spare
        if best_ev:
            b = min(best_ev, key=lambda f: best_ev[f]["score"])
            if _passes(best_ev[b], params) and best_ev[b]["fine_ber"] <= params.strong_fine_ber:
                break

    ranked_ev = sorted(best_ev, key=lambda f: best_ev[f]["score"])
    # time-reversal check for the leading candidates that could be accepted
    for f in ranked_ev[: params.reverse_check_top]:
        ev = best_ev[f]
        ev["reverse_score"] = None
        if params.reverse_margin is not None and _passes_abs(ev, params):
            r = _reverse_evidence(idx, f, hcs[(ev["semitones"], ev["tempo"])], params)
            ev["reverse_score"] = round(r, 4) if r is not None else None

    timing["verify_ms"] = (time.perf_counter() - t0) * 1000
    timing["secondary_ms"] = timing["verify_ms"]
    timing["n_hypotheses"] = n_hyp

    accepted_fid = None
    if ranked_ev:
        b = ranked_ev[0]
        ev_b = best_ev[b]
        if _passes(ev_b, params):
            accepted_fid = b
            second = next((f for f in ranked_ev[1:params.reverse_check_top] if _passes(best_ev[f], params)), None)
            if (second is not None and ev_b["fine_ber"] > params.strong_fine_ber
                    and best_ev[second]["score"] - ev_b["score"] < params.margin_ber):
                accepted_fid = None  # two different files verify about equally well: ambiguous
        if len(ranked_ev) > 1:
            ev_b["second_score"] = best_ev[ranked_ev[1]]["score"]
    for f, ev in best_ev.items():
        ev["passes"] = _passes(ev, params) and (f in ranked_ev[: params.reverse_check_top]
                                                or params.reverse_margin is None)
        ev["stage"] = "verify"
        ev["kind"] = _kind(ev, params)

    ranked = ranked_ev + [f for f in sorted(vote_order, key=vote_order.get, reverse=True) if f not in best_ev]
    ranked = ranked[:max(top_k, 1)]
    top_fid = accepted_fid if accepted_fid is not None else (ranked[0] if ranked else None)
    best_c = primary_by_fid.get(top_fid)
    ev_top = best_ev.get(top_fid, {})
    kind = _kind(ev_top, params) if accepted_fid is not None else None

    return MatchResult(
        file_id=accepted_fid,
        confident=False,
        stage_used="primary+secondary",
        primary_confidence=(best_c.query_confidence if best_c else 0.0),
        offset_confidence=(best_c.offset_confidence if best_c else 0.0),
        weighted_score=(best_c.weighted_score if best_c else float(best_votes.get(top_fid, 0.0))),
        margin=margin,
        escalated=True,
        ranked_ids=ranked,
        secondary_shortlist=[],
        timing_ms=timing,
        hypothesis=({"semitones": ev_top["semitones"], "tempo": ev_top["tempo"]} if ev_top else {}),
        evidence=dict(ev_top),
        candidate_evidence=best_ev,
        match_kind=kind,
    )
