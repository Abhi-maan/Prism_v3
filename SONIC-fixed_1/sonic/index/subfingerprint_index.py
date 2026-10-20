"""Multi-window sub-fingerprint index (Haitsma & Kalker, ISMIR 2002 style
majority-vote matching). Unlike BinaryHNSWIndex (one global code per file),
this stores MANY short overlapping window codes per indexed file and matches
a query's own window codes against all of them, voting per candidate file.

This is the fix for the redundancy gap diagnosed in pitch/tempo-shifted
queries: a single aggregate code has no fallback if scrambled by a
phase-vocoder pitch/speed modification, but with many independent overlapping
windows, some (e.g. more tonal/stable segments) usually still match even when
others don't. See fingerprint/binary_embed.py's module docstring for the
measured motivation and retrieval/pipeline.py for when this runs (only during
confidence-gated escalation, never on the fast/confident path).
"""

from dataclasses import dataclass, field

import faiss
import numpy as np

HNSW_M = 32
# HNSW search breadth, applied ONCE at build time (never per query: the app
# queries from several threads and faiss' efSearch is shared index state).
# faiss' default (16) was tuned for the 200-file research indexes; at the
# app's 1.9M-window unified index it misses true neighbours often enough to
# matter for heavily edited queries. See retrieval/pipeline.py RetrievalParams
# for the measured choice.
SUBFP_EF_SEARCH = 64

# popcount lookup for packed uint8 codes (Hamming distance = popcount(a ^ b))
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Hamming distance between two (n, n_bytes) packed uint8 arrays."""
    return _POPCOUNT8[np.bitwise_xor(a, b)].sum(axis=1, dtype=np.int32)


@dataclass
class SubFingerprintIndex:
    index: object = None
    window_file_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    n_bits: int = 0
    idf_weight: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    # Stored packed codes (same array the HNSW was built from - the archive
    # arrays are already in memory, so keeping a reference costs no extra RAM)
    # plus each file's contiguous label range. Windows of one file are always
    # added as one contiguous, time-ordered block (every builder in this repo
    # does so), so label - file_start[fid] is the window's index in the file
    # and its start time is that index * the sub-fingerprint hop. This is what
    # lets retrieval/pipeline.py verify a candidate by comparing the query's
    # codes with the candidate's codes at an estimated time alignment
    # (Haitsma & Kalker 2002 bit-error-rate test) instead of trusting votes.
    codes: np.ndarray = None
    file_start: dict = field(default_factory=dict)
    file_count: dict = field(default_factory=dict)
    window_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    ef_search: int = SUBFP_EF_SEARCH

    def build(self, window_file_ids: list[int], codes: np.ndarray, ef_search: int | None = None) -> None:
        if len(window_file_ids) == 0:
            return
        self.n_bits = codes.shape[1] * 8
        self.index = faiss.IndexBinaryHNSW(self.n_bits, HNSW_M)
        self.index.add(codes)
        self.window_file_ids = np.array(window_file_ids, dtype=np.int64)
        self.codes = np.ascontiguousarray(codes, dtype=np.uint8)
        self._compute_file_ranges()
        if ef_search is not None:
            self.ef_search = int(ef_search)
        self.set_ef_search(self.ef_search)
        self._compute_idf_weights(codes)

    def set_ef_search(self, ef_search: int) -> None:
        """Build-time / single-threaded setup only (shared faiss state)."""
        self.ef_search = int(ef_search)
        if self.index is not None:
            self.index.hnsw.efSearch = max(self.ef_search, 1)

    def _compute_file_ranges(self) -> None:
        fids = self.window_file_ids
        if fids.size == 0:
            self.file_start, self.file_count = {}, {}
            return
        change = np.flatnonzero(np.diff(fids)) + 1
        starts = np.concatenate([[0], change])
        counts = np.diff(np.concatenate([starts, [fids.size]]))
        run_fids = fids[starts]
        if np.unique(run_fids).size != run_fids.size:
            raise ValueError("sub-fingerprint windows of a file must be stored contiguously")
        self.file_start = dict(zip(run_fids.tolist(), starts.tolist()))
        self.file_count = dict(zip(run_fids.tolist(), counts.tolist()))
        # label -> window index inside its file (vectorised alignment lookups)
        self.window_pos = (np.arange(fids.size, dtype=np.int64) - np.repeat(starts, counts)).astype(np.int32)

    def _compute_idf_weights(self, codes: np.ndarray, threshold_frac: float = 0.15, k: int = 30) -> None:
        """Diagnosed on real failures: a few 'hub' window codes (generic
        content shared across many DIFFERENT files, e.g. similar drum/loop
        patterns in this music dataset) win votes for the wrong file even when
        the true file's own windows also match closely - the same hub
        pathology fixed for the primary landmark index via IDF weighting (see
        index/landmark_index.py), applied here the same way: down-weight
        window codes whose near-neighborhood spans many other files, one-time
        cost at index build, not part of query-time latency."""
        n = codes.shape[0]
        if n > 20000:
            # On large corpuses (e.g. >28,000 files with >1M subfingerprint codes),
            # an all-pairs N x N self-search in HNSW causes a massive multi-minute stall.
            # Default to uniform IDF weighting for instant sub-second index assembly.
            self.idf_weight = np.empty(0, dtype=np.float64)
            return

        k = min(k, n)
        distances, labels = self.index.search(codes, k)
        threshold = max(1, int(round(self.n_bits * threshold_frac)))

        commonness = np.zeros(n, dtype=np.int32)
        batch_size = 50000
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b_labels = labels[start:end]
            b_dists = distances[start:end]
            b_fids = self.window_file_ids[start:end, None]
            other_mask = (b_labels >= 0) & (b_labels != np.arange(start, end)[:, None]) & (self.window_file_ids[b_labels.clip(min=0)] != b_fids)
            close_mask = b_dists <= threshold
            commonness[start:end] = np.sum(other_mask & close_mask, axis=1)

        self.idf_weight = 1.0 / (1.0 + commonness.astype(np.float64))

    def vote(self, query_codes: np.ndarray, top_k_per_window: int = 5, final_top_k: int = 5,
             max_dist: int | None = None) -> list[tuple[int, float]]:
        """query_codes: (n_windows, n_bytes) packed codes, one per query window.
        Returns [(file_id, vote_score), ...] sorted descending. Each query
        window contributes its BEST match per candidate file (so a file with
        many windows near one query window isn't over-counted), summed across
        all query windows - this is the majority-vote step. Each match is
        weighted by the matched index entry's IDF weight (see
        _compute_idf_weights), down-weighting generic/common window codes.
        `max_dist` (optional) ignores neighbours farther than that many bits;
        None keeps the original behaviour."""
        if self.index is None or query_codes.shape[0] == 0 or len(self.window_file_ids) == 0:
            return []

        k = min(top_k_per_window, len(self.window_file_ids))
        distances, labels = self.index.search(query_codes, k)

        scores: dict[int, float] = {}
        for w in range(labels.shape[0]):
            best_per_file: dict[int, float] = {}
            for lbl, dist in zip(labels[w], distances[w]):
                if lbl < 0:
                    continue
                if max_dist is not None and dist > max_dist:
                    continue
                fid = int(self.window_file_ids[lbl])
                weight = self.idf_weight[lbl] if self.idf_weight.size else 1.0
                sim = float(self.n_bits - dist) * weight
                if fid not in best_per_file or sim > best_per_file[fid]:
                    best_per_file[fid] = sim
            for fid, sim in best_per_file.items():
                scores[fid] = scores.get(fid, 0.0) + sim

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:final_top_k]
        return ranked

    def search_matches(self, query_codes: np.ndarray, top_k_per_window: int = 20):
        """Vectorised neighbour lookup for the verification stage. Returns
        (query_window_idx, label, file_id, dist) flat int arrays, keeping only
        the best (smallest-distance) label per (query window, file)."""
        empty = (np.empty(0, np.int64),) * 4
        if self.index is None or query_codes.shape[0] == 0 or len(self.window_file_ids) == 0:
            return empty
        k = min(top_k_per_window, len(self.window_file_ids))
        distances, labels = self.index.search(np.ascontiguousarray(query_codes), k)
        qw = np.repeat(np.arange(labels.shape[0], dtype=np.int64), labels.shape[1])
        lbl = labels.reshape(-1).astype(np.int64)
        dist = distances.reshape(-1).astype(np.int64)
        ok = lbl >= 0
        qw, lbl, dist = qw[ok], lbl[ok], dist[ok]
        fid = self.window_file_ids[lbl]
        # faiss returns neighbours sorted by distance per row, so the first
        # occurrence of each (qw, fid) pair is that file's best label.
        key = qw * (int(fid.max()) + 1 if fid.size else 1) + fid
        _, first = np.unique(key, return_index=True)
        return qw[first], lbl[first], fid[first], dist[first]

    def vote_detailed(self, query_codes: np.ndarray, top_k_per_window: int = 20) -> dict:
        """{file_id: [(query_window_idx, label, dist), ...]} - the best label
        per (query window, file), i.e. the same matches vote() sums, kept
        individually so the caller can estimate a time alignment."""
        qw, lbl, fid, dist = self.search_matches(query_codes, top_k_per_window)
        out: dict[int, list] = {}
        for w, l, f, d in zip(qw.tolist(), lbl.tolist(), fid.tolist(), dist.tolist()):
            out.setdefault(f, []).append((w, l, d))
        return out

    def aligned_distances(self, fid: int, db_window_idx: np.ndarray, query_codes: np.ndarray) -> np.ndarray:
        """Hamming distance between query_codes[i] and the candidate file's
        stored window db_window_idx[i]; -1 where that window index is outside
        the file. Pure read of stored codes (thread-safe)."""
        db_window_idx = np.asarray(db_window_idx, dtype=np.int64)
        out = np.full(db_window_idx.shape, -1, dtype=np.int32)
        start = self.file_start.get(int(fid))
        if start is None or self.codes is None:
            return out
        count = self.file_count[int(fid)]
        ok = (db_window_idx >= 0) & (db_window_idx < count)
        if ok.any():
            out[ok] = hamming_rows(self.codes[start + db_window_idx[ok]], query_codes[ok])
        return out

    def nbytes(self) -> int:
        if self.index is None:
            return 0
        code_bytes = (self.n_bits // 8) * len(self.window_file_ids)
        graph_overhead = len(self.window_file_ids) * HNSW_M * 2 * 4
        return code_bytes + graph_overhead
