"""SubFingerprintIndex: contiguous per-file layout, aligned lookups, unchanged vote()."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _synth import melody, write, SR  # noqa: E402  (also puts SONIC on sys.path)
from sonic.fingerprint.binary_embed import (BinaryEmbedIndex, SUBFP_HOP_FRAMES, SUBFP_HOP_SEC_EXACT,
                                            sub_fingerprints_from_magnitude)
from sonic.fingerprint.landmarks import stft_magnitude
from sonic.index.subfingerprint_index import SubFingerprintIndex
from sonic.retrieval.pipeline import build_domain_index


def _codes(y):
    mag, ft = stft_magnitude(y)
    b = BinaryEmbedIndex()
    return np.stack([b.binarize(v) for _t, v in sub_fingerprints_from_magnitude(mag, ft)])


def _reference_vote(sidx, query_codes, k, final_top_k):
    """The original vote() algorithm, re-implemented from its documented semantics."""
    distances, labels = sidx.index.search(query_codes, k)
    scores = {}
    for w in range(labels.shape[0]):
        best = {}
        for lbl, dist in zip(labels[w], distances[w]):
            if lbl < 0:
                continue
            fid = int(sidx.window_file_ids[lbl])
            wt = sidx.idf_weight[lbl] if sidx.idf_weight.size else 1.0
            sim = float(sidx.n_bits - dist) * wt
            if fid not in best or sim > best[fid]:
                best[fid] = sim
        for fid, sim in best.items():
            scores[fid] = scores.get(fid, 0.0) + sim
    return sorted(scores.items(), key=lambda kv: -kv[1])[:final_top_k]


class SubFpIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sonic_test_subfp_"))
        cls.ys = [melody(s, 8.0) for s in (11, 12, 13)]
        entries = [{"file_id": 100 + i, "path": str(write(cls.tmp / f"f{i}.wav", y))} for i, y in enumerate(cls.ys)]
        cls.idx = build_domain_index("music", entries)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_contiguous_ranges(self):
        s = self.idx.subfp_index
        self.assertEqual(sorted(s.file_start), [100, 101, 102])
        self.assertEqual(sum(s.file_count.values()), len(s.window_file_ids))
        for fid, start in s.file_start.items():
            block = s.window_file_ids[start:start + s.file_count[fid]]
            self.assertTrue(np.all(block == fid))
            self.assertEqual(int(s.window_pos[start]), 0)

    def test_non_contiguous_rejected(self):
        s = SubFingerprintIndex()
        codes = np.random.default_rng(0).integers(0, 255, (6, 8), dtype=np.uint8)
        with self.assertRaises(ValueError):
            s.build([1, 1, 2, 2, 1, 1], codes)

    def test_query_windows_advance_with_file_position(self):
        y = self.ys[1]
        start = int(round(2.1 / SUBFP_HOP_SEC_EXACT)) * SUBFP_HOP_FRAMES * 256  # on the stored window grid
        q = _codes(y[start:start + 4 * SR])[:-3]  # drop the partial trailing windows
        matches = self.idx.subfp_index.vote_detailed(q, top_k_per_window=5)
        rows = sorted(matches[101])
        pos = [lbl - self.idx.subfp_index.file_start[101] for _w, lbl, _d in rows]
        qws = [w for w, _l, _d in rows]
        first = int(round(2.1 / SUBFP_HOP_SEC_EXACT))
        self.assertGreaterEqual(len(rows), len(q) - 1)
        for w, p in zip(qws, pos):
            self.assertEqual(p, first + w)

    def test_aligned_distances(self):
        s = self.idx.subfp_index
        y = self.ys[2]
        k0 = 7
        q = _codes(y[k0 * SUBFP_HOP_FRAMES * 256:])[:8]
        d_true = s.aligned_distances(102, np.arange(k0, k0 + 8), q)
        d_wrong = s.aligned_distances(102, np.arange(k0 + 5, k0 + 13), q)
        self.assertLess(d_true.mean() / 64, 0.1)
        self.assertGreater(d_wrong.mean() / 64, 0.3)
        self.assertEqual(int(s.aligned_distances(102, np.array([10 ** 6]), q[:1])[0]), -1)

    def test_vote_unchanged(self):
        s = self.idx.subfp_index
        q = _codes(self.ys[0][SR:5 * SR])
        self.assertEqual(s.vote(q, 20, 20), _reference_vote(s, q, 20, 20))
        self.assertEqual(s.vote(q, 20, 20, max_dist=None), s.vote(q, 20, 20))


if __name__ == "__main__":
    unittest.main()
