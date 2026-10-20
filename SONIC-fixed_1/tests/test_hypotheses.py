"""Pitch / tempo hypothesis codes re-align edited queries with the stored codes."""
import unittest

import numpy as np

from _synth import SR, melody  # noqa: E402
from sonic.fingerprint.binary_embed import (BinaryEmbedIndex, SUBFP_HOP_FRAMES, SUBFP_WINDOW_FRAMES, hypothesis_codes,
                                            hypothesis_sub_fingerprints, log_mel_bands, sub_fingerprints_from_magnitude,
                                            windowed_codes)
from sonic.fingerprint.landmarks import stft_magnitude
from sonic.index.subfingerprint_index import hamming_rows


def _db_codes(y):
    mag, ft = stft_magnitude(y)
    b = BinaryEmbedIndex()
    return np.stack([b.binarize(v) for _t, v in sub_fingerprints_from_magnitude(mag, ft)])


def _mean_dist(db, starts, codes, tempo):
    """Mean Hamming distance of stored window k vs the query window starting at k*hop/tempo."""
    d = []
    for k in range(2, db.shape[0]):
        qf = k * SUBFP_HOP_FRAMES / tempo
        j = int(np.argmin(np.abs(starts - qf)))
        if abs(starts[j] - qf) <= 1.0:
            d.append(int(hamming_rows(db[k:k + 1], codes[j:j + 1])[0]))
    return float(np.mean(d)), len(d)


class HypothesisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y = melody(5, 6.0)
        cls.db = _db_codes(cls.y)

    def test_vectorised_codes_match_index_codes(self):
        mag, _ = stft_magnitude(self.y)
        starts, codes = windowed_codes(log_mel_bands(mag), SUBFP_WINDOW_FRAMES, SUBFP_HOP_FRAMES)
        self.assertTrue(np.array_equal(codes, self.db[: len(starts)]))

    def test_pitch_hypothesis(self):
        import librosa
        q = librosa.effects.pitch_shift(y=self.y, sr=SR, n_steps=2.0)
        mag, _ = stft_magnitude(q.astype(np.float32))
        s0, c0 = hypothesis_codes(mag, semitone=0.0, query_hop_frames=1)
        s2, c2 = hypothesis_codes(mag, semitone=-2.0, query_hop_frames=1)  # compensates +2 st
        d0, n0 = _mean_dist(self.db, s0, c0, 1.0)
        d2, n2 = _mean_dist(self.db, s2, c2, 1.0)
        self.assertGreater(n2, 5)
        self.assertLess(d2, d0)

    def test_tempo_hypothesis(self):
        import librosa
        q = librosa.effects.time_stretch(y=self.y, rate=1.1)
        mag, _ = stft_magnitude(q.astype(np.float32))
        s0, c0 = hypothesis_codes(mag, tempo=1.0, query_hop_frames=1)
        s1, c1 = hypothesis_codes(mag, tempo=1.1, query_hop_frames=1)
        d0, _ = _mean_dist(self.db, s0, c0, 1.1)  # identity windows, placed at the true positions
        d1, n1 = _mean_dist(self.db, s1, c1, 1.1)
        self.assertGreater(n1, 5)
        self.assertLess(d1, d0)

    def test_list_api(self):
        mag, ft = stft_magnitude(self.y)
        out = hypothesis_sub_fingerprints(mag, ft, semitone=1.0, tempo=1.2)
        self.assertTrue(out and out[0][1].size == 64)


if __name__ == "__main__":
    unittest.main()
