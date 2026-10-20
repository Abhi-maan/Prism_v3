"""Leading-silence skip for the query window."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _synth import SR, melody, write  # noqa: E402
from sonic.io import load_audio, load_query_window, skip_leading_silence


class QueryWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sonic_test_qwin_"))
        cls.y = melody(3, 10.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _content_start_matches(self, w):
        ref = self.y[:SR] / np.max(np.abs(self.y[:5 * SR]))
        corr = np.corrcoef(w[:SR], ref)[0, 1]
        self.assertGreater(corr, 0.95)

    def test_zero_pads_skipped(self):
        for pad in (5, 20):
            p = write(self.tmp / f"pad{pad}.wav", np.concatenate([np.zeros(pad * SR, np.float32), self.y]))
            w = load_query_window(str(p), 5.0)
            self.assertEqual(w.size, 5 * SR)
            self._content_start_matches(w)

    def test_quiet_noise_pad_skipped(self):
        noise = (10 ** (-60 / 20)) * np.random.default_rng(0).standard_normal(2 * SR).astype(np.float32)
        p = write(self.tmp / "noisepad.wav", np.concatenate([noise, self.y]))
        w = load_query_window(str(p), 5.0)
        self._content_start_matches(w)

    def test_content_at_start_untouched(self):
        p = write(self.tmp / "plain.wav", self.y)
        self.assertTrue(np.array_equal(load_query_window(str(p), 5.0), load_audio(str(p), max_duration=5.0)))
        arr = self.y.copy()
        self.assertTrue(np.array_equal(skip_leading_silence(arr, SR, 5 * SR), arr[:5 * SR]))

    def test_soft_onset_not_cut(self):
        soft = melody(4, 1.0) * (10 ** (-30 / 20))
        y = np.concatenate([soft, self.y]).astype(np.float32)
        out = skip_leading_silence(y, SR, 5 * SR)
        self.assertTrue(np.array_equal(out, y[:5 * SR]))


if __name__ == "__main__":
    unittest.main()
