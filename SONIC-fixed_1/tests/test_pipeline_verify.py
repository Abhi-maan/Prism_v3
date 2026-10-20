"""End-to-end verification stage on a small synthetic index (no DB needed)."""
import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _synth import SR, melody, write  # noqa: E402
from sonic.retrieval.pipeline import _passes_abs, build_domain_index, query


class PipelineVerifyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import librosa
        cls.tmp = Path(tempfile.mkdtemp(prefix="sonic_test_verify_"))
        cls.ys = [melody(100 + i, 10.0) for i in range(6)]
        entries = [{"file_id": i, "path": str(write(cls.tmp / f"db{i}.wav", y))} for i, y in enumerate(cls.ys)]
        cls.idx = build_domain_index("music", entries)
        y = cls.ys[2]
        cls.queries = {
            "copy": (write(cls.tmp / "q_copy.wav", y), "duplicate"),
            "pitch+2": (write(cls.tmp / "q_pitch.wav", librosa.effects.pitch_shift(y=y, sr=SR, n_steps=2.0)), "edited"),
            "tempo1.1": (write(cls.tmp / "q_tempo.wav", librosa.effects.time_stretch(y=y, rate=1.1)), "edited"),
            "pad5": (write(cls.tmp / "q_pad.wav", np.concatenate([np.zeros(5 * SR, np.float32), y])), None),
            "trim3": (write(cls.tmp / "q_trim.wav", y[3 * SR:]), None),
        }
        cls.unrelated = write(cls.tmp / "q_other.wav", melody(999, 10.0))
        cls.reversed = write(cls.tmp / "q_rev.wav", y[::-1].copy())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_edited_copies_accepted(self):
        for name, (path, kind) in self.queries.items():
            with self.subTest(name):
                res = query(str(path), self.idx)
                self.assertEqual(res.file_id, 2)
                self.assertTrue(res.candidate_evidence[2]["passes"])
                if kind is not None:
                    self.assertEqual(res.match_kind, kind)
        res = query(str(self.queries["pitch+2"][0]), self.idx)
        if not res.confident:
            self.assertEqual(res.hypothesis["semitones"], -2.0)  # compensating warp for a +2 st edit

    def test_unrelated_and_reversed_rejected(self):
        for path in (self.unrelated, self.reversed):
            with self.subTest(path.name):
                self.assertIsNone(query(str(path), self.idx).file_id)

    def test_legacy_mode_reproduces_old_path(self):
        legacy = dataclasses.replace(self.idx, params=dataclasses.replace(self.idx.params, decision_mode="legacy"))
        res = query(str(self.queries["tempo1.1"][0]), legacy)
        self.assertEqual(res.candidate_evidence, {})
        self.assertIsNone(res.match_kind)


def drone(seed: int, seconds: float, scale: float, noise: float) -> np.ndarray:
    """A steady harmonic drone (no onsets): its band-delta window codes are
    almost the same at every time offset."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    y = sum(a * np.sin(2 * np.pi * f * scale * t + rng.uniform(0, 6.28)) for f, a in
            ((110, 1), (165, .6), (220, .5), (330, .3), (440, .2), (660, .15), (880, .1), (1320, .08), (1760, .05)))
    y = y + noise * rng.standard_normal(t.size)
    return (y / np.max(np.abs(y))).astype(np.float32)


class SteadyContrastTest(unittest.TestCase):
    """B1 (review): a non-DB steady passage matched another file's steady passage
    over 11 aligned windows with near-zero misalignment contrast."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="sonic_test_drone_"))
        ys = [melody(100 + i, 10.0) for i in range(4)] + [drone(1, 12.0, 1.0, 0.3)]
        idx = build_domain_index("music", [{"file_id": i, "path": str(write(cls.tmp / f"db{i}.wav", y))}
                                           for i, y in enumerate(ys)])
        # 6 s window so the alignment spans >= 10 windows; min_score_tau forces the
        # verification stage (the landmark fast path is out of scope here)
        cls.idx = dataclasses.replace(idx, min_score_tau=1e9, query_max_duration_sec=6.0)
        cls.q = write(cls.tmp / "q_drone.wav", drone(2, 8.0, 1.03, 0.3))  # a different, detuned drone

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_steady_drone_rejected_with_many_windows(self):
        res = query(str(self.q), self.idx)
        self.assertIsNone(res.file_id)
        ev = res.candidate_evidence[4]
        self.assertGreaterEqual(ev["n_aligned"], self.idx.params.contrast_below_windows)
        self.assertLess(ev["contrast"], self.idx.params.min_contrast_long)
        self.assertFalse(ev["passes"])
        # the long-alignment contrast floor is what rejects it
        off = dataclasses.replace(self.idx, params=dataclasses.replace(
            self.idx.params, min_contrast_long=None, reverse_margin=None))
        self.assertEqual(query(str(self.q), off).file_id, 4)

    def test_passes_abs_long_alignment_contrast(self):
        p = self.idx.params
        held_out_fp = {"n_aligned": 11, "coarse_ber": 0.021, "fine_ber": 0.235, "contrast": 0.0085}
        self.assertFalse(_passes_abs(held_out_fp, p))
        self.assertFalse(_passes_abs({**held_out_fp, "contrast": None}, p))
        self.assertTrue(_passes_abs({**held_out_fp, "contrast": 0.10}, p))


if __name__ == "__main__":
    unittest.main()
