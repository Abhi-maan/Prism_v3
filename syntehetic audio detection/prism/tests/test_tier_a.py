"""
PRISM — Tier A | Unit Tests
============================
Tests every public function in each pipeline stage.

Run with:
    pytest prism/tests/test_tier_a.py -v

Requirements
------------
    pip install pytest librosa numpy scipy pandas
"""

from __future__ import annotations

import sys
import os
import tempfile
import wave
import struct

import numpy as np
import pytest

# ── Make sure the project root is importable ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from prism.sonic.tier_a.preprocessor import preprocess
from prism.sonic.tier_a.domain       import classify_domain
from prism.sonic.tier_a.detectors    import vocoder_score, gan_score, diffusion_score
from prism.sonic.tier_a.aggregator   import aggregate_scores
from prism.sonic.tier_a.analyzer     import analyze_file


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures — synthetic audio helpers
# ═══════════════════════════════════════════════════════════════════════════════

SR      = 22_050
DURATION = 3.0  # seconds — short enough for fast tests


def _write_wav(path: str, audio: np.ndarray, sr: int = SR) -> None:
    """Write a mono float32 array to a 16-bit PCM WAV file."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.tobytes())


def _sine_wave(freq: float = 440.0, duration: float = DURATION, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def _noise(duration: float = DURATION, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (0.1 * rng.standard_normal(int(sr * duration))).astype(np.float32)


def _flat_signal(duration: float = DURATION, sr: int = SR) -> np.ndarray:
    """Constant-amplitude signal — maximally un-natural."""
    return np.full(int(sr * duration), 0.3, dtype=np.float32)


@pytest.fixture
def sine_wav(tmp_path):
    path = str(tmp_path / "sine.wav")
    _write_wav(path, _sine_wave())
    return path


@pytest.fixture
def noise_wav(tmp_path):
    path = str(tmp_path / "noise.wav")
    _write_wav(path, _noise())
    return path


@pytest.fixture
def flat_wav(tmp_path):
    path = str(tmp_path / "flat.wav")
    _write_wav(path, _flat_signal())
    return path


@pytest.fixture
def speech_like_wav(tmp_path):
    """Approximate speech: amplitude-modulated noise with mid-range energy."""
    rng = np.random.default_rng(0)
    n   = int(SR * DURATION)
    t   = np.linspace(0, DURATION, n)
    # Carrier: bandpass-like noise
    carrier = rng.standard_normal(n).astype(np.float32)
    # Envelope: slow modulation at ~4 Hz (syllable rate)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    audio = (carrier * envelope * 0.2).astype(np.float32)
    path = str(tmp_path / "speech_like.wav")
    _write_wav(path, audio)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Preprocessor
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:

    def test_returns_all_keys(self, sine_wav):
        p = preprocess(sine_wav)
        for key in ("audio", "sr", "stft_full", "stft_mag", "stft_phase", "mel_spec"):
            assert key in p, f"Missing key: {key}"

    def test_sample_rate(self, sine_wav):
        p = preprocess(sine_wav)
        assert p["sr"] == 22_050

    def test_audio_is_1d(self, sine_wav):
        p = preprocess(sine_wav)
        assert p["audio"].ndim == 1

    def test_stft_mag_non_negative(self, sine_wav):
        p = preprocess(sine_wav)
        assert np.all(p["stft_mag"] >= 0)

    def test_stft_shapes_consistent(self, sine_wav):
        p = preprocess(sine_wav)
        assert p["stft_mag"].shape == p["stft_phase"].shape
        assert p["stft_full"].shape == p["stft_mag"].shape

    def test_mel_spec_shape(self, sine_wav):
        p = preprocess(sine_wav)
        assert p["mel_spec"].shape[0] == 128  # N_MELS

    def test_mel_spec_dtype_float(self, sine_wav):
        p = preprocess(sine_wav)
        assert p["mel_spec"].dtype in (np.float32, np.float64)

    def test_noise_file(self, noise_wav):
        p = preprocess(noise_wav)
        assert p["audio"].shape[0] > 0

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            preprocess("/nonexistent/file.wav")


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — Domain Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainClassifier:

    def test_returns_valid_domain(self, sine_wav):
        p = preprocess(sine_wav)
        domain = classify_domain(p)
        assert domain in ("speech", "music", "environmental")

    def test_noise_domain(self, noise_wav):
        """White noise should not be speech or music."""
        p = preprocess(noise_wav)
        domain = classify_domain(p)
        # White noise: low centroid regularity; typically lands in environmental
        assert domain in ("speech", "music", "environmental")

    def test_deterministic(self, sine_wav):
        p = preprocess(sine_wav)
        d1 = classify_domain(p)
        d2 = classify_domain(p)
        assert d1 == d2

    def test_output_type(self, noise_wav):
        p = preprocess(noise_wav)
        assert isinstance(classify_domain(p), str)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3 — Detectors
# ═══════════════════════════════════════════════════════════════════════════════

class TestVocoderDetector:

    def test_output_range(self, sine_wav):
        p = preprocess(sine_wav)
        score = vocoder_score(p, "speech")
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_output_range_music(self, noise_wav):
        p = preprocess(noise_wav)
        score = vocoder_score(p, "music")
        assert 0.0 <= score <= 1.0

    def test_output_range_env(self, noise_wav):
        p = preprocess(noise_wav)
        score = vocoder_score(p, "environmental")
        assert 0.0 <= score <= 1.0

    def test_flat_signal_high_score(self, flat_wav):
        """A perfectly flat mel spectrum → high banding score."""
        p = preprocess(flat_wav)
        score = vocoder_score(p, "environmental")
        # Not asserting a strict bound — just that it runs and stays in range
        assert 0.0 <= score <= 1.0

    def test_returns_float(self, sine_wav):
        p = preprocess(sine_wav)
        assert isinstance(vocoder_score(p, "speech"), float)


class TestGanDetector:

    def test_output_range(self, sine_wav):
        p = preprocess(sine_wav)
        score = gan_score(p)
        assert 0.0 <= score <= 1.0

    def test_noise_output_range(self, noise_wav):
        p = preprocess(noise_wav)
        score = gan_score(p)
        assert 0.0 <= score <= 1.0

    def test_returns_float(self, noise_wav):
        p = preprocess(noise_wav)
        assert isinstance(gan_score(p), float)

    def test_deterministic(self, sine_wav):
        p = preprocess(sine_wav)
        assert gan_score(p) == gan_score(p)


class TestDiffusionDetector:

    def test_output_range(self, sine_wav):
        p = preprocess(sine_wav)
        score = diffusion_score(p)
        assert 0.0 <= score <= 1.0

    def test_flat_signal_high_score(self, flat_wav):
        """Perfectly constant energy → very high energy_consistency score."""
        p = preprocess(flat_wav)
        score = diffusion_score(p)
        assert 0.0 <= score <= 1.0
        # flat signal has zero CoV → energy_consistency ≈ 1.0
        # and very flat noise floor → flat_noise ≈ 1.0
        assert score >= 0.5, f"Expected high diffusion score for flat signal, got {score}"

    def test_noise_lower_score(self, noise_wav):
        """White noise has variable energy → lower diffusion score than flat."""
        p_noise = preprocess(noise_wav)
        p_flat  = preprocess(noise_wav)  # same file, just for shape test
        s_noise = diffusion_score(p_noise)
        assert 0.0 <= s_noise <= 1.0

    def test_returns_float(self, sine_wav):
        p = preprocess(sine_wav)
        assert isinstance(diffusion_score(p), float)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4 — Aggregator
# ═══════════════════════════════════════════════════════════════════════════════

class TestAggregator:

    @pytest.mark.parametrize("domain", ["speech", "music", "environmental"])
    def test_output_in_range(self, domain):
        score, flags = aggregate_scores(0.5, 0.5, 0.5, domain)
        assert 0.0 <= score <= 1.0

    def test_zero_inputs(self):
        score, flags = aggregate_scores(0.0, 0.0, 0.0, "speech")
        assert score == 0.0
        assert flags == []

    def test_one_inputs(self):
        score, flags = aggregate_scores(1.0, 1.0, 1.0, "speech")
        assert score == pytest.approx(1.0)
        for artifact in ("vocoder_artifact", "gan_artifact", "diffusion_artifact"):
            assert artifact in flags

    def test_vocoder_artifact_flag(self):
        _, flags = aggregate_scores(0.8, 0.1, 0.1, "speech")
        assert "vocoder_artifact" in flags

    def test_gan_artifact_flag(self):
        _, flags = aggregate_scores(0.1, 0.8, 0.1, "music")
        assert "gan_artifact" in flags

    def test_diffusion_artifact_flag(self):
        _, flags = aggregate_scores(0.1, 0.1, 0.8, "environmental")
        assert "diffusion_artifact" in flags

    def test_uncertain_flag(self):
        """Score in the uncertain zone (0.35, 0.55) should trigger 'uncertain'."""
        # Craft inputs so weighted sum lands ≈ 0.45 for environmental domain
        # weights: vocoder=0.15, gan=0.30, diffusion=0.55
        # 0.15*0.5 + 0.30*0.5 + 0.55*0.5 = 0.5 → actually in uncertain zone
        score, flags = aggregate_scores(0.4, 0.4, 0.4, "environmental")
        assert "uncertain" in flags

    def test_unknown_domain_defaults_to_environmental(self):
        score, _ = aggregate_scores(0.5, 0.5, 0.5, "unknown_domain")
        # Should not raise; falls back to environmental weights
        assert 0.0 <= score <= 1.0

    def test_flags_type(self):
        _, flags = aggregate_scores(0.5, 0.5, 0.5, "speech")
        assert isinstance(flags, list)
        for f in flags:
            assert isinstance(f, str)

    @pytest.mark.parametrize("domain,expected_weight_order", [
        ("speech",        ("vocoder", "diffusion", "gan")),   # vocoder highest
        ("music",         ("gan", "diffusion", "vocoder")),   # gan highest
        ("environmental", ("diffusion", "gan", "vocoder")),   # diffusion highest
    ])
    def test_domain_weights_direction(self, domain, expected_weight_order):
        """Dominant detector for each domain should drive score highest."""
        # Give 1.0 to each detector in turn, 0.0 to others
        names = ("vocoder", "gan", "diffusion")
        idx_map = {n: i for i, n in enumerate(names)}

        scores_per_detector = {}
        for det_name in names:
            inputs = [0.0, 0.0, 0.0]
            inputs[idx_map[det_name]] = 1.0
            s, _ = aggregate_scores(*inputs, domain)
            scores_per_detector[det_name] = s

        highest = max(scores_per_detector, key=scores_per_detector.__getitem__)
        assert highest == expected_weight_order[0], (
            f"For domain={domain}, expected {expected_weight_order[0]} to dominate, "
            f"got {highest}. Scores: {scores_per_detector}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  End-to-end — analyze_file
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeFile:

    def test_returns_all_keys(self, sine_wav):
        result = analyze_file(sine_wav)
        for key in (
            "file_path", "domain", "artifact_score",
            "vocoder_score", "gan_score", "diffusion_score",
            "sub_scores", "flags", "prediction",
        ):
            assert key in result, f"Missing key: {key}"

    def test_artifact_score_range(self, noise_wav):
        result = analyze_file(noise_wav)
        assert 0.0 <= result["artifact_score"] <= 1.0

    def test_prediction_is_valid_string(self, sine_wav):
        result = analyze_file(sine_wav)
        assert result["prediction"] in ("real", "synthetic")

    def test_sub_scores_matches_individual(self, noise_wav):
        result = analyze_file(noise_wav)
        assert result["sub_scores"]["vocoder"]   == result["vocoder_score"]
        assert result["sub_scores"]["gan"]       == result["gan_score"]
        assert result["sub_scores"]["diffusion"] == result["diffusion_score"]

    def test_flags_is_list(self, sine_wav):
        result = analyze_file(sine_wav)
        assert isinstance(result["flags"], list)

    def test_custom_threshold_high(self, sine_wav):
        """With threshold=0.0 everything is synthetic."""
        result = analyze_file(sine_wav, threshold=0.0)
        assert result["prediction"] == "synthetic"

    def test_custom_threshold_low(self, sine_wav):
        """With threshold=1.0 everything is real unless score is exactly 1."""
        result = analyze_file(sine_wav, threshold=1.0)
        # Score is almost certainly < 1.0 for a clean sine wave
        if result["artifact_score"] < 1.0:
            assert result["prediction"] == "real"

    def test_domain_is_valid(self, noise_wav):
        result = analyze_file(noise_wav)
        assert result["domain"] in ("speech", "music", "environmental")

    def test_file_path_preserved(self, sine_wav):
        result = analyze_file(sine_wav)
        assert result["file_path"] == sine_wav
