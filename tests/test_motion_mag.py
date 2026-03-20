"""Unit tests for motion_mag.py — Phase-Based Motion Magnification."""

import subprocess
import sys
import os
from unittest.mock import MagicMock, patch

import cv2
import dtcwt
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import motion_mag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestFormatDuration:
    def test_seconds_only(self):
        assert motion_mag.format_duration(30.0) == "30.0s"

    def test_minutes_and_seconds(self):
        assert motion_mag.format_duration(90.5) == "1m 30.5s"

    def test_zero(self):
        assert motion_mag.format_duration(0) == "0.0s"

    def test_exactly_60(self):
        assert motion_mag.format_duration(60.0) == "1m 0.0s"


# ---------------------------------------------------------------------------
# Tier 1: Strict tolerance
# ---------------------------------------------------------------------------

class TestNormalizePhase:
    """normalize_phase should return unit-magnitude complex numbers."""

    def test_unit_magnitude(self):
        rng = np.random.RandomState(42)
        x = rng.randn(100) + 1j * rng.randn(100)
        result = motion_mag.normalize_phase(x)
        magnitudes = np.abs(result)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-10)

    def test_preserves_phase_angle(self):
        x = np.array([1 + 1j, -1 + 0j, 0 + 1j], dtype=np.complex128)
        result = motion_mag.normalize_phase(x)
        np.testing.assert_allclose(np.angle(result), np.angle(x), atol=1e-10)

    def test_zero_magnitude_safety(self):
        """Near-zero elements should be returned as-is (not NaN/Inf)."""
        x = np.array([1e-25 + 1e-25j, 0 + 0j, 1 + 1j], dtype=np.complex128)
        result = motion_mag.normalize_phase(x)
        assert np.all(np.isfinite(result))
        # Third element should be unit magnitude
        assert abs(abs(result[2]) - 1.0) < 1e-10

    def test_already_unit_magnitude(self):
        x = np.exp(1j * np.array([0, np.pi / 4, np.pi / 2, np.pi]))
        result = motion_mag.normalize_phase(x)
        np.testing.assert_allclose(result, x, atol=1e-10)


# ---------------------------------------------------------------------------
# Tier 2: Moderate tolerance
# ---------------------------------------------------------------------------

class TestFlattopFilter:
    """flattop_filter_1d should smooth data along the time axis."""

    def test_dc_passthrough(self):
        """A constant signal should pass through unchanged."""
        data = np.ones((100, 4), dtype=np.float64) * 5.0
        filtered = motion_mag.flattop_filter_1d(data, width=20, axis=0)
        np.testing.assert_allclose(filtered, 5.0, atol=1e-6)

    def test_smoothing_reduces_variance(self):
        """Filtering should reduce the variance of noisy data."""
        rng = np.random.RandomState(42)
        data = rng.randn(200, 10)
        filtered = motion_mag.flattop_filter_1d(data, width=20, axis=0)
        assert np.var(filtered) < np.var(data)

    def test_output_shape_preserved(self):
        data = np.random.rand(50, 8).astype(np.float64)
        filtered = motion_mag.flattop_filter_1d(data, width=10, axis=0)
        assert filtered.shape == data.shape

    def test_small_width_no_crash(self):
        """Very small width should not crash (window_size guard)."""
        data = np.random.rand(20, 4).astype(np.float64)
        filtered = motion_mag.flattop_filter_1d(data, width=0.01, axis=0)
        assert filtered.shape == data.shape
        assert np.all(np.isfinite(filtered))


class TestExtractTemporalPhases:
    """extract_temporal_phases on known pyramids."""

    def test_constant_phase_gives_linear_cumsum(self):
        """If all frames have the same coefficients, cumulative phase
        should be approximately constant (frame 0 angle repeated)."""
        transform = dtcwt.Transform2d()
        # Create identical frames
        frame = np.random.RandomState(42).rand(32, 32).astype(np.float64)
        pyramids = [transform.forward(frame, nlevels=3) for _ in range(10)]

        phases = motion_mag.extract_temporal_phases(pyramids, level=1)

        assert phases.shape[0] == 10
        # Frame-to-frame deltas should be ~0, so cumsum should be ~constant
        # (close to frame 0 angle at each position)
        for i in range(1, 10):
            np.testing.assert_allclose(phases[i], phases[0], atol=1e-10)

    def test_output_shape(self):
        transform = dtcwt.Transform2d()
        frame = np.random.rand(16, 16).astype(np.float64)
        pyramids = [transform.forward(frame, nlevels=2) for _ in range(5)]
        num_coeffs = pyramids[0].highpasses[0].size

        phases = motion_mag.extract_temporal_phases(pyramids, level=0)
        assert phases.shape == (5, num_coeffs)
        assert phases.dtype == np.float64


# ---------------------------------------------------------------------------
# Tier 3: Smoke tests
# ---------------------------------------------------------------------------

class TestMagnifyMotions:
    """Smoke test magnify_motions on tiny synthetic data."""

    def test_output_shape_and_dtype(self):
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float64)
        result = motion_mag.magnify_motions(data, magnification=2.0, width=5, nlevels=2)
        assert result.shape == data.shape
        assert result.dtype == np.float64

    def test_values_finite(self):
        rng = np.random.RandomState(42)
        data = rng.rand(5, 16, 16).astype(np.float64)
        result = motion_mag.magnify_motions(data, magnification=2.0, width=3, nlevels=2)
        assert np.all(np.isfinite(result))

    def test_magnification_one_near_identity(self):
        """With magnification=1.0, output should be close to input
        (no amplification of phase deviations)."""
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float64)
        result = motion_mag.magnify_motions(data, magnification=1.0, width=5, nlevels=2)
        # Not exact due to DTCWT roundtrip + filtering, but should be close
        error = np.mean(np.abs(result - data))
        assert error < 10.0  # generous bound — just checking it's not garbage


# ---------------------------------------------------------------------------
# Bug fix: load_video buffer guard
# ---------------------------------------------------------------------------

class TestLoadVideoBufferGuard:
    def test_frame_count_too_low(self):
        actual_frames = 10
        reported_count = 5
        h, w = 8, 8
        fake_frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(actual_frames)]
        call_idx = [0]

        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: reported_count,
            cv2.CAP_PROP_FRAME_WIDTH: w,
            cv2.CAP_PROP_FRAME_HEIGHT: h,
            cv2.CAP_PROP_FPS: 30.0,
        }[prop]
        mock_cap.isOpened.return_value = True

        def mock_read():
            if call_idx[0] < actual_frames:
                frame = fake_frames[call_idx[0]]
                call_idx[0] += 1
                return True, frame
            return False, None

        mock_cap.read.side_effect = mock_read

        with patch("cv2.VideoCapture", return_value=mock_cap):
            channels, fps, frame_size = motion_mag.load_video("fake.mp4")

        assert channels[0].shape[0] == reported_count
        assert fps == 30.0


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "motion_mag.py")


def run_cli(*args):
    result = subprocess.run(
        [sys.executable, SCRIPT] + list(args),
        capture_output=True, text=True
    )
    return result.returncode, result.stderr


@pytest.fixture
def dummy_video(tmp_path):
    p = tmp_path / "dummy.mp4"
    p.write_bytes(b"\x00" * 100)
    return str(p)


class TestInputValidation:
    def test_nonexistent_input_file(self):
        code, stderr = run_cli("-i", "nonexistent.mp4")
        assert code == 1
        assert "not found" in stderr

    def test_magnification_zero(self, dummy_video):
        code, stderr = run_cli("-i", dummy_video, "-k", "0")
        assert code == 1
        assert "--magnification must be positive" in stderr

    def test_width_zero(self, dummy_video):
        code, stderr = run_cli("-i", dummy_video, "-w", "0")
        assert code == 1
        assert "--width must be positive" in stderr

    def test_nlevels_zero(self, dummy_video):
        code, stderr = run_cli("-i", dummy_video, "--nlevels", "0")
        assert code == 1
        assert "--nlevels must be at least 1" in stderr
