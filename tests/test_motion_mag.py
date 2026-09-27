"""Unit tests for motion_mag.py — Phase-Based Motion Magnification."""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import cv2
import dtcwt
import numpy as np
import pytest
from scipy import ndimage

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

    @pytest.mark.parametrize("width", [2, 7, 8, 20, 80, 81])
    def test_fft_path_matches_direct_convolution(self, width):
        """FFT and ndimage paths must agree everywhere, edges included."""
        rng = np.random.RandomState(0)
        data = np.cumsum(rng.randn(400, 5), axis=0)
        window = motion_mag._flattop_window(width)
        assert len(window) % 2 == 1
        with patch.object(motion_mag, "_FFT_THRESHOLD", 0):
            fft = motion_mag.flattop_filter_1d(data, width)
        direct = ndimage.convolve1d(data, window, axis=0, mode="reflect")
        np.testing.assert_allclose(fft, direct, atol=1e-10)

    def test_linear_drift_passes_without_lag(self):
        """A zero-phase filter leaves a linear ramp unchanged in the interior."""
        data = np.arange(400, dtype=np.float64)[:, None] * 0.05 * np.ones((1, 3))
        out = motion_mag.flattop_filter_1d(data, 80)
        half = len(motion_mag._flattop_window(80)) // 2
        np.testing.assert_allclose(out[half:-half], data[half:-half], atol=1e-10)

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
        assert phases.dtype == np.float32

    def test_zero_coefficients_give_finite_phase(self):
        """Exact-zero coefficients (e.g. a black border) must not give NaN."""
        transform = dtcwt.Transform2d()
        rng = np.random.RandomState(0)
        frames = rng.rand(6, 32, 32)
        frames[:, :, :8] = 0.0
        pyramids = [transform.forward(f, nlevels=2) for f in frames]
        assert np.any(pyramids[0].highpasses[0] == 0)

        phases = motion_mag.extract_temporal_phases(pyramids, level=0)
        assert np.isfinite(phases).all()


def test_magnify_motions_zero_region_is_finite():
    """A clip with an all-zero region must reconstruct without NaN."""
    rng = np.random.RandomState(0)
    data = rng.rand(8, 32, 32) * 255
    data[:, :8, :] = 0.0
    result = motion_mag.magnify_motions(data, magnification=3.0, width=3, nlevels=2)
    assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

class TestEstimateMemory:
    """estimate_memory should return CPU RAM and VRAM estimates in bytes."""

    def test_returns_cpu_and_vram(self):
        cpu_bytes, vram_bytes = motion_mag.estimate_memory(
            num_frames=100, height=480, width=640, nlevels=4, gpu=True,
        )
        assert cpu_bytes > 0
        assert vram_bytes > 0

    def test_cpu_only_returns_zero_vram(self):
        cpu_bytes, vram_bytes = motion_mag.estimate_memory(
            num_frames=100, height=480, width=640, nlevels=4, gpu=False,
        )
        assert cpu_bytes > 0
        assert vram_bytes == 0

    def test_more_frames_uses_more_memory(self):
        small_cpu, _ = motion_mag.estimate_memory(100, 480, 640, 4, gpu=False)
        large_cpu, _ = motion_mag.estimate_memory(500, 480, 640, 4, gpu=False)
        assert large_cpu > small_cpu

    def test_uses_real_coefficient_count(self):
        """The estimate must cover the complex64 pyramids: 8 bytes per
        coefficient, and DTCWT has about 2 coefficients per pixel."""
        n, h, w = 100, 480, 640
        cpu_bytes, _ = motion_mag.estimate_memory(n, h, w, 4, gpu=False)
        assert cpu_bytes > n * h * w * 2 * 8

    def test_gpu_ram_grows_slower_than_cpu_path(self):
        """GPU path keeps float32 phases, CPU keeps complex64 pyramids, so the
        per-frame cost is lower on GPU (its fixed torch overhead is higher)."""
        def per_frame(gpu):
            small, _ = motion_mag.estimate_memory(100, 480, 640, 4, gpu=gpu)
            large, _ = motion_mag.estimate_memory(500, 480, 640, 4, gpu=gpu)
            return (large - small) / 400
        assert per_frame(gpu=True) < per_frame(gpu=False)


# ---------------------------------------------------------------------------
# Tier 3: Smoke tests
# ---------------------------------------------------------------------------

class TestMagnifyMotions:
    """Smoke test magnify_motions on tiny synthetic data."""

    def test_accepts_biort_and_qshift_params(self):
        """magnify_motions should accept biort and qshift filter parameters."""
        rng = np.random.RandomState(42)
        data = rng.rand(5, 16, 16).astype(np.float64)
        result = motion_mag.magnify_motions(
            data, magnification=2.0, width=3, nlevels=2,
            biort='near_sym_a', qshift='qshift_a',
        )
        assert result.shape == data.shape
        assert np.all(np.isfinite(result))

    def test_different_filters_produce_different_output(self):
        """Changing biort/qshift should change the output."""
        rng = np.random.RandomState(42)
        data = rng.rand(5, 16, 16).astype(np.float64)
        result_a = motion_mag.magnify_motions(
            data, magnification=2.0, width=3, nlevels=2,
            biort='near_sym_a', qshift='qshift_a',
        )
        result_b = motion_mag.magnify_motions(
            data, magnification=2.0, width=3, nlevels=2,
            biort='near_sym_b', qshift='qshift_b',
        )
        assert not np.allclose(result_a, result_b, atol=1e-6)

    def test_output_shape_and_dtype(self):
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float64)
        result = motion_mag.magnify_motions(data, magnification=2.0, width=5, nlevels=2)
        assert result.shape == data.shape
        assert result.dtype == np.float32

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

class TestMagnifyMotionsUint8:
    def test_uint8_input_matches_float_input(self):
        """load_video returns uint8 channels; results must match float input."""
        rng = np.random.RandomState(0)
        data = (rng.rand(6, 16, 16) * 255).astype(np.uint8)
        a = motion_mag.magnify_motions(data, magnification=2.0, width=3, nlevels=2)
        b = motion_mag.magnify_motions(data.astype(np.float64), magnification=2.0,
                                       width=3, nlevels=2)
        np.testing.assert_allclose(a, b, atol=1e-3)


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


class TestSaveVideo:
    def test_raises_when_writer_cannot_open(self, tmp_path):
        channels = [np.zeros((2, 8, 8)) for _ in range(3)]
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = False
        with patch("cv2.VideoWriter", return_value=mock_writer):
            with pytest.raises(RuntimeError, match="could not open video writer"):
                motion_mag.save_video(channels, 30.0, str(tmp_path / "o.avi"), (8, 8))
        mock_writer.write.assert_not_called()

    def test_writes_readable_file(self, tmp_path):
        channels = [np.full((3, 16, 16), 100.0) for _ in range(3)]
        path = str(tmp_path / "o.avi")
        motion_mag.save_video(channels, 30.0, path, (16, 16))
        cap = cv2.VideoCapture(path)
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        cap.release()


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


def test_cli_reports_backend(tmp_path):
    """A real (tiny) run prints which backend is active."""
    src = str(tmp_path / "in.avi")
    writer = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"MJPG"), 30, (16, 16))
    rng = np.random.RandomState(0)
    for _ in range(4):
        writer.write((rng.rand(16, 16, 3) * 255).astype(np.uint8))
    writer.release()
    result = subprocess.run(
        [sys.executable, SCRIPT, "-i", src, "-o", str(tmp_path / "out.avi"),
         "-w", "2", "--nlevels", "1"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Backend:         CPU (dtcwt)" in result.stdout


class TestInputValidation:
    def test_gpu_requires_torch(self, dummy_video):
        """--gpu without torch should exit with clear error."""
        # Only meaningful on CPU image where torch is not installed
        try:
            import torch  # noqa: F401
            pytest.skip("torch is installed — test only applies to CPU image")
        except ImportError:
            pass
        code, stderr = run_cli("-i", dummy_video, "--gpu")
        assert code == 1
        assert "requires PyTorch" in stderr

    def test_gpu_flag_accepted(self, dummy_video):
        """CLI should accept --gpu without 'unrecognized arguments' error."""
        code, stderr = run_cli("-i", dummy_video, "--gpu")
        assert "unrecognized arguments" not in stderr

    def test_device_flag_accepted(self, dummy_video):
        """CLI should accept --device without 'unrecognized arguments' error."""
        code, stderr = run_cli("-i", dummy_video, "--device", "0")
        assert "unrecognized arguments" not in stderr

    def test_biort_flag_accepted(self, dummy_video):
        """CLI should accept --biort without error (validation only, no processing)."""
        code, stderr = run_cli("-i", dummy_video, "--biort", "near_sym_a", "--qshift", "qshift_a")
        # Will fail because dummy_video isn't a real video, but should NOT fail
        # on argument parsing — no "unrecognized arguments" error
        assert "unrecognized arguments" not in stderr

    def test_nonexistent_input_file(self):
        code, stderr = run_cli("-i", "nonexistent.mp4")
        assert code == 1
        assert "not found" in stderr

    def test_missing_output_directory(self, dummy_video, tmp_path):
        out = str(tmp_path / "no" / "such" / "dir" / "out.avi")
        code, stderr = run_cli("-i", dummy_video, "-o", out)
        assert code == 1
        assert "output directory does not exist" in stderr
        assert "Traceback" not in stderr

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
