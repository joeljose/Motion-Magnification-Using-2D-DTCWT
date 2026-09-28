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

def test_version_matches_version_file():
    with open(os.path.join(os.path.dirname(SCRIPT), "VERSION")) as f:
        assert motion_mag.__version__ == f.read().strip()


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


def _oscillating_texture(n=96, amp=0.1, period=16, size=64):
    """A smooth random texture shifted by amp * sin(2 pi t / period) pixels."""
    base = ndimage.gaussian_filter(np.random.RandomState(0).rand(size, size), 2) * 255
    spectrum = np.fft.fft2(base)
    shifts = amp * np.sin(2 * np.pi * np.arange(n) / period)
    frames = np.stack([np.real(np.fft.ifft2(ndimage.fourier_shift(spectrum, (0, s))))
                       for s in shifts])
    return frames, shifts


def _shift_gain(clip, shifts, mid=slice(24, 72)):
    """Regression slope of the per-frame horizontal shift of `clip` against
    `shifts`, estimated from the image gradient (linear for sub-pixel motion).
    `mid` picks frames away from the temporal filter's edges."""
    ref = clip[mid].mean(axis=0)
    gx = np.gradient(ref, axis=1)[8:-8, 8:-8]
    est = np.array([-(gx * (f - ref)[8:-8, 8:-8]).sum() / (gx * gx).sum()
                    for f in clip[mid]])
    s = shifts[mid] - shifts[mid].mean()
    return (est - est.mean()) @ s / (s @ s)


def _psnr(a, b):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(255 ** 2 / mse)


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

    @pytest.mark.parametrize("shape", [(6, 31, 33), (6, 32, 33)])
    def test_odd_frame_size(self, shape):
        """dtcwt pads odd sizes; the output must be cropped back."""
        data = np.random.RandomState(0).rand(*shape) * 255
        result = motion_mag.magnify_motions(data, magnification=3.0, width=3, nlevels=2)
        assert result.shape == data.shape
        assert np.isfinite(result).all()

    def test_odd_frame_size_identity(self):
        """With k=1 an odd-sized static clip reconstructs to the input."""
        frame = np.random.RandomState(0).rand(31, 33) * 255
        data = np.repeat(frame[None], 6, axis=0)
        result = motion_mag.magnify_motions(data, magnification=1.0, width=3, nlevels=2)
        np.testing.assert_allclose(result, data, atol=1e-2)

    def test_values_finite(self):
        rng = np.random.RandomState(42)
        data = rng.rand(5, 16, 16).astype(np.float64)
        result = motion_mag.magnify_motions(data, magnification=2.0, width=3, nlevels=2)
        assert np.all(np.isfinite(result))

    def test_magnification_one_near_identity(self):
        """With k=1 a static textured clip must come back unchanged."""
        texture = ndimage.gaussian_filter(np.random.RandomState(0).rand(64, 64), 1.5) * 255
        data = np.repeat(texture[None], 8, axis=0)
        result = motion_mag.magnify_motions(data, magnification=1.0, width=5, nlevels=3)
        assert _psnr(result, data) >= 60  # measured ~138 dB

    @pytest.mark.parametrize("k", [2, 4])
    def test_motion_is_magnified_k_times(self, k):
        """A texture moving by 0.1 px * sin(2 pi t / 16) must move about k times
        as far in the output."""
        frames, shifts = _oscillating_texture()
        out = motion_mag.magnify_motions(frames, magnification=k, width=20, nlevels=4)
        ratio = _shift_gain(out, shifts) / _shift_gain(frames, shifts) / k
        assert 0.8 <= ratio <= 1.2, ratio  # measured 0.95 (k=2), 0.93 (k=4)

    def test_golden_output(self):
        """Regression against a stored crop of face.mp4 and its output.
        Regenerate with scripts/make_golden.py only for intended changes."""
        golden = np.load(os.path.join(os.path.dirname(__file__), "data", "golden_face.npz"))
        sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT), "scripts"))
        import make_golden
        output = make_golden.magnify(golden["input"])
        assert _psnr(output, golden["output"]) >= 60
        # the golden output itself must differ from the input (motion was magnified)
        assert _psnr(golden["output"], golden["input"]) < 55


# ---------------------------------------------------------------------------
# Bug fix: load_video buffer guard
# ---------------------------------------------------------------------------

class TestBandMode:
    FPS = 30.0

    @pytest.mark.parametrize("freq, expected", [(1.2, (8, 12)), (0.3, (0, 2)), (5.0, (0, 2))])
    def test_only_in_band_motion_is_amplified(self, freq, expected):
        """--freq-low 0.8 --freq-high 2 -k 10: a 1.2 Hz oscillation is
        amplified about 10x, 0.3 Hz and 5 Hz less than 2x."""
        frames, shifts = _oscillating_texture(n=300, amp=0.03, period=self.FPS / freq)
        out = motion_mag.magnify_motions(frames, magnification=10, nlevels=4,
                                         band=(0.8 / self.FPS, 2.0 / self.FPS))
        mid = slice(50, 250)
        gain = _shift_gain(out, shifts, mid) / _shift_gain(frames, shifts, mid)
        assert expected[0] <= gain <= expected[1], gain

    def test_bandpass_keeps_only_the_band(self):
        t = np.arange(600)
        low_f, mid_f, high_f = 0.01, 0.05, 0.2  # cycles per frame
        x = (np.sin(2 * np.pi * low_f * t) + np.sin(2 * np.pi * mid_f * t)
             + np.sin(2 * np.pi * high_f * t))[:, None]
        y = motion_mag.bandpass_1d(x, 0.03, 0.08)[:, 0]
        expected = np.sin(2 * np.pi * mid_f * t)
        assert np.abs(y - expected)[100:500].max() < 0.05

    def test_flattop_band_matches_documented_defaults(self):
        low, high = motion_mag.flattop_band(80)
        assert low * 30 == pytest.approx(0.20, abs=0.01)
        assert high * 30 == pytest.approx(8.6, abs=0.1)


class TestPhaseSigma:
    def test_reduces_amplified_noise_in_low_contrast_texture(self):
        """Static low-contrast texture + sensor noise at k=10: amplitude-weighted
        phase smoothing must lower the output's temporal noise."""
        rng = np.random.RandomState(0)
        texture = 110 + 30 * ndimage.gaussian_filter(rng.randn(64, 64), 1.5)
        clip = texture + rng.normal(0, 2, (60, 64, 64))
        kwargs = dict(magnification=10, width=20, nlevels=4)
        plain = motion_mag.magnify_motions(clip, **kwargs)
        smooth = motion_mag.magnify_motions(clip, phase_sigma=1.0, **kwargs)
        mid = slice(10, 50)
        assert smooth[mid].std(axis=0).mean() < 0.9 * plain[mid].std(axis=0).mean()


class TestLumaMode:
    def _rgb_clip(self):
        frames, shifts = _oscillating_texture()
        # different brightness per channel, well inside 0-255 so nothing clips
        return [frames * scale * 0.8 + 20 for scale in (1.0, 0.8, 0.6)], shifts

    def test_luma_weights(self):
        grey = [np.full((2, 4, 4), 100.0)] * 3
        np.testing.assert_allclose(motion_mag.luma(grey), 100.0, rtol=1e-6)

    @pytest.mark.parametrize("k", [2, 4])
    def test_motion_is_magnified_k_times_in_luma_mode(self, k):
        channels, shifts = self._rgb_clip()
        out = motion_mag.magnify_luma(channels, lambda y: motion_mag.magnify_motions(
            y, magnification=k, width=20, nlevels=4))
        y_in, y_out = motion_mag.luma(channels), motion_mag.luma(out)
        ratio = _shift_gain(y_out, shifts) / _shift_gain(y_in, shifts) / k
        assert 0.8 <= ratio <= 1.2, ratio

    def test_chroma_is_unchanged(self):
        """I and Q must be kept (up to uint8 rounding) where nothing clips."""
        channels, _ = self._rgb_clip()
        out = motion_mag.magnify_luma(channels, lambda y: motion_mag.magnify_motions(
            y, magnification=4, width=20, nlevels=4))

        def iq(c):
            r, g, b = (np.asarray(x, np.float64) for x in c)
            return 0.596 * r - 0.274 * g - 0.322 * b, 0.211 * r - 0.523 * g + 0.312 * b
        for before, after in zip(iq(channels), iq(out)):
            assert np.abs(after - before).max() < 1.0


class TestParallelJobs:
    def test_result_does_not_depend_on_jobs(self):
        """Frames and coefficient columns are independent, so worker count
        must not change a single bit of the output."""
        data = (np.random.RandomState(0).rand(12, 40, 36) * 255).astype(np.uint8)
        kwargs = dict(magnification=3.0, width=5, nlevels=3)
        serial = motion_mag.magnify_motions(data, jobs=1, **kwargs)
        with patch.object(motion_mag, "_PHASE_CHUNK", 7):  # many chunks per level
            parallel = motion_mag.magnify_motions(data, jobs=3, **kwargs)
        np.testing.assert_array_equal(parallel, serial)

    def test_default_jobs(self):
        assert motion_mag._default_jobs(None) == min(motion_mag._DEFAULT_JOBS, os.cpu_count())
        assert motion_mag._default_jobs(1) == 1

    def test_estimate_grows_with_jobs(self):
        one, _ = motion_mag.estimate_memory(100, 480, 640, 4, jobs=1)
        four, _ = motion_mag.estimate_memory(100, 480, 640, 4, jobs=4)
        assert four > one


class TestMagnifyMotionsUint8:
    def test_uint8_input_matches_float_input(self):
        """load_video returns uint8 channels; results must match float input."""
        rng = np.random.RandomState(0)
        data = (rng.rand(6, 16, 16) * 255).astype(np.uint8)
        a = motion_mag.magnify_motions(data, magnification=2.0, width=3, nlevels=2)
        b = motion_mag.magnify_motions(data.astype(np.float64), magnification=2.0,
                                       width=3, nlevels=2)
        np.testing.assert_allclose(a, b, atol=1e-3)


def _mock_capture(actual_frames, reported_count, fps=30.0, h=8, w=8):
    """A cv2.VideoCapture stand-in yielding `actual_frames` frames."""
    frames = [np.full((h, w, 3), i, dtype=np.uint8) for i in range(actual_frames)]
    it = iter(frames)
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: reported_count,
        cv2.CAP_PROP_FPS: fps,
    }[prop]
    cap.read.side_effect = lambda: next(((True, f) for f in it), (False, None))
    return cap


class TestLoadVideo:
    @pytest.mark.parametrize("reported", [0, 3, 5, 10, 20])
    def test_reads_all_frames_whatever_the_reported_count(self, reported):
        with patch("cv2.VideoCapture", return_value=_mock_capture(10, reported)):
            channels, fps, frame_size = motion_mag.load_video("fake.mp4")
        assert channels[0].shape == (10, 8, 8)
        assert channels[0][:, 0, 0].tolist() == list(range(10))
        assert frame_size == (8, 8)
        assert fps == 30.0

    def test_unopenable_file_raises(self):
        cap = MagicMock()
        cap.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=cap):
            with pytest.raises(ValueError, match="cannot open video"):
                motion_mag.load_video("fake.mp4")

    def test_no_frames_raises(self):
        with patch("cv2.VideoCapture", return_value=_mock_capture(0, 10)):
            with pytest.raises(ValueError, match="no decodable frames"):
                motion_mag.load_video("fake.mp4")


def _run_main(argv, load_result):
    """Run main() in-process with load_video patched."""
    with patch.object(sys, "argv", ["motion_mag.py"] + argv), \
            patch.object(motion_mag, "load_video", return_value=load_result):
        motion_mag.main()


def _channels(n, size=16):
    rng = np.random.RandomState(0)
    return [(rng.rand(n, size, size) * 255).astype(np.uint8) for _ in range(3)]


class TestMainLoadChecks:
    def test_zero_fps_without_override_exits(self, dummy_video, capsys):
        with pytest.raises(SystemExit) as e:
            _run_main(["-i", dummy_video], (_channels(6), 0.0, (16, 16)))
        assert e.value.code == 1
        assert "pass --fps" in capsys.readouterr().err

    def test_fps_override_is_used(self, dummy_video, tmp_path):
        out = str(tmp_path / "out.avi")
        _run_main(["-i", dummy_video, "-o", out, "--fps", "25", "-w", "2", "--nlevels", "1"],
                  (_channels(6), 0.0, (16, 16)))
        cap = cv2.VideoCapture(out)
        assert cap.get(cv2.CAP_PROP_FPS) == 25.0
        cap.release()

    def test_too_few_frames_exits(self, dummy_video, capsys):
        with pytest.raises(SystemExit) as e:
            _run_main(["-i", dummy_video], (_channels(2), 30.0, (16, 16)))
        assert e.value.code == 1
        assert "at least 3 frames" in capsys.readouterr().err


class TestSaveVideo:
    def test_raises_when_writer_cannot_open(self, tmp_path):
        channels = [np.zeros((2, 8, 8)) for _ in range(3)]
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = False
        with patch("cv2.VideoWriter", return_value=mock_writer):
            with pytest.raises(RuntimeError, match="could not open video writer"):
                motion_mag.save_video(channels, 30.0, str(tmp_path / "o.avi"), (8, 8))
        mock_writer.write.assert_not_called()

    def test_rounds_instead_of_truncating(self, tmp_path):
        """Values just under an integer must round up, not drop a level."""
        channels = [np.full((1, 8, 8), 99.9999) for _ in range(3)]
        written = {}
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_writer.write.side_effect = lambda frame: written.setdefault("f", frame.copy())
        path = tmp_path / "o.avi"
        path.write_bytes(b"x")  # the size check after release needs a file
        with patch("cv2.VideoWriter", return_value=mock_writer):
            motion_mag.save_video(channels, 30.0, str(path), (8, 8))
        assert (written["f"] == 100).all()

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


@pytest.fixture
def tiny_video(tmp_path):
    """A real 12-frame 64x48 MJPG clip."""
    path = str(tmp_path / "in.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    rng = np.random.RandomState(0)
    for _ in range(12):
        writer.write((rng.rand(48, 64, 3) * 255).astype(np.uint8))
    writer.release()
    return path


def run_cli_full(*args):
    return subprocess.run([sys.executable, SCRIPT] + list(args), capture_output=True, text=True)


class TestCliEndToEnd:
    def _check_output(self, path):
        cap = cv2.VideoCapture(path)
        count = 0
        while cap.read()[0]:
            count += 1
        size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        assert count == 12
        assert size == (64, 48)

    def test_cpu_run_reports_backend_and_writes_output(self, tiny_video, tmp_path):
        out = str(tmp_path / "out.avi")
        result = run_cli_full("-i", tiny_video, "-o", out, "-w", "2", "--nlevels", "2")
        assert result.returncode == 0, result.stderr
        assert "Backend:         CPU (dtcwt, " in result.stdout
        self._check_output(out)

    def test_filter_flags_are_used(self, tiny_video, tmp_path):
        out = str(tmp_path / "out.avi")
        result = run_cli_full("-i", tiny_video, "-o", out, "-w", "2", "--nlevels", "2",
                              "--biort", "near_sym_a", "--qshift", "qshift_a", "--device", "0")
        assert result.returncode == 0, result.stderr
        assert "Biort filter:    near_sym_a" in result.stdout
        assert "Qshift filter:   qshift_a" in result.stdout
        self._check_output(out)

    def test_band_mode_run(self, tiny_video, tmp_path):
        out = str(tmp_path / "out.avi")
        result = run_cli_full("-i", tiny_video, "-o", out, "--nlevels", "2",
                              "--freq-low", "2", "--freq-high", "8")
        assert result.returncode == 0, result.stderr
        assert "Band:            2–8 Hz (ideal band-pass)" in result.stdout
        self._check_output(out)

    def test_default_band_is_printed(self, tiny_video, tmp_path):
        result = run_cli_full("-i", tiny_video, "-o", str(tmp_path / "o.avi"), "--nlevels", "2")
        assert result.returncode == 0, result.stderr
        assert "Band:            ~0.20–8.59 Hz (flat-top, width 80)" in result.stdout
        assert "deprecated" not in result.stderr

    def test_luma_mode_run(self, tiny_video, tmp_path):
        out = str(tmp_path / "out.avi")
        result = run_cli_full("-i", tiny_video, "-o", out, "-w", "2", "--nlevels", "2",
                              "--color-space", "yiq")
        assert result.returncode == 0, result.stderr
        assert "Color space:     yiq" in result.stdout
        assert "Processing luma (Y) channel" in result.stdout
        self._check_output(out)

    def test_gpu_run(self, tiny_video, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("pytorch_wavelets")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA GPU")
        out = str(tmp_path / "out.avi")
        result = run_cli_full("-i", tiny_video, "-o", out, "-w", "2", "--nlevels", "2", "--gpu")
        assert result.returncode == 0, result.stderr
        assert "Backend:         GPU" in result.stdout
        self._check_output(out)


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

    def test_invalid_device(self, tiny_video):
        torch = pytest.importorskip("torch")
        pytest.importorskip("pytorch_wavelets")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA GPU")
        code, stderr = run_cli("-i", tiny_video, "--gpu", "--device", "99")
        assert code == 1
        assert "--device 99 is not a valid CUDA device" in stderr
        assert "Traceback" not in stderr

    def test_corrupt_input_file(self, dummy_video):
        code, stderr = run_cli("-i", dummy_video)
        assert code == 1
        assert "Error:" in stderr
        assert "Traceback" not in stderr

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

    @pytest.mark.parametrize("flag", ["-k", "-w"])
    @pytest.mark.parametrize("value", ["nan", "inf"])
    def test_non_finite_values_rejected(self, dummy_video, flag, value):
        code, stderr = run_cli("-i", dummy_video, flag, value)
        assert code == 1
        assert "must be positive and finite" in stderr
        assert "Traceback" not in stderr

    def test_unknown_filter_name_rejected_before_loading(self, dummy_video):
        code, stderr = run_cli("-i", dummy_video, "--biort", "near_sym_x")
        assert code == 2
        assert "invalid choice: 'near_sym_x'" in stderr

    @pytest.mark.parametrize("args, message", [
        (["--freq-low", "1"], "must be given together"),
        (["--freq-low", "2", "--freq-high", "1"], "need 0 < --freq-low < --freq-high"),
        (["--freq-low", "1", "--freq-high", "2", "-w", "80"], "not both"),
    ])
    def test_band_arguments_validated(self, dummy_video, args, message):
        code, stderr = run_cli("-i", dummy_video, *args)
        assert code == 1
        assert message in stderr

    def test_band_above_nyquist(self, tiny_video):
        code, stderr = run_cli("-i", tiny_video, "--freq-low", "1", "--freq-high", "20")
        assert code == 1
        assert "above the Nyquist frequency (15 Hz at 30 fps)" in stderr

    def test_width_is_deprecated(self, tiny_video, tmp_path):
        result = run_cli_full("-i", tiny_video, "-o", str(tmp_path / "o.avi"),
                              "-w", "2", "--nlevels", "2")
        assert result.returncode == 0
        assert "-w/--width is deprecated" in result.stderr

    def test_phase_sigma_validated(self, dummy_video):
        code, stderr = run_cli("-i", dummy_video, "--phase-sigma", "-1")
        assert code == 1
        assert "--phase-sigma must be >= 0" in stderr

    def test_phase_sigma_is_cpu_only(self, dummy_video):
        code, stderr = run_cli("-i", dummy_video, "--phase-sigma", "1", "--gpu")
        assert code == 1
        assert "only supported on the CPU path" in stderr

    def test_jobs_zero(self, dummy_video):
        code, stderr = run_cli("-i", dummy_video, "--jobs", "0")
        assert code == 1
        assert "--jobs must be at least 1" in stderr

    def test_nlevels_zero(self, dummy_video):
        code, stderr = run_cli("-i", dummy_video, "--nlevels", "0")
        assert code == 1
        assert "--nlevels must be at least 1" in stderr
