"""Tests for the GPU code path of motion_mag.py.

They need torch + pytorch_wavelets and run on CUDA when available, otherwise
on CPU tensors (CI installs the CPU build of PyTorch for this).
"""

import os
import sys
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytorch_wavelets = pytest.importorskip("pytorch_wavelets")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import motion_mag  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Tier 2: GPU forward pass
# ---------------------------------------------------------------------------

class TestGpuForwardPass:
    """_gpu_forward_pass should extract phase arrays from video frames."""

    def test_returns_phase_arrays_with_correct_count(self):
        """Should return one phase array per DTCWT level."""
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        nlevels = 3
        device = DEVICE

        phases = motion_mag._gpu_forward_pass(
            data, nlevels=nlevels, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        assert len(phases) == nlevels

    def test_phase_arrays_have_correct_frame_count(self):
        """Each phase array should have num_frames rows."""
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        device = DEVICE

        phases = motion_mag._gpu_forward_pass(
            data, nlevels=3, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        for level, phase in enumerate(phases):
            assert phase.shape[0] == 10, f"Level {level}: expected 10 frames"

    def test_phase_values_are_finite(self):
        """Phase arrays should contain no NaN or Inf."""
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        device = DEVICE

        phases = motion_mag._gpu_forward_pass(
            data, nlevels=3, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        for level, phase in enumerate(phases):
            assert np.all(np.isfinite(phase)), f"Level {level} has non-finite values"

    def test_batched_matches_single_batch(self):
        """Processing in small batches should match processing all at once."""
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        device = DEVICE

        # Single batch (all 10 frames)
        phases_single = motion_mag._gpu_forward_pass(
            data, nlevels=3, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )

        # Force small batches by reporting little free memory (batch size 2)
        small_vram = data.shape[1] * data.shape[2] * 4 * 15 * 3
        with mock.patch.object(motion_mag, '_free_memory', return_value=small_vram):
            phases_batched = motion_mag._gpu_forward_pass(
                data, nlevels=3, biort='near_sym_b', qshift='qshift_b',
                device=device,
            )

        for level in range(3):
            np.testing.assert_allclose(
                phases_single[level], phases_batched[level],
                atol=1e-5, rtol=1e-5,
                err_msg=f"Level {level} mismatch between batched and single",
            )

class TestGpuTemporalFilter:
    """_gpu_temporal_filter should modify phase arrays via cuFFT filtering."""

    def test_output_shapes_unchanged(self):
        """Phase arrays should keep their shape after filtering."""
        phases = [
            np.random.randn(20, 1000).astype(np.float32),
            np.random.randn(20, 250).astype(np.float32),
        ]
        device = DEVICE
        motion_mag._gpu_temporal_filter(phases, magnification=3.0, width=5.0,
                                        device=device)
        assert phases[0].shape == (20, 1000)
        assert phases[1].shape == (20, 250)

    def test_output_values_finite(self):
        """Filtered phases should contain no NaN or Inf."""
        phases = [np.random.randn(20, 500).astype(np.float32)]
        device = DEVICE
        motion_mag._gpu_temporal_filter(phases, magnification=3.0, width=5.0,
                                        device=device)
        assert np.all(np.isfinite(phases[0]))

    def test_dc_signal_preserved(self):
        """A constant phase must pass unchanged, boundary frames included
        (each chunk is padded along time before the FFT, like the CPU path)."""
        phases = [np.ones((50, 100), dtype=np.float32) * 2.5]
        motion_mag._gpu_temporal_filter(phases, magnification=3.0, width=5.0,
                                        device=DEVICE)
        # float32 FFT level (CUDA measured 4e-4); zero-padding gave ~0.03 at the ends
        np.testing.assert_allclose(phases[0], 2.5, atol=2e-3)


class TestGpuInversePass:
    """_gpu_inverse_pass should reconstruct frames from modified phases."""

    def test_output_shape_matches_input(self):
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        device = DEVICE
        nlevels = 3

        # Extract phases, then reconstruct without modification (identity test)
        phases = motion_mag._gpu_forward_pass(
            data, nlevels=nlevels, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        result = motion_mag._gpu_inverse_pass(
            data, phases, nlevels=nlevels, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        assert result.shape == data.shape

    def test_output_values_finite(self):
        rng = np.random.RandomState(42)
        data = rng.rand(5, 16, 16).astype(np.float32)
        device = DEVICE
        nlevels = 2

        phases = motion_mag._gpu_forward_pass(
            data, nlevels=nlevels, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        result = motion_mag._gpu_inverse_pass(
            data, phases, nlevels=nlevels, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        assert np.all(np.isfinite(result))

    def test_identity_roundtrip(self):
        """Forward → extract phases → reconstruct with same phases → close to input."""
        rng = np.random.RandomState(42)
        data = rng.rand(5, 32, 32).astype(np.float32)
        device = DEVICE
        nlevels = 3

        phases = motion_mag._gpu_forward_pass(
            data, nlevels=nlevels, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        result = motion_mag._gpu_inverse_pass(
            data, phases, nlevels=nlevels, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        # Unmodified phases must reconstruct the input to float32 precision
        # (measured mean error 1e-7 on [0, 1] data)
        mean_err = np.mean(np.abs(data - result))
        assert mean_err < 1e-5, f"Identity roundtrip mean error too large: {mean_err}"


class TestMagnifyMotionsGpu:
    """End-to-end GPU pipeline smoke tests."""

    def test_output_shape_and_dtype(self):
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        device = DEVICE
        result = motion_mag.magnify_motions_gpu(
            data, magnification=2.0, width=3, nlevels=2,
            biort='near_sym_b', qshift='qshift_b', device=device,
        )
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_output_values_finite(self):
        rng = np.random.RandomState(42)
        data = rng.rand(5, 16, 16).astype(np.float32)
        device = DEVICE
        result = motion_mag.magnify_motions_gpu(
            data, magnification=2.0, width=3, nlevels=2,
            biort='near_sym_b', qshift='qshift_b', device=device,
        )
        assert np.all(np.isfinite(result))

    def test_output_in_reasonable_range(self):
        """Output pixel values should be in a plausible range."""
        rng = np.random.RandomState(42)
        # Use 0-255 range like real video frames
        data = (rng.rand(10, 32, 32) * 255).astype(np.float32)
        device = DEVICE
        result = motion_mag.magnify_motions_gpu(
            data, magnification=3.0, width=3, nlevels=2,
            biort='near_sym_b', qshift='qshift_b', device=device,
        )
        # Should be roughly in the same ballpark (not all zeros or huge)
        assert result.mean() > 10
        assert result.mean() < 500

    @pytest.mark.parametrize("shape", [(6, 31, 33), (6, 32, 33)])
    def test_odd_frame_size(self, shape):
        data = (np.random.RandomState(0).rand(*shape) * 255).astype(np.float32)
        result = motion_mag.magnify_motions_gpu(
            data, magnification=3.0, width=3, nlevels=2, device=DEVICE,
        )
        assert result.shape == data.shape
        assert np.isfinite(result).all()


class TestGpuForwardPassDtype:
    """Separate class for dtype test to keep TestGpuForwardPass clean."""

    def test_phase_dtype_is_float32(self):
        """Phase arrays should be float32 (matching GPU precision)."""
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        device = DEVICE

        phases = motion_mag._gpu_forward_pass(
            data, nlevels=3, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        for phase in phases:
            assert phase.dtype == np.float32


# ---------------------------------------------------------------------------
# Out-of-memory handling
# ---------------------------------------------------------------------------

def _oom_above(limit):
    """A DTCWTForward that raises CUDA OOM for batches larger than `limit`."""
    class LimitedForward(pytorch_wavelets.DTCWTForward):
        def forward(self, x):
            if x.shape[0] > limit:
                raise torch.cuda.OutOfMemoryError("simulated")
            return super().forward(x)
    return LimitedForward


class TestOutOfMemoryRetry:
    def test_pipeline_completes_with_smaller_batches(self, capsys):
        data = (np.random.RandomState(0).rand(12, 32, 32) * 255).astype(np.float32)
        kwargs = dict(magnification=3.0, width=3, nlevels=2, device=DEVICE)
        expected = motion_mag.magnify_motions_gpu(data, **kwargs)
        with mock.patch.object(pytorch_wavelets, "DTCWTForward", _oom_above(3)):
            result = motion_mag.magnify_motions_gpu(data, **kwargs)
        assert "Out of GPU memory; retrying with batch size" in capsys.readouterr().out
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_single_frame_oom_is_raised(self):
        data = np.random.RandomState(0).rand(4, 16, 16).astype(np.float32)
        with mock.patch.object(pytorch_wavelets, "DTCWTForward", _oom_above(0)):
            with pytest.raises(torch.cuda.OutOfMemoryError):
                motion_mag.magnify_motions_gpu(data, width=3, nlevels=2, device=DEVICE)

    def test_run_batches_halves_until_it_fits(self):
        calls = []

        def process(start, end):
            calls.append((start, end))
            if end - start > 3:
                raise torch.cuda.OutOfMemoryError("simulated")

        motion_mag._run_batches(10, 10, process, "chunk")
        done = [c for c in calls if c[1] - c[0] <= 3]
        assert sorted(done) == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]


# ---------------------------------------------------------------------------
# Agreement with the CPU path
# ---------------------------------------------------------------------------

def test_gpu_path_agrees_with_cpu_path():
    """Both paths implement the same algorithm with the same filters, so on a
    moving textured clip they should agree closely. Measured ~115 dB PSNR on
    CUDA and on CPU tensors; float32 vs float64 sets the limit."""
    from scipy import ndimage
    base = ndimage.gaussian_filter(np.random.RandomState(0).rand(64, 64), 1.5) * 255
    spectrum = np.fft.fft2(base)
    shifts = 0.3 * np.sin(2 * np.pi * np.arange(24) / 8)
    data = np.stack([np.real(np.fft.ifft2(ndimage.fourier_shift(spectrum, (0, s))))
                     for s in shifts]).astype(np.float32)
    kwargs = dict(magnification=3.0, width=5, nlevels=3)
    cpu = motion_mag.magnify_motions(data, **kwargs)
    gpu = motion_mag.magnify_motions_gpu(data, device=DEVICE, **kwargs)
    mse = np.mean((cpu.astype(np.float64) - gpu) ** 2)
    assert 10 * np.log10(255 ** 2 / mse) >= 60
