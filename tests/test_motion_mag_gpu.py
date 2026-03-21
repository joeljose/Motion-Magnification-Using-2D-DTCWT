"""GPU unit tests for motion_mag.py — requires CUDA GPU + pytorch_wavelets."""

import os
import sys

import numpy as np
import pytest

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import motion_mag  # noqa: E402


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
        device = torch.device('cuda')

        phases = motion_mag._gpu_forward_pass(
            data, nlevels=nlevels, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        assert len(phases) == nlevels

    def test_phase_arrays_have_correct_frame_count(self):
        """Each phase array should have num_frames rows."""
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        device = torch.device('cuda')

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
        device = torch.device('cuda')

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
        device = torch.device('cuda')

        # Single batch (all 10 frames)
        phases_single = motion_mag._gpu_forward_pass(
            data, nlevels=3, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )

        # Force small batches by temporarily patching VRAM query
        # Use a wrapper that forces batch_size=3
        import unittest.mock
        # Return tiny free VRAM to force batch_size=3
        small_vram = data.shape[1] * data.shape[2] * 4 * 15 * 3  # 3 frames worth
        with unittest.mock.patch('torch.cuda.mem_get_info',
                                 return_value=(small_vram, small_vram)):
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
        device = torch.device('cuda')
        motion_mag._gpu_temporal_filter(phases, magnification=3.0, width=5.0,
                                        device=device)
        assert phases[0].shape == (20, 1000)
        assert phases[1].shape == (20, 250)

    def test_output_values_finite(self):
        """Filtered phases should contain no NaN or Inf."""
        phases = [np.random.randn(20, 500).astype(np.float32)]
        device = torch.device('cuda')
        motion_mag._gpu_temporal_filter(phases, magnification=3.0, width=5.0,
                                        device=device)
        assert np.all(np.isfinite(phases[0]))

    def test_dc_signal_preserved_in_interior(self):
        """A constant phase should be preserved away from boundaries.

        cuFFT uses zero-padding (not reflect), so boundary frames are affected.
        Interior frames should still be close to the original DC value.
        """
        phases = [np.ones((50, 100), dtype=np.float32) * 2.5]
        device = torch.device('cuda')
        motion_mag._gpu_temporal_filter(phases, magnification=3.0, width=5.0,
                                        device=device)
        # Check interior frames (skip boundary region)
        interior = phases[0][15:35, :]
        np.testing.assert_allclose(interior, 2.5, atol=0.1)


class TestGpuInversePass:
    """_gpu_inverse_pass should reconstruct frames from modified phases."""

    def test_output_shape_matches_input(self):
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        device = torch.device('cuda')
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
        device = torch.device('cuda')
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
        device = torch.device('cuda')
        nlevels = 3

        phases = motion_mag._gpu_forward_pass(
            data, nlevels=nlevels, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        result = motion_mag._gpu_inverse_pass(
            data, phases, nlevels=nlevels, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        # Should be close to input (phase extraction + reconstruction roundtrip)
        # Not exact due to atan2 phase extraction losing information at
        # near-zero magnitude coefficients. Check mean error is small.
        mean_err = np.mean(np.abs(data - result))
        assert mean_err < 0.5, f"Identity roundtrip mean error too large: {mean_err}"


class TestMagnifyMotionsGpu:
    """End-to-end GPU pipeline smoke tests."""

    def test_output_shape_and_dtype(self):
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        device = torch.device('cuda')
        result = motion_mag.magnify_motions_gpu(
            data, magnification=2.0, width=3, nlevels=2,
            biort='near_sym_b', qshift='qshift_b', device=device,
        )
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_output_values_finite(self):
        rng = np.random.RandomState(42)
        data = rng.rand(5, 16, 16).astype(np.float32)
        device = torch.device('cuda')
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
        device = torch.device('cuda')
        result = motion_mag.magnify_motions_gpu(
            data, magnification=3.0, width=3, nlevels=2,
            biort='near_sym_b', qshift='qshift_b', device=device,
        )
        # Should be roughly in the same ballpark (not all zeros or huge)
        assert result.mean() > 10
        assert result.mean() < 500


class TestGpuForwardPassDtype:
    """Separate class for dtype test to keep TestGpuForwardPass clean."""

    def test_phase_dtype_is_float32(self):
        """Phase arrays should be float32 (matching GPU precision)."""
        rng = np.random.RandomState(42)
        data = rng.rand(10, 32, 32).astype(np.float32)
        device = torch.device('cuda')

        phases = motion_mag._gpu_forward_pass(
            data, nlevels=3, biort='near_sym_b', qshift='qshift_b',
            device=device,
        )
        for phase in phases:
            assert phase.dtype == np.float32
