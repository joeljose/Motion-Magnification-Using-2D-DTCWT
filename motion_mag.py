"""
Phase-Based Motion Magnification Using 2D DTCWT — CLI tool.

Amplifies subtle motions in video by manipulating the phase of complex wavelet
coefficients. Unlike Eulerian (color-based) methods, phase-based magnification
operates directly on motion information encoded in wavelet phase, enabling
larger amplification factors with fewer artifacts.

Based on: Anfinogentov & Nakariakov, "Motion Magnification in Coronal
Seismology", Solar Physics (2016), which adapts Wadhwa et al.'s phase-based
motion magnification (SIGGRAPH 2013) using 2D DTCWT instead of complex
steerable pyramids.
"""

__version__ = "1.1.0"

import argparse
import os
import sys
import time

import cv2
import numpy as np
from scipy import ndimage, signal

import dtcwt


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def normalize_phase(x):
    """Normalize complex array to unit magnitude, preserving phase.

    For each element, returns x / |x|. Elements with magnitude below 1e-20
    are left as-is to avoid division by zero.

    Args:
        x: Complex numpy array.

    Returns:
        Complex numpy array with unit magnitude (where |x| > 1e-20).
    """
    magnitude = np.abs(x)
    magnitude = np.where(magnitude > 1e-20, magnitude, 1.0)
    return x / magnitude


def extract_temporal_phases(pyramids, level):
    """Extract cumulative phase evolution across frames at a given DTCWT level.

    Computes frame-to-frame phase changes via complex division (more
    numerically stable than phase subtraction), then takes the cumulative
    sum to get absolute phase relative to frame 0.

    Memory-efficient: computes angle() per frame into a pre-allocated float64
    array, avoiding a full (num_frames, num_coeffs) complex intermediate.

    Args:
        pyramids: List of dtcwt Pyramid objects, one per frame.
        level: DTCWT decomposition level index.

    Returns:
        2D numpy array of shape (num_frames, num_coefficients) containing
        the cumulative phase angle at each coefficient position over time.
    """
    num_frames = len(pyramids)
    num_coeffs = pyramids[0].highpasses[level].size

    # Allocate float array directly — no full-size complex intermediate
    angles = np.empty((num_frames, num_coeffs), dtype=np.float64)

    prev_phase = normalize_phase(pyramids[0].highpasses[level].flatten())
    # Frame 0 stores absolute phase angle (needed for reconstruction)
    angles[0, :] = np.angle(prev_phase)

    for i in range(1, num_frames):
        curr_phase = normalize_phase(pyramids[i].highpasses[level].flatten())
        # Complex division gives frame-to-frame phase ratio;
        # suppress warnings from near-zero coefficients (result is harmless)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = curr_phase / prev_phase
        # np.angle returns 0 for NaN/Inf inputs, so zero-magnitude
        # coefficients contribute zero phase change as desired
        angles[i, :] = np.angle(ratio)
        prev_phase = curr_phase

    # Accumulate to get absolute phase relative to frame 0
    np.cumsum(angles, axis=0, out=angles)
    return angles


def _flattop_window(width):
    """Compute a normalized flat-top window for the given filter width."""
    window_size = max(1, round(width / 0.2327))
    window = signal.windows.flattop(window_size)
    return window / np.sum(window)


# Threshold: windows larger than this use FFT convolution (faster for large
# kernels due to O(n log n) vs O(n*k) complexity).
_FFT_THRESHOLD = 32


def flattop_filter_1d(data, width, axis=0, mode='reflect'):
    """Apply a flat-top window low-pass filter along the specified axis.

    Uses a flat-top window (scipy.signal.flattop) as a smoothing kernel.
    The window size is determined by width / 0.2327, where 0.2327 is the
    flat-top window's equivalent noise bandwidth in bins.

    For large windows (>32 samples), uses FFT-based convolution with
    reflect-padded boundaries for ~4x speedup. For small windows, uses
    direct convolution which is faster due to lower overhead.

    Args:
        data: Input numpy array.
        width: Filter width in frames. Controls the cutoff frequency —
            larger values produce more smoothing (lower cutoff).
        axis: Axis along which to filter (default: 0, the time axis).
        mode: Boundary handling mode for convolution (default: 'reflect').

    Returns:
        Filtered numpy array with same shape as input.
    """
    window = _flattop_window(width)

    if len(window) <= _FFT_THRESHOLD:
        return ndimage.convolve1d(data, window, axis=axis, mode=mode)

    # FFT path: manually reflect-pad, then use fftconvolve in chunks
    # for cache efficiency. Chunking along the non-convolution axis
    # keeps working sets in L2/L3 cache.
    pad_size = len(window) // 2
    n_along = data.shape[axis]
    n_across = data.size // n_along
    chunk_size = min(n_across, 10000)

    # Build axis-aware shapes for padding and kernel
    pad_widths = [(0, 0)] * data.ndim
    pad_widths[axis] = (pad_size, pad_size)
    kernel_shape = [1] * data.ndim
    kernel_shape[axis] = len(window)
    kernel = window.reshape(kernel_shape)

    # For 2D (frames, coeffs) with axis=0, chunk along axis=1
    if data.ndim == 2 and axis == 0:
        result = np.empty_like(data)
        for start in range(0, data.shape[1], chunk_size):
            end = min(start + chunk_size, data.shape[1])
            chunk = data[:, start:end]
            padded = np.pad(chunk, [(pad_size, pad_size), (0, 0)], mode=mode)
            conv = signal.fftconvolve(padded, kernel, mode='same', axes=0)
            result[:, start:end] = conv[pad_size:pad_size + n_along]
        return result

    # General fallback: no chunking
    padded = np.pad(data, pad_widths, mode=mode)
    conv = signal.fftconvolve(padded, kernel, mode='same', axes=axis)
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(pad_size, pad_size + n_along)
    return conv[tuple(slices)]


def estimate_memory(num_frames, height, width, nlevels, gpu=False):
    """Estimate peak CPU RAM and VRAM usage in bytes.

    Args:
        num_frames: Number of video frames.
        height: Frame height in pixels.
        width: Frame width in pixels.
        nlevels: Number of DTCWT decomposition levels.
        gpu: If True, estimate for GPU path (float32); otherwise CPU (float64).

    Returns:
        Tuple of (cpu_ram_bytes, vram_bytes).
    """
    bytes_per_pixel = 4 if gpu else 8  # float32 vs float64

    # Frames: num_frames × H × W × 3 channels
    frames_bytes = num_frames * height * width * 3 * bytes_per_pixel

    # Phase arrays (GPU path) or pyramid storage (CPU path)
    # DTCWT coefficients per level: (H/2^l, W/2^l, 6) — geometric series
    # sums to roughly 1.33× input size per channel
    coeff_factor = 1.33
    phase_bytes = int(num_frames * height * width * coeff_factor * bytes_per_pixel)
    # CPU path stores complex pyramids (2× for real+imag) per channel
    # GPU path stores float32 phase arrays per channel
    if gpu:
        # Phase arrays for all 3 channels (stored on CPU between passes)
        cpu_ram = frames_bytes + phase_bytes * 3
    else:
        # Pyramids stored as complex (2× the coefficient size) per channel,
        # but processed one channel at a time
        cpu_ram = frames_bytes + phase_bytes * 2  # complex = 2× float

    # VRAM estimate (GPU only)
    if gpu:
        # DTCWT batch: ~13 MB per frame at 528×592
        batch_vram = 10 * height * width * 13 * 4  # 10 frames × overhead
        # cuFFT chunk: largest level phase chunk + FFT buffers
        fft_vram = min(phase_bytes, 500 * 1024 * 1024)  # cap at 500 MB
        vram = batch_vram + fft_vram + 300 * 1024 * 1024  # 300 MB overhead
    else:
        vram = 0

    return cpu_ram, vram


def load_video(path):
    """Load a video file into separate R, G, B channel arrays.

    Each channel is stored as a float64 array of shape (num_frames, height,
    width). OpenCV reads in BGR order; channels are split accordingly.

    Args:
        path: Path to the video file.

    Returns:
        Tuple of (channels, fps, frame_size) where:
        - channels: list of 3 numpy arrays [R, G, B], each (N, H, W) float64
        - fps: frame rate as integer
        - frame_size: (width, height) tuple
    """
    cap = cv2.VideoCapture(path)
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Read all frames into a single array, then split channels
        frames = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
        i = 0
        t_start = time.time()
        while cap.isOpened():
            if i >= frame_count:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frames[i] = frame
            i += 1

            if (i) % max(1, frame_count // 10) == 0:
                elapsed = time.time() - t_start
                pct = i / frame_count
                eta = elapsed / pct * (1 - pct)
                print(f"  Reading: {i}/{frame_count} frames "
                      f"({pct:.0%}) — {format_duration(eta)} remaining")
    finally:
        cap.release()

    frames = frames[:i]

    # Split BGR channels to separate float64 arrays
    channels = [
        frames[:, :, :, 2].astype(np.float64),  # R
        frames[:, :, :, 1].astype(np.float64),  # G
        frames[:, :, :, 0].astype(np.float64),  # B
    ]
    del frames

    return channels, fps, (width, height)


def save_video(channels, fps, path, frame_size):
    """Save R, G, B channel arrays to an AVI video file.

    Recombines the three channels into BGR frames, clips to [0, 255],
    and writes using MJPG codec.

    Args:
        channels: List of 3 numpy arrays [R, G, B], each (N, H, W).
        fps: Frame rate for the output video.
        path: Output file path.
        frame_size: (width, height) tuple.
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size, True)
    try:
        frame_count = channels[0].shape[0]
        result = np.empty(
            (frame_count, channels[0].shape[1], channels[0].shape[2], 3),
            dtype=np.uint8
        )
        result[:, :, :, 2] = np.nan_to_num(np.clip(channels[0], 0, 255)).astype(np.uint8)
        result[:, :, :, 1] = np.nan_to_num(np.clip(channels[1], 0, 255)).astype(np.uint8)
        result[:, :, :, 0] = np.nan_to_num(np.clip(channels[2], 0, 255)).astype(np.uint8)

        for i in range(frame_count):
            writer.write(result[i])
    finally:
        writer.release()
    print(f"Output saved to {path}")


def _gpu_forward_pass(data, nlevels, biort, qshift, device):
    """GPU Pass 1: Batched forward DTCWT + phase extraction.

    Processes frames in GPU batches, extracting cumulative phase arrays
    per DTCWT level. Phase deltas are computed via vectorized conjugate
    multiply within each batch, with cross-batch boundary handling.

    Args:
        data: 3D numpy array (num_frames, H, W), float32.
        nlevels: Number of DTCWT decomposition levels.
        biort: Biorthogonal filter name.
        qshift: Quarter-shift filter name.
        device: torch.device for GPU.

    Returns:
        List of nlevels numpy arrays, each (num_frames, num_coeffs) float32,
        containing cumulative phase per coefficient.
    """
    import torch
    from pytorch_wavelets import DTCWTForward

    xfm = DTCWTForward(J=nlevels, biort=biort, qshift=qshift).to(device)
    num_frames = data.shape[0]

    # Determine batch size from available VRAM (70% of free)
    device_idx = device.index if device.index is not None else 0
    free_vram = torch.cuda.mem_get_info(device_idx)[0]
    frame_bytes = data.shape[1] * data.shape[2] * 4 * 15  # ~15x for DTCWT overhead
    batch_size = max(1, int(free_vram * 0.7 / frame_bytes))
    batch_size = min(batch_size, num_frames)

    # Get level shapes from a single-frame forward pass
    with torch.no_grad():
        test_frame = torch.from_numpy(data[0:1, np.newaxis, :, :]).to(device)
        _, Yh_test = xfm(test_frame)
        level_shapes = []
        level_coeffs = []
        for level in range(nlevels):
            shape = Yh_test[level].shape[2:]  # (6, H_l, W_l, 2)
            nc = int(np.prod(shape[:-1]))  # 6 * H_l * W_l
            level_shapes.append(shape)
            level_coeffs.append(nc)
        del test_frame, Yh_test
        torch.cuda.empty_cache()

    # Allocate phase delta arrays on CPU
    delta_arrays = [np.empty((num_frames, nc), dtype=np.float32)
                    for nc in level_coeffs]

    # Track previous frame's normalized coefficients for cross-batch boundary
    prev_n_real = [None] * nlevels
    prev_n_imag = [None] * nlevels

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        batch = torch.from_numpy(data[start:end, np.newaxis, :, :]).to(device)

        with torch.no_grad():
            _, Yh = xfm(batch)

        for level in range(nlevels):
            hp = Yh[level]  # (B, 1, 6, H, W, 2)
            c_real = hp[..., 0]  # (B, 1, 6, H, W)
            c_imag = hp[..., 1]

            # Normalize to unit magnitude
            mag = torch.sqrt(c_real ** 2 + c_imag ** 2)
            mag = torch.where(mag > 1e-20, mag, torch.ones_like(mag))
            n_real = c_real / mag
            n_imag = c_imag / mag

            # Frame 0 (global): absolute phase
            if start == 0:
                phase0 = torch.atan2(n_imag[0:1], n_real[0:1])
                delta_arrays[level][0] = phase0.reshape(1, -1).cpu().numpy()
                intra_start = 1
            else:
                # Cross-batch boundary: first frame vs prev batch's last frame
                pr = prev_n_real[level]
                pi = prev_n_imag[level]
                boundary_real = n_real[0:1] * pr + n_imag[0:1] * pi
                boundary_imag = n_imag[0:1] * pr - n_real[0:1] * pi
                delta_arrays[level][start] = (
                    torch.atan2(boundary_imag, boundary_real)
                    .reshape(1, -1).cpu().numpy()
                )
                intra_start = 1

            # Intra-batch deltas (vectorized)
            if end - start > intra_start:
                idx = intra_start
                pr = n_real[idx:] * n_real[idx - 1:-1] + n_imag[idx:] * n_imag[idx - 1:-1]
                pi = n_imag[idx:] * n_real[idx - 1:-1] - n_real[idx:] * n_imag[idx - 1:-1]
                deltas = torch.atan2(pi, pr)
                delta_arrays[level][start + idx:end] = (
                    deltas.reshape(end - start - idx, -1).cpu().numpy()
                )

            # Save last frame for next batch boundary
            prev_n_real[level] = n_real[-1:].clone()
            prev_n_imag[level] = n_imag[-1:].clone()

        del batch, Yh
        torch.cuda.empty_cache()

    # Cumulative sum on CPU to get absolute phase
    phase_arrays = []
    for level in range(nlevels):
        np.cumsum(delta_arrays[level], axis=0, out=delta_arrays[level])
        phase_arrays.append(delta_arrays[level])

    del xfm, prev_n_real, prev_n_imag
    torch.cuda.empty_cache()
    return phase_arrays


def _gpu_inverse_pass(data, phase_arrays, nlevels, biort, qshift, device):
    """GPU Pass 2: Re-run forward DTCWT, reconstruct with modified phases, inverse.

    Re-runs forward DTCWT to recover Yl (lowpass) and amplitudes, applies
    modified phases from the temporal filter, then runs inverse DTCWT.

    Args:
        data: Original frames (num_frames, H, W), float32.
        phase_arrays: List of nlevels numpy arrays (num_frames, num_coeffs), float32.
        nlevels: Number of DTCWT decomposition levels.
        biort: Biorthogonal filter name.
        qshift: Quarter-shift filter name.
        device: torch.device for GPU.

    Returns:
        Reconstructed frames (num_frames, H, W), float32.
    """
    import torch
    from pytorch_wavelets import DTCWTForward, DTCWTInverse

    xfm = DTCWTForward(J=nlevels, biort=biort, qshift=qshift).to(device)
    ifm = DTCWTInverse(biort=biort, qshift=qshift).to(device)
    num_frames = data.shape[0]
    h, w = data.shape[1], data.shape[2]

    # Get level shapes
    with torch.no_grad():
        test_frame = torch.from_numpy(data[0:1, np.newaxis, :, :]).to(device)
        _, Yh_test = xfm(test_frame)
        level_shapes = [Yh_test[lv].shape[2:] for lv in range(nlevels)]
        del test_frame, Yh_test
        torch.cuda.empty_cache()

    # Auto-tune batch size
    device_idx = device.index if device.index is not None else 0
    free_vram = torch.cuda.mem_get_info(device_idx)[0]
    frame_bytes = h * w * 4 * 20  # ~20x overhead for fwd + inv
    batch_size = max(1, int(free_vram * 0.7 / frame_bytes))
    batch_size = min(batch_size, num_frames)

    result = np.empty_like(data)

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        batch = torch.from_numpy(data[start:end, np.newaxis, :, :]).to(device)

        with torch.no_grad():
            Yl, Yh = xfm(batch)

            # Reconstruct Yh with modified phases
            Yh_mod = []
            for level in range(nlevels):
                hp = Yh[level]  # (B, 1, 6, H_l, W_l, 2)
                c_real = hp[..., 0]
                c_imag = hp[..., 1]
                amp = torch.sqrt(c_real ** 2 + c_imag ** 2)

                # Load modified phase, reshape to match coefficient layout
                # phase_arrays[level] is (num_frames, 6*H*W) flattened from (B, 1, 6, H, W)
                coeff_shape = level_shapes[level][:-1]  # (6, H_l, W_l)
                mod_phase = torch.from_numpy(
                    phase_arrays[level][start:end].reshape(
                        end - start, 1, *coeff_shape)
                ).to(device)

                new_real = amp * torch.cos(mod_phase)
                new_imag = amp * torch.sin(mod_phase)
                Yh_mod.append(torch.stack([new_real, new_imag], dim=-1))

            recon = ifm((Yl, Yh_mod))

        result[start:end] = recon.cpu().numpy()[:, 0, :h, :w]
        del batch, Yl, Yh, Yh_mod, recon
        torch.cuda.empty_cache()

    del xfm, ifm
    torch.cuda.empty_cache()
    return result


def magnify_motions_gpu(data, magnification=3.0, width=80, nlevels=8,
                        biort='near_sym_b', qshift='qshift_b', device=None):
    """GPU-accelerated phase-based motion magnification on a single channel.

    Two-pass pipeline:
    1. Forward DTCWT + phase extraction (batched on GPU)
    2. Temporal filtering (chunked cuFFT on GPU)
    3. Coefficient reconstruction + inverse DTCWT (batched on GPU)

    Args:
        data: 3D numpy array (num_frames, height, width), float32.
        magnification: Amplification factor for phase deviations.
        width: Temporal filter width in frames.
        nlevels: Number of DTCWT decomposition levels.
        biort: Biorthogonal filter name.
        qshift: Quarter-shift filter name.
        device: torch.device for GPU.

    Returns:
        3D numpy array of same shape as input with magnified motions, float32.
    """
    import torch
    if device is None:
        device = torch.device('cuda')

    # Pass 1: Forward DTCWT + phase extraction
    print("  GPU Forward DTCWT + phase extraction...")
    t0 = time.time()
    phase_arrays = _gpu_forward_pass(data, nlevels, biort, qshift, device)
    print(f"    Done in {format_duration(time.time() - t0)}")

    # Temporal filtering
    print("  GPU Temporal filtering...")
    t0 = time.time()
    _gpu_temporal_filter(phase_arrays, magnification, width, device)
    print(f"    Done in {format_duration(time.time() - t0)}")

    # Pass 2: Reconstruction + inverse DTCWT
    print("  GPU Inverse DTCWT...")
    t0 = time.time()
    result = _gpu_inverse_pass(data, phase_arrays, nlevels, biort, qshift, device)
    print(f"    Done in {format_duration(time.time() - t0)}")

    return result


def _gpu_temporal_filter(phase_arrays, magnification, width, device):
    """GPU temporal filtering via chunked cuFFT.

    Applies flat-top window filtering and phase modification in-place.
    Chunks along the coefficient dimension to fit in VRAM.

    Args:
        phase_arrays: List of numpy arrays (num_frames, num_coeffs), float32.
            Modified in-place.
        magnification: Amplification factor for phase detail.
        width: Temporal filter width in frames.
        device: torch.device for GPU.
    """
    import torch

    large_window = _flattop_window(width)
    small_window = _flattop_window(2.0)
    num_frames = phase_arrays[0].shape[0]

    # Reflect-pad size matches the larger window's half-width
    pad_size = len(large_window) // 2
    padded_frames = num_frames + 2 * pad_size

    # Pre-compute FFT of windows for the padded length
    large_fft_n = int(2 ** np.ceil(np.log2(padded_frames + len(large_window) - 1)))
    small_fft_n = int(2 ** np.ceil(np.log2(padded_frames + len(small_window) - 1)))

    # Center windows at index 0 for zero-phase filtering via circular shift
    def _center_window_fft(window_np, fft_n, dev):
        win_t = torch.from_numpy(window_np.astype(np.float32)).to(dev)
        padded = torch.zeros(fft_n, device=dev)
        half = len(window_np) // 2
        padded[:len(window_np) - half] = win_t[half:]
        if half > 0:
            padded[-half:] = win_t[:half]
        return torch.fft.rfft(padded)

    large_win_fft = _center_window_fft(large_window, large_fft_n, device)
    small_win_fft = _center_window_fft(small_window, small_fft_n, device)

    # Auto-tune chunk size from available VRAM
    device_idx = device.index if device.index is not None else 0
    free_vram = torch.cuda.mem_get_info(device_idx)[0]
    # Each coefficient needs: padded_frames * 80 bytes (FFT overhead ~20x float32)
    bytes_per_coeff = padded_frames * 80
    chunk_size = max(1000, int(free_vram * 0.5 / max(bytes_per_coeff, 1)))

    for level in range(len(phase_arrays)):
        phase = phase_arrays[level]
        num_coeffs = phase.shape[1]

        for start in range(0, num_coeffs, chunk_size):
            end = min(start + chunk_size, num_coeffs)

            # Reflect-pad along time axis on CPU before GPU transfer
            chunk_np = phase[:, start:end]
            chunk_padded = np.pad(chunk_np, [(pad_size, pad_size), (0, 0)],
                                  mode='reflect')
            chunk = torch.from_numpy(chunk_padded).to(device)
            del chunk_padded

            # Large window filter → phase0 (base motion)
            data_fft = torch.fft.rfft(chunk, n=large_fft_n, dim=0)
            phase0 = torch.fft.irfft(
                data_fft * large_win_fft.unsqueeze(1), n=large_fft_n, dim=0
            )[:padded_frames]
            del data_fft

            # Amplify detail
            chunk = phase0 + (chunk - phase0) * magnification
            del phase0

            # Small window smoothing
            data_fft2 = torch.fft.rfft(chunk, n=small_fft_n, dim=0)
            chunk = torch.fft.irfft(
                data_fft2 * small_win_fft.unsqueeze(1), n=small_fft_n, dim=0
            )[:padded_frames]
            del data_fft2

            # Trim padding, write back
            phase[:, start:end] = chunk[pad_size:pad_size + num_frames].cpu().numpy()
            del chunk
            torch.cuda.empty_cache()


def magnify_motions(data, magnification=3.0, width=80, nlevels=8,
                    biort='near_sym_b', qshift='qshift_b'):
    """Run the phase-based motion magnification pipeline on a single channel.

    The algorithm:
    1. Forward 2D DTCWT — decompose each frame into nlevels scales x 6 orientations
    2. Phase extraction — cumulative phase relative to frame 0 via complex division
    3. Temporal filtering — flat-top low-pass separates base motion from detail
    4. Phase modification — amplify detail: phase0 + (phase - phase0) * k
    5. Smoothing — additional low-pass (width=2) removes high-freq phase noise
    6. Inverse DTCWT — reconstruct with modified phase, preserved amplitude

    Note: All frame pyramids must remain in memory for temporal filtering.
    This is memory-intensive for long videos.

    Args:
        data: 3D numpy array of shape (num_frames, height, width), single channel.
        magnification: Amplification factor for phase deviations (default: 3.0).
        width: Temporal filter width in frames (default: 80).
        nlevels: Number of DTCWT decomposition levels (default: 8).
        biort: Biorthogonal filter for DTCWT level 1 (default: 'near_sym_b').
        qshift: Quarter-shift filter for DTCWT levels 2+ (default: 'qshift_b').

    Returns:
        3D numpy array of same shape as input with magnified motions.
    """
    transform = dtcwt.Transform2d(biort=biort, qshift=qshift)
    num_frames = data.shape[0]
    pyramids = []

    # Step 1: Forward DTCWT
    print("  Forward DTCWT...")
    t_start = time.time()
    for i in range(num_frames):
        pyramids.append(transform.forward(data[i, :, :], nlevels=nlevels))

        if (i + 1) % max(1, num_frames // 10) == 0:
            elapsed = time.time() - t_start
            pct = (i + 1) / num_frames
            eta = elapsed / pct * (1 - pct)
            print(f"    {i + 1}/{num_frames} frames "
                  f"({pct:.0%}) — {format_duration(eta)} remaining")

    # Steps 2–5: Phase extraction, filtering, modification
    print("  Modifying phase...")
    for level in range(nlevels):
        print(f"    Level {level + 1}/{nlevels}")

        # Step 2: Extract cumulative temporal phase
        phase = extract_temporal_phases(pyramids, level)

        # Step 3: Temporal filtering — separate base motion from detail
        phase0 = flattop_filter_1d(phase, width, axis=0, mode='reflect')

        # Step 4: Amplify detail phase deviations by magnification factor
        phase = phase0 + (phase - phase0) * magnification

        # Step 5: Additional smoothing to remove high-frequency phase noise
        phase = flattop_filter_1d(phase, 2.0, axis=0, mode='reflect')

        # Reconstruct coefficients: preserve amplitude, replace phase
        # Process frame-by-frame to avoid materializing large intermediate arrays
        shape = pyramids[0].highpasses[level].shape
        for i in range(num_frames):
            coeffs = pyramids[i].highpasses[level].flatten()
            amp = np.abs(coeffs)
            pyramids[i].highpasses[level][:] = (
                amp * np.exp(1j * phase[i])
            ).reshape(shape)

    # Step 6: Inverse DTCWT
    print("  Inverse DTCWT...")
    result = np.empty_like(data)
    t_start = time.time()
    for i in range(num_frames):
        result[i, :, :] = transform.inverse(pyramids[i])

        if (i + 1) % max(1, num_frames // 10) == 0:
            elapsed = time.time() - t_start
            pct = (i + 1) / num_frames
            eta = elapsed / pct * (1 - pct)
            print(f"    {i + 1}/{num_frames} frames "
                  f"({pct:.0%}) — {format_duration(eta)} remaining")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Phase-Based Motion Magnification Using 2D DTCWT — "
                    "amplify subtle motions in video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python motion_mag.py -i face.mp4\n"
            "  python motion_mag.py -i face.mp4 -o magnified.avi -k 5\n"
            "  python motion_mag.py -i face.mp4 -k 3 -w 80 --nlevels 6"
        )
    )
    parser.add_argument(
        '--version', action='version',
        version=f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Input video path'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output video path (default: <input>_magnified.avi)'
    )
    parser.add_argument(
        '-k', '--magnification', type=float, default=3,
        help='Magnification factor (default: 3)'
    )
    parser.add_argument(
        '-w', '--width', type=float, default=80,
        help='Temporal filter width in frames (default: 80)'
    )
    parser.add_argument(
        '--nlevels', type=int, default=8,
        help='Number of DTCWT decomposition levels (default: 8)'
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Use GPU acceleration (requires PyTorch + pytorch_wavelets)'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='CUDA device index (default: 0)'
    )
    parser.add_argument(
        '--biort', default='near_sym_b',
        help='DTCWT biorthogonal filter (default: near_sym_b)'
    )
    parser.add_argument(
        '--qshift', default='qshift_b',
        help='DTCWT quarter-shift filter (default: qshift_b)'
    )

    args = parser.parse_args()

    # --- Validation ---
    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.magnification <= 0:
        print("Error: --magnification must be positive", file=sys.stderr)
        sys.exit(1)

    if args.width <= 0:
        print("Error: --width must be positive", file=sys.stderr)
        sys.exit(1)

    if args.nlevels < 1:
        print("Error: --nlevels must be at least 1", file=sys.stderr)
        sys.exit(1)

    # --- GPU validation ---
    if args.gpu:
        try:
            import torch
        except ImportError:
            print("Error: --gpu requires PyTorch. Install it or use "
                  "Dockerfile.gpu.", file=sys.stderr)
            sys.exit(1)
        try:
            from pytorch_wavelets import DTCWTForward  # noqa: F401
        except ImportError:
            print("Error: --gpu requires pytorch_wavelets. Install with: "
                  "pip install git+https://github.com/fbcotter/pytorch_wavelets.git",
                  file=sys.stderr)
            sys.exit(1)
        if not torch.cuda.is_available():
            print("Error: --gpu requires CUDA but no GPU is available.",
                  file=sys.stderr)
            sys.exit(1)

    # --- Default output path ---
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_magnified.avi"

    # --- Load video ---
    total_start = time.time()
    print(f"Loading {args.input}...")
    channels, fps, frame_size = load_video(args.input)
    frame_count = channels[0].shape[0]
    print(f"  {frame_count} frames, {frame_size[0]}x{frame_size[1]}, {fps} fps")

    # --- Parameters ---
    print("\nParameters:")
    print(f"  Magnification:   {args.magnification}x")
    print(f"  Filter width:    {args.width}")
    print(f"  DTCWT levels:    {args.nlevels}")
    print(f"  Biort filter:    {args.biort}")
    print(f"  Qshift filter:   {args.qshift}\n")

    # --- Process each channel independently ---
    channel_names = ['red', 'green', 'blue']

    if args.gpu:
        import torch
        device = torch.device('cuda', args.device)
        gpu_name = torch.cuda.get_device_name(args.device)
        gpu_vram = torch.cuda.get_device_properties(args.device).total_memory
        print(f"GPU: {gpu_name} ({gpu_vram / 1024**3:.1f} GB VRAM)")
        # Convert to float32 for GPU path
        channels = [ch.astype(np.float32) for ch in channels]

    for idx, name in enumerate(channel_names):
        print(f"Processing {name} channel...")
        t0 = time.time()
        if args.gpu:
            channels[idx] = magnify_motions_gpu(
                channels[idx],
                magnification=args.magnification,
                width=args.width,
                nlevels=args.nlevels,
                biort=args.biort,
                qshift=args.qshift,
                device=device,
            )
        else:
            channels[idx] = magnify_motions(
                channels[idx],
                magnification=args.magnification,
                width=args.width,
                nlevels=args.nlevels,
                biort=args.biort,
                qshift=args.qshift,
            )
        print(f"  Done in {format_duration(time.time() - t0)}")

    # --- Save ---
    print("Saving output...")
    save_video(channels, fps, args.output, frame_size)

    print(f"Total processing time: "
          f"{format_duration(time.time() - total_start)}")


if __name__ == '__main__':
    main()
