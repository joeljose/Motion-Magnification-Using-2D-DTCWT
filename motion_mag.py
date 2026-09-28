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

__version__ = "2.0.0"

import argparse
import mmap
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import cv2
import dtcwt
import numpy as np
from scipy import ndimage, signal


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


def temporal_phase(coeffs):
    """Cumulative phase over time for a (num_frames, num_coeffs) array.

    Frame-to-frame phase changes come from conjugate multiplication (no
    phase wrapping, unlike subtracting angles); their cumulative sum is the
    phase relative to frame 0. Row 0 holds frame 0's absolute phase, which
    reconstruction needs. A zero coefficient gives angle(0) = 0 (no change);
    a division would give 0/0 = NaN.

    Args:
        coeffs: Complex array (num_frames, num_coeffs), e.g. a column chunk
            of one DTCWT level.

    Returns:
        float32 array of the same shape.
    """
    unit = normalize_phase(coeffs)
    angles = np.empty(coeffs.shape, dtype=np.float32)
    angles[0] = np.angle(unit[0])
    angles[1:] = np.angle(unit[1:] * np.conj(unit[:-1]))
    np.cumsum(angles, axis=0, out=angles)
    return angles


def extract_temporal_phases(pyramids, level):
    """Cumulative phase over time at one DTCWT level of a list of pyramids.

    Args:
        pyramids: List of dtcwt Pyramid objects, one per frame.
        level: DTCWT decomposition level index.

    Returns:
        float32 array (num_frames, num_coefficients); see temporal_phase().
    """
    return temporal_phase(np.stack([p.highpasses[level].ravel() for p in pyramids]))


def _flattop_window(width):
    """Compute a normalized flat-top window for the given filter width.

    The length is forced odd so the window has a centre tap and the filter
    is zero-phase; an even length shifts the output by half a frame.
    """
    window_size = max(1, round(width / 0.2327)) | 1
    window = signal.windows.flattop(window_size)
    return window / np.sum(window)


# Threshold: windows larger than this use FFT convolution (faster for large
# kernels due to O(n log n) vs O(n*k) complexity).
_FFT_THRESHOLD = 32

# np.pad modes matching each ndimage boundary mode ('reflect' in ndimage
# repeats the edge sample, which is 'symmetric' in np.pad)
_NP_PAD_MODE = {'reflect': 'symmetric', 'mirror': 'reflect', 'nearest': 'edge',
                'constant': 'constant', 'wrap': 'wrap'}


def flattop_filter_1d(data, width, axis=0, mode='reflect'):
    """Apply a flat-top window low-pass filter along the specified axis.

    Uses a flat-top window (scipy.signal.windows.flattop) as a smoothing
    kernel. The window length is round(width / 0.2327), forced odd; 0.2327
    is an empirical width-to-length factor from the reference IDL
    implementation, not the window's equivalent noise bandwidth (~3.77 bins).
    At 30 fps, width 80 gives a half-amplitude cutoff of about 0.20 Hz.

    For large windows (>32 samples), uses FFT-based convolution with the
    same boundary padding as ndimage, for ~4x speedup. For small windows, uses
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

    # FFT path: pad with the same boundary rule as ndimage (see
    # _NP_PAD_MODE), then use fftconvolve in chunks
    # for cache efficiency. Chunking along the non-convolution axis
    # keeps working sets in L2/L3 cache.
    pad_size = len(window) // 2
    pad_mode = _NP_PAD_MODE[mode]
    n_along = data.shape[axis]
    n_across = data.size // n_along
    chunk_size = min(n_across, 10000)

    # Build axis-aware shapes for padding and kernel
    pad_widths = [(0, 0)] * data.ndim
    pad_widths[axis] = (pad_size, pad_size)
    kernel_shape = [1] * data.ndim
    kernel_shape[axis] = len(window)
    # Match the data's precision so float32 input gets float32 FFTs (half
    # the memory of float64); float64 input is unchanged
    kernel = window.astype(np.result_type(data.dtype, np.float32)).reshape(kernel_shape)

    # For 2D (frames, coeffs) with axis=0, chunk along axis=1
    if data.ndim == 2 and axis == 0:
        result = np.empty_like(data)
        for start in range(0, data.shape[1], chunk_size):
            end = min(start + chunk_size, data.shape[1])
            chunk = data[:, start:end]
            padded = np.pad(chunk, [(pad_size, pad_size), (0, 0)], mode=pad_mode)
            conv = signal.fftconvolve(padded, kernel, mode='same', axes=0)
            result[:, start:end] = conv[pad_size:pad_size + n_along]
        return result

    # General fallback: no chunking
    padded = np.pad(data, pad_widths, mode=pad_mode)
    conv = signal.fftconvolve(padded, kernel, mode='same', axes=axis)
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(pad_size, pad_size + n_along)
    return conv[tuple(slices)]


def estimate_memory(num_frames, height, width, nlevels, gpu=False, jobs=2):
    """Estimate peak CPU RAM and VRAM usage in bytes.

    Uses the real DTCWT coefficient count (about 2x the pixel count), from a
    forward transform of one blank frame.

    Args:
        num_frames: Number of video frames.
        height: Frame height in pixels.
        width: Frame width in pixels.
        nlevels: Number of DTCWT decomposition levels.
        gpu: If True, estimate for the GPU path; otherwise the CPU path.
        jobs: CPU worker threads (each holds one phase chunk's temporaries).

    Returns:
        Tuple of (cpu_ram_bytes, vram_bytes).
    """
    highpasses = dtcwt.Transform2d().forward(
        np.zeros((height, width)), nlevels=nlevels).highpasses
    coeffs = sum(h.size for h in highpasses)
    pixels = num_frames * height * width
    frames_bytes = pixels * 3  # uint8 R, G, B

    # ponytail: constants fitted to face.mp4 (301x592x528, nlevels 8) with
    # ru_maxrss / cgroup memory.peak; re-fit if the memory layout changes
    if gpu:
        # float32 input channel + float32 result + float32 phases, all
        # levels; 1.2x allocator factor; 640 MiB for imports incl. torch/CUDA
        working = pixels * 4 * 2 + num_frames * coeffs * 4
        cpu_ram = int((frames_bytes + working) * 1.2) + 640 * 1024**2
    else:
        # shared copy of the input channel + complex64 coefficients + float32
        # result + ~130 MiB of phase-chunk temporaries per thread; 128 MiB
        # for imports
        working = pixels + num_frames * coeffs * 8 + pixels * 4
        cpu_ram = frames_bytes + working + (jobs * 130 + 128) * 1024**2

    # VRAM estimate (GPU only)
    if gpu:
        # DTCWT batch: ~13 MB per frame at 528×592
        batch_vram = 10 * height * width * 13 * 4  # 10 frames × overhead
        # cuFFT chunk: largest level phase chunk + FFT buffers
        fft_vram = min(num_frames * coeffs * 4, 500 * 1024 * 1024)  # cap at 500 MB
        vram = batch_vram + fft_vram + 300 * 1024 * 1024  # 300 MB overhead
    else:
        vram = 0

    return cpu_ram, vram


def _available_memory():
    """Available RAM in bytes from /proc/meminfo, or None if unknown."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def load_video(path):
    """Load a video file into separate R, G, B channel arrays.

    Each channel is a uint8 array of shape (num_frames, height, width);
    processing converts one channel at a time, which keeps memory low.
    OpenCV reads in BGR order; channels are split accordingly.

    Frames are read until the decoder stops, so a container that
    under-reports its frame count loses nothing.

    Args:
        path: Path to the video file.

    Returns:
        Tuple of (channels, fps, frame_size) where:
        - channels: list of 3 numpy arrays [R, G, B], each (N, H, W) uint8
        - fps: frame rate reported by the container (float; 0 if unknown)
        - frame_size: (width, height) tuple

    Raises:
        ValueError: If the file cannot be opened or has no decodable frames.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        reported = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = None
        i = 0
        t_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frames is None:
                frames = np.empty((max(reported, 1),) + frame.shape, dtype=np.uint8)
            elif i == len(frames):
                # More frames than reported: grow the buffer by half again
                grow = np.empty((len(frames) // 2 + 1,) + frame.shape, dtype=np.uint8)
                frames = np.concatenate([frames, grow])
            frames[i] = frame
            i += 1

            if reported and i <= reported and i % max(1, reported // 10) == 0:
                elapsed = time.time() - t_start
                pct = i / reported
                eta = elapsed / pct * (1 - pct)
                print(f"  Reading: {i}/{reported} frames "
                      f"({pct:.0%}) — {format_duration(eta)} remaining")
    finally:
        cap.release()

    if i == 0:
        raise ValueError(f"no decodable frames in {path}")
    if abs(i - reported) > 1:
        print(f"  Note: container reported {reported} frames, decoded {i}")

    frames = frames[:i]
    height, width = frames.shape[1:3]

    # Split BGR channels into separate contiguous uint8 arrays
    channels = [
        np.ascontiguousarray(frames[:, :, :, 2]),  # R
        np.ascontiguousarray(frames[:, :, :, 1]),  # G
        np.ascontiguousarray(frames[:, :, :, 0]),  # B
    ]
    del frames

    return channels, fps, (width, height)


def save_video(channels, fps, path, frame_size):
    """Save R, G, B channel arrays to an AVI video file.

    Recombines the three channels into BGR frames, rounds and clips to
    [0, 255], and writes using MJPG codec.

    Args:
        channels: List of 3 numpy arrays [R, G, B], each (N, H, W).
        fps: Frame rate for the output video.
        path: Output file path.
        frame_size: (width, height) tuple.

    Raises:
        RuntimeError: If the writer cannot be opened or writes no data.
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size, True)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not open video writer for {path} "
                           f"(codec MJPG, fps {fps})")
    try:
        frame_count = channels[0].shape[0]
        result = np.empty(
            (frame_count, channels[0].shape[1], channels[0].shape[2], 3),
            dtype=np.uint8
        )
        result[:, :, :, 2] = np.nan_to_num(np.clip(np.rint(channels[0]), 0, 255)).astype(np.uint8)
        result[:, :, :, 1] = np.nan_to_num(np.clip(np.rint(channels[1]), 0, 255)).astype(np.uint8)
        result[:, :, :, 0] = np.nan_to_num(np.clip(np.rint(channels[2]), 0, 255)).astype(np.uint8)

        for i in range(frame_count):
            writer.write(result[i])
    finally:
        writer.release()
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"video writer produced no data in {path}")
    print(f"Output saved to {path}")


def _free_memory(device):
    """Free bytes on a torch device: VRAM for CUDA, RAM for CPU."""
    import torch
    if device.type == 'cuda':
        return torch.cuda.mem_get_info(device.index or 0)[0]
    available = _available_memory()
    return available if available is not None else 4 * 1024**3


def _run_batches(total, batch_size, process, what):
    """Call process(start, end) over [0, total) in batches.

    On a CUDA out-of-memory error the batch is retried at half the size,
    down to 1; only a failure at size 1 is raised.
    """
    import torch
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        try:
            process(start, end)
        except torch.cuda.OutOfMemoryError:
            if end - start == 1:
                raise
            batch_size = max(1, (end - start) // 2)
            torch.cuda.empty_cache()
            print(f"    Out of GPU memory; retrying with {what} size {batch_size}")
            continue
        start = end


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
        device: torch.device (CUDA, or CPU for testing).

    Returns:
        List of nlevels numpy arrays, each (num_frames, num_coeffs) float32,
        containing cumulative phase per coefficient.
    """
    import torch
    from pytorch_wavelets import DTCWTForward

    xfm = DTCWTForward(J=nlevels, biort=biort, qshift=qshift).to(device)
    num_frames = data.shape[0]

    # Batch size from free memory (70%); OOM retries shrink it further
    frame_bytes = data.shape[1] * data.shape[2] * 4 * 15  # ~15x for DTCWT overhead
    batch_size = max(1, min(num_frames, int(_free_memory(device) * 0.7 / frame_bytes)))

    # Coefficient counts per level from a single-frame forward pass
    with torch.no_grad():
        test_frame = torch.from_numpy(data[0:1, np.newaxis, :, :]).to(device)
        _, Yh_test = xfm(test_frame)
        level_coeffs = [int(np.prod(Yh_test[level].shape[2:-1]))  # 6 * H_l * W_l
                        for level in range(nlevels)]
        del test_frame, Yh_test
        torch.cuda.empty_cache()

    # Allocate phase delta arrays on CPU
    delta_arrays = [np.empty((num_frames, nc), dtype=np.float32)
                    for nc in level_coeffs]

    # Previous frame's normalized coefficients, for the cross-batch boundary
    prev = [None] * nlevels

    def process(start, end):
        batch = torch.from_numpy(data[start:end, np.newaxis, :, :]).to(device)
        with torch.no_grad():
            _, Yh = xfm(batch)

        last = [None] * nlevels
        for level in range(nlevels):
            hp = Yh[level]  # (B, 1, 6, H, W, 2)
            c_real = hp[..., 0]  # (B, 1, 6, H, W)
            c_imag = hp[..., 1]

            # Normalize to unit magnitude
            mag = torch.sqrt(c_real ** 2 + c_imag ** 2)
            mag = torch.where(mag > 1e-20, mag, torch.ones_like(mag))
            n_real = c_real / mag
            n_imag = c_imag / mag

            if start == 0:
                # Frame 0 (global): absolute phase
                phase0 = torch.atan2(n_imag[0:1], n_real[0:1])
                delta_arrays[level][0] = phase0.reshape(1, -1).cpu().numpy()
            else:
                # Cross-batch boundary: first frame vs prev batch's last frame
                pr, pi = prev[level]
                boundary_real = n_real[0:1] * pr + n_imag[0:1] * pi
                boundary_imag = n_imag[0:1] * pr - n_real[0:1] * pi
                delta_arrays[level][start] = (
                    torch.atan2(boundary_imag, boundary_real)
                    .reshape(1, -1).cpu().numpy()
                )

            # Intra-batch deltas (vectorized)
            if end - start > 1:
                pr = n_real[1:] * n_real[:-1] + n_imag[1:] * n_imag[:-1]
                pi = n_imag[1:] * n_real[:-1] - n_real[1:] * n_imag[:-1]
                deltas = torch.atan2(pi, pr)
                delta_arrays[level][start + 1:end] = (
                    deltas.reshape(end - start - 1, -1).cpu().numpy()
                )

            last[level] = (n_real[-1:].clone(), n_imag[-1:].clone())

        # Commit the boundary state only once the whole batch succeeded, so
        # an OOM retry of this batch starts from the right previous frame
        prev[:] = last
        del batch, Yh
        torch.cuda.empty_cache()

    _run_batches(num_frames, batch_size, process, "batch")

    # Cumulative sum on CPU to get absolute phase
    for level in range(nlevels):
        np.cumsum(delta_arrays[level], axis=0, out=delta_arrays[level])

    torch.cuda.empty_cache()
    return delta_arrays


def _gpu_temporal_filter(phase_arrays, magnification, width, device):
    """GPU temporal filtering via chunked cuFFT.

    Applies flat-top window filtering and phase modification in-place.
    Chunks along the coefficient dimension to fit in VRAM.

    Args:
        phase_arrays: List of numpy arrays (num_frames, num_coeffs), float32.
            Modified in-place.
        magnification: Amplification factor for phase detail.
        width: Temporal filter width in frames.
        device: torch.device (CUDA, or CPU for testing).
    """
    import torch

    large_window = _flattop_window(width)
    small_window = _flattop_window(2.0)
    num_frames = phase_arrays[0].shape[0]

    # Padding size matches the larger window's half-width
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

    # Chunk size from free memory (50%): each coefficient needs about
    # padded_frames * 80 bytes (FFT overhead ~20x float32)
    bytes_per_coeff = padded_frames * 80
    chunk_size = max(64, int(_free_memory(device) * 0.5 / bytes_per_coeff))

    for phase in phase_arrays:
        def process(start, end):
            # Pad along time on CPU before transfer, with the same boundary
            # rule as the CPU path (ndimage 'reflect')
            chunk_padded = np.pad(phase[:, start:end], [(pad_size, pad_size), (0, 0)],
                                  mode='symmetric')
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

        _run_batches(phase.shape[1], chunk_size, process, "chunk")


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
        device: torch.device (CUDA, or CPU for testing).

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

    # Batch size from free memory (70%); OOM retries shrink it further
    frame_bytes = h * w * 4 * 20  # ~20x overhead for fwd + inv
    batch_size = max(1, min(num_frames, int(_free_memory(device) * 0.7 / frame_bytes)))

    result = np.empty_like(data)

    def process(start, end):
        batch = torch.from_numpy(data[start:end, np.newaxis, :, :]).to(device)

        with torch.no_grad():
            Yl, Yh = xfm(batch)

            # Reconstruct Yh with modified phases
            Yh_mod = []
            for level in range(nlevels):
                hp = Yh[level]  # (B, 1, 6, H_l, W_l, 2)
                amp = torch.sqrt(hp[..., 0] ** 2 + hp[..., 1] ** 2)

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

    _run_batches(num_frames, batch_size, process, "batch")

    torch.cuda.empty_cache()
    return result


def magnify_motions_gpu(data, magnification=3.0, width=80, nlevels=8,
                        biort='near_sym_b', qshift='qshift_b', device=None):
    """GPU-accelerated phase-based motion magnification on a single channel.

    Two-pass pipeline:
    1. Forward DTCWT + phase extraction (batched on GPU)
    2. Temporal filtering (chunked cuFFT on GPU)
    3. Coefficient reconstruction + inverse DTCWT (batched on GPU)

    Batches and chunks that run out of GPU memory are retried at half the
    size; torch.cuda.OutOfMemoryError is raised only if a single frame or
    coefficient chunk does not fit.

    Args:
        data: 3D numpy array (num_frames, height, width), any real dtype
            (converted to float32).
        magnification: Amplification factor for phase deviations.
        width: Temporal filter width in frames.
        nlevels: Number of DTCWT decomposition levels.
        biort: Biorthogonal filter name.
        qshift: Quarter-shift filter name.
        device: torch.device (default CUDA; CPU works too, for testing).

    Returns:
        3D numpy array of same shape as input with magnified motions, float32.
    """
    import torch
    if device is None:
        device = torch.device('cuda')
    data = np.asarray(data, dtype=np.float32)  # the torch DTCWT modules are float32

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


# Coefficient columns per phase-processing task in magnify_motions; each
# thread holds a few (num_frames x chunk) temporaries
_PHASE_CHUNK = 10000


def _shared_empty(shape, dtype):
    """An array in anonymous shared memory, visible to forked workers."""
    nbytes = max(1, int(np.prod(shape)) * np.dtype(dtype).itemsize)
    return np.ndarray(shape, dtype, buffer=mmap.mmap(-1, nbytes))


# State shared with forked worker processes (set before the pool forks)
_POOL = {}


def _transform(biort, qshift):
    key = (biort, qshift)
    if key not in _POOL.setdefault('transforms', {}):
        _POOL['transforms'][key] = dtcwt.Transform2d(biort=biort, qshift=qshift)
    return _POOL['transforms'][key]


def _forward_block(start, end):
    """Forward DTCWT of frames [start, end) into the shared level arrays."""
    st = _POOL['state']
    transform = _transform(st['biort'], st['qshift'])
    for i in range(start, end):
        pyramid = transform.forward(st['data'][i], nlevels=st['nlevels'])
        st['lowpass'][i] = pyramid.lowpass
        for level, hp in enumerate(pyramid.highpasses):
            st['highpasses'][level][i] = hp
    return end - start


def _inverse_block(start, end):
    """Inverse DTCWT of frames [start, end) into the shared result array."""
    st = _POOL['state']
    transform = _transform(st['biort'], st['qshift'])
    h, w = st['result'].shape[1:]
    for i in range(start, end):
        pyramid = dtcwt.Pyramid(st['lowpass'][i],
                                tuple(hp[i] for hp in st['highpasses']))
        # dtcwt pads odd sizes by one row/column; crop back to the input size
        st['result'][i] = transform.inverse(pyramid)[:h, :w]
    return end - start


def _run_frame_blocks(func, num_frames, jobs, label):
    """Run func(start, end) over frame blocks, in forked processes if jobs > 1."""
    blocks = max(1, min(num_frames, jobs * 4))
    bounds = np.linspace(0, num_frames, blocks + 1).astype(int)
    tasks = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    t_start, done, next_report = time.time(), 0, 0.1

    def report(n):
        nonlocal done, next_report
        done += n
        if done / num_frames >= next_report or done == num_frames:
            pct = done / num_frames
            eta = (time.time() - t_start) / pct * (1 - pct)
            print(f"    {label}: {done}/{num_frames} frames "
                  f"({pct:.0%}) — {format_duration(eta)} remaining")
            next_report = pct + 0.1

    if jobs == 1:
        for a, b in tasks:
            report(func(a, b))
        return
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('fork')) as pool:
        for future in as_completed([pool.submit(func, a, b) for a, b in tasks]):
            report(future.result())


# Default worker count. The dtcwt transforms are memory-bound: on a 6-core
# laptop (Ryzen 7 7445HS) 2-3 workers were fastest and 4+ were slower than
# 2, so more is not better by default. Raise --jobs on machines with more
# memory bandwidth.
_DEFAULT_JOBS = 2


def _default_jobs(jobs):
    """Worker count: `jobs`, else _DEFAULT_JOBS; 1 where fork is unavailable."""
    if 'fork' not in multiprocessing.get_all_start_methods():
        return 1
    return max(1, min(jobs or _DEFAULT_JOBS, os.cpu_count() or 1))


def magnify_motions(data, magnification=3.0, width=80, nlevels=8,
                    biort='near_sym_b', qshift='qshift_b', jobs=None):
    """Run the phase-based motion magnification pipeline on a single channel.

    The algorithm:
    1. Forward 2D DTCWT — decompose each frame into nlevels scales x 6 orientations
    2. Phase extraction — cumulative phase relative to frame 0 via conjugate multiply
    3. Temporal filtering — flat-top low-pass separates base motion from detail
    4. Phase modification — amplify detail: phase0 + (phase - phase0) * k
    5. Smoothing — additional low-pass (width=2) removes high-freq phase noise
    6. Inverse DTCWT — reconstruct with modified phase, preserved amplitude

    The DTCWT steps run in forked worker processes over blocks of frames,
    writing into shared memory; steps 2-5 run in threads over column chunks
    of each level (NumPy and SciPy's FFT release the GIL). Every frame and
    every coefficient column is independent, so the result does not depend
    on `jobs`.

    Note: All coefficients must remain in memory for temporal filtering.
    They are stored as complex64 and the phase maths runs in float32, which
    roughly halves memory against float64 with no visible difference.

    Args:
        data: 3D numpy array of shape (num_frames, height, width), single
            channel, any real dtype (uint8 frames are converted per frame).
        magnification: Amplification factor for phase deviations (default: 3.0).
        width: Temporal filter width in frames (default: 80).
        nlevels: Number of DTCWT decomposition levels (default: 8).
        biort: Biorthogonal filter for DTCWT level 1 (default: 'near_sym_b').
        qshift: Quarter-shift filter for DTCWT levels 2+ (default: 'qshift_b').
        jobs: Worker processes/threads (default: 2; 1 = serial).

    Returns:
        float32 array of the same shape as the input, with magnified motions.
    """
    jobs = _default_jobs(jobs)
    num_frames, h, w = data.shape
    probe = _transform(biort, qshift).forward(np.zeros((h, w)), nlevels=nlevels)

    # Shared arrays: input frames, lowpass and one complex64 array per level
    shared_data = _shared_empty(data.shape, data.dtype)
    shared_data[:] = data
    state = {
        'data': shared_data, 'biort': biort, 'qshift': qshift, 'nlevels': nlevels,
        'lowpass': _shared_empty((num_frames,) + probe.lowpass.shape, np.float64),
        'highpasses': [_shared_empty((num_frames,) + hp.shape, np.complex64)
                       for hp in probe.highpasses],
    }
    _POOL['state'] = state
    try:
        # Step 1: Forward DTCWT
        print(f"  Forward DTCWT ({jobs} {'job' if jobs == 1 else 'jobs'})...")
        _run_frame_blocks(_forward_block, num_frames, jobs, "forward")
        state['data'] = None
        del shared_data

        # Steps 2–5 per level, in threads over column chunks; the
        # reconstructed coefficients overwrite the originals in place
        print("  Modifying phase...")

        def process(coeffs, start):
            chunk = coeffs[:, start:start + _PHASE_CHUNK]

            # Step 2: Extract cumulative temporal phase
            phase = temporal_phase(chunk)

            # Step 3: Temporal filtering — separate base motion from detail
            phase0 = flattop_filter_1d(phase, width, axis=0, mode='reflect')

            # Step 4: Amplify detail phase deviations by magnification factor
            phase -= phase0
            phase *= magnification
            phase += phase0
            del phase0

            # Step 5: Additional smoothing to remove high-frequency phase noise
            phase = flattop_filter_1d(phase, 2.0, axis=0, mode='reflect')

            # Reconstruct coefficients: preserve amplitude, replace phase
            chunk[:] = np.abs(chunk) * np.exp(1j * phase)

        with ThreadPoolExecutor(jobs) as threads:
            for level, hp in enumerate(state['highpasses']):
                print(f"    Level {level + 1}/{nlevels}")
                coeffs = hp.reshape(num_frames, -1)
                list(threads.map(lambda start, c=coeffs: process(c, start),
                                 range(0, coeffs.shape[1], _PHASE_CHUNK)))

        # Step 6: Inverse DTCWT
        print("  Inverse DTCWT...")
        state['result'] = _shared_empty(data.shape, np.float32)
        _run_frame_blocks(_inverse_block, num_frames, jobs, "inverse")
        return state['result']
    finally:
        _POOL.pop('state', None)


# Filter names available in both dtcwt (CPU) and pytorch_wavelets (GPU)
BIORT_FILTERS = ('antonini', 'legall', 'near_sym_a', 'near_sym_b')
QSHIFT_FILTERS = ('qshift_06', 'qshift_a', 'qshift_b', 'qshift_c', 'qshift_d')

# Fewer frames than this leave nothing for the temporal filter to separate
_MIN_FRAMES = 3


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
        '--fps', type=float, default=None,
        help='Frame rate of the output (default: from the input video)'
    )
    parser.add_argument(
        '--jobs', type=int, default=None,
        help='CPU worker processes/threads (default: 2; 1 = serial)'
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
        '--biort', default='near_sym_b', choices=BIORT_FILTERS,
        help='DTCWT biorthogonal filter (default: near_sym_b)'
    )
    parser.add_argument(
        '--qshift', default='qshift_b', choices=QSHIFT_FILTERS,
        help='DTCWT quarter-shift filter (default: qshift_b)'
    )

    args = parser.parse_args()

    # --- Validation ---
    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not (np.isfinite(args.magnification) and args.magnification > 0):
        print("Error: --magnification must be positive and finite", file=sys.stderr)
        sys.exit(1)

    if not (np.isfinite(args.width) and args.width > 0):
        print("Error: --width must be positive and finite", file=sys.stderr)
        sys.exit(1)

    if args.nlevels < 1:
        print("Error: --nlevels must be at least 1", file=sys.stderr)
        sys.exit(1)

    if args.jobs is not None and args.jobs < 1:
        print("Error: --jobs must be at least 1", file=sys.stderr)
        sys.exit(1)

    if args.fps is not None and not args.fps > 0:
        print("Error: --fps must be positive", file=sys.stderr)
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
        if not 0 <= args.device < torch.cuda.device_count():
            print(f"Error: --device {args.device} is not a valid CUDA device "
                  f"(found {torch.cuda.device_count()}: 0 to "
                  f"{torch.cuda.device_count() - 1}).", file=sys.stderr)
            sys.exit(1)

    # --- Default output path ---
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_magnified.avi"

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(out_dir):
        print(f"Error: output directory does not exist: {out_dir}", file=sys.stderr)
        sys.exit(1)

    # --- Load video ---
    total_start = time.time()
    print(f"Loading {args.input}...")
    try:
        channels, fps, frame_size = load_video(args.input)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    frame_count = channels[0].shape[0]
    if args.fps is not None:
        fps = args.fps
    elif not (np.isfinite(fps) and fps > 0):
        print(f"Error: could not read the frame rate of {args.input}; "
              f"pass --fps", file=sys.stderr)
        sys.exit(1)
    if frame_count < _MIN_FRAMES:
        print(f"Error: need at least {_MIN_FRAMES} frames for temporal "
              f"filtering, got {frame_count}", file=sys.stderr)
        sys.exit(1)
    print(f"  {frame_count} frames, {frame_size[0]}x{frame_size[1]}, {fps} fps")

    need, _ = estimate_memory(frame_count, frame_size[1], frame_size[0],
                              args.nlevels, gpu=args.gpu,
                              jobs=_default_jobs(args.jobs))
    available = _available_memory()
    print(f"  Estimated peak RAM: {need / 1024**3:.1f} GiB")
    if available is not None and need > available:
        print(f"Warning: estimated peak RAM ({need / 1024**3:.1f} GiB) exceeds "
              f"available memory ({available / 1024**3:.1f} GiB). Consider a "
              f"shorter or smaller clip, or fewer --nlevels.", file=sys.stderr)

    # --- Parameters ---
    print("\nParameters:")
    backend = ("GPU (pytorch_wavelets, float32)" if args.gpu
               else f"CPU (dtcwt, {_default_jobs(args.jobs)} jobs)")
    print(f"  Backend:         {backend}")
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

    for idx, name in enumerate(channel_names):
        print(f"Processing {name} channel...")
        t0 = time.time()
        if args.gpu:
            try:
                result = magnify_motions_gpu(
                    channels[idx].astype(np.float32),
                    magnification=args.magnification,
                    width=args.width,
                    nlevels=args.nlevels,
                    biort=args.biort,
                    qshift=args.qshift,
                    device=device,
                )
            except torch.cuda.OutOfMemoryError:
                print("Error: out of GPU memory even with single-frame batches. "
                      "Try fewer --nlevels, a smaller or shorter clip, or the "
                      "CPU path (omit --gpu).", file=sys.stderr)
                sys.exit(1)
        else:
            result = magnify_motions(
                channels[idx],
                magnification=args.magnification,
                width=args.width,
                nlevels=args.nlevels,
                biort=args.biort,
                qshift=args.qshift,
                jobs=args.jobs,
            )
        # Keep only the uint8 result so finished channels cost 1 byte/pixel
        channels[idx] = np.clip(np.rint(result), 0, 255).astype(np.uint8)
        del result
        print(f"  Done in {format_duration(time.time() - t0)}")

    # --- Save ---
    print("Saving output...")
    try:
        save_video(channels, fps, args.output, frame_size)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Total processing time: "
          f"{format_duration(time.time() - total_start)}")


if __name__ == '__main__':
    main()
