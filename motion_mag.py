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


def flattop_filter_1d(data, width, axis=0, mode='reflect'):
    """Apply a flat-top window low-pass filter along the specified axis.

    Uses a flat-top window (scipy.signal.flattop) as a smoothing kernel.
    The window size is determined by width / 0.2327, where 0.2327 is the
    flat-top window's equivalent noise bandwidth in bins.

    Args:
        data: Input numpy array.
        width: Filter width in frames. Controls the cutoff frequency —
            larger values produce more smoothing (lower cutoff).
        axis: Axis along which to filter (default: 0, the time axis).
        mode: Boundary handling mode for convolution (default: 'reflect').

    Returns:
        Filtered numpy array with same shape as input.
    """
    window_size = max(1, round(width / 0.2327))
    window = signal.windows.flattop(window_size)
    window = window / np.sum(window)
    return ndimage.convolve1d(data, window, axis=axis, mode=mode)


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


def magnify_motions(data, magnification=3.0, width=80, nlevels=8):
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

    Returns:
        3D numpy array of same shape as input with magnified motions.
    """
    transform = dtcwt.Transform2d()
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
    print(f"  DTCWT levels:    {args.nlevels}\n")

    # --- Process each channel independently ---
    channel_names = ['red', 'green', 'blue']
    for idx, name in enumerate(channel_names):
        print(f"Processing {name} channel...")
        t0 = time.time()
        channels[idx] = magnify_motions(
            channels[idx],
            magnification=args.magnification,
            width=args.width,
            nlevels=args.nlevels,
        )
        print(f"  Done in {format_duration(time.time() - t0)}")

    # --- Save ---
    print("Saving output...")
    save_video(channels, fps, args.output, frame_size)

    print(f"Total processing time: "
          f"{format_duration(time.time() - total_start)}")


if __name__ == '__main__':
    main()
