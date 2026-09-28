"""Synthetic pulsating shapes with exact ground truth, for checking the
magnification (issue #39).

A circle (or square) of radius r(t) = r0 + r1 * sin(2 pi f t) is rendered
with a smooth, camera-like edge (Gaussian blur of `edge_sigma` px), so
sub-pixel sizes are exact. Magnifying by k should give exactly the shape
with radius r0 + k * r1 * sin(2 pi f t).

The edge position is measured along many rays from the centre: where the
intensity crosses the midpoint between background and foreground, with
sub-pixel interpolation. From that per-frame, per-angle radius:
  gain        = fitted sine amplitude of the mean radius / r1 (ideal: k)
  phase_lag   = phase of that sine relative to the input, in frames
  harmonics   = energy at 2f, 3f relative to f (distortion)
  angle_spread= how much the gain varies around the shape (isotropy)
  jitter      = residual of the radius after removing the fitted sine
  edge_width  = 10-90% width of the edge profile (blur / doubling)
"""

import os
import sys

import numpy as np
from scipy import ndimage, signal, special

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import motion_mag  # noqa: E402

BG, FG = 60.0, 180.0


def render(shape, radii, size=128, edge_sigma=0.8, thickness=None):
    """Frames (len(radii), size, size) of a centred circle or square.

    With `thickness`, only an outline of that width (px) centred on the
    radius is drawn: a ring or a hollow square.
    """
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2
    if shape == "circle":
        dist = np.hypot(yy, xx)
    else:  # square: Chebyshev distance (half side = radius)
        dist = np.maximum(np.abs(yy), np.abs(xx))

    def inside(r):
        return 0.5 * special.erfc((dist - r) / (np.sqrt(2) * edge_sigma))

    if thickness is None:
        frames = [BG + (FG - BG) * inside(r) for r in radii]
    else:
        frames = [BG + (FG - BG) * (inside(r + thickness / 2) - inside(r - thickness / 2))
                  for r in radii]
    return np.stack(frames)


def edge_radii(frames, shape, r_guess, n_angles=64):
    """Radius per frame and angle where the intensity crosses (BG + FG) / 2."""
    size = frames.shape[1]
    c = (size - 1) / 2
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    if shape == "circle":
        scale = np.ones_like(angles)
    else:  # distance to a square's edge along the ray, per unit half-side
        scale = 1 / np.maximum(np.abs(np.cos(angles)), np.abs(np.sin(angles)))
    t = np.linspace(0.3, 1.7, 281)[:, None] * r_guess  # samples along each ray
    ys = c + t * np.sin(angles) * scale
    xs = c + t * np.cos(angles) * scale
    mid = (BG + FG) / 2
    out = np.empty((len(frames), n_angles))
    for i, frame in enumerate(frames):
        prof = ndimage.map_coordinates(frame, [ys, xs], order=3)  # (len(t), angles)
        for a in range(n_angles):
            p = prof[:, a]
            # last crossing from inside (above mid) to outside (below mid)
            idx = np.nonzero((p[:-1] >= mid) & (p[1:] < mid))[0]
            if len(idx) == 0:
                out[i, a] = np.nan
                continue
            j = idx[-1]
            frac = (p[j] - mid) / (p[j] - p[j + 1])
            out[i, a] = t[j, 0] + frac * (t[j + 1, 0] - t[j, 0])
    return out


def _rays(size, shape, r_guess, span, n_angles=64):
    c = (size - 1) / 2
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    scale = (np.ones_like(angles) if shape == "circle"
             else 1 / np.maximum(np.abs(np.cos(angles)), np.abs(np.sin(angles))))
    t = r_guess + np.linspace(-span, span, int(span * 20) + 1)[:, None]
    return t[:, 0], c + t * np.sin(angles) * scale, c + t * np.cos(angles) * scale


def outline_profiles(frames, shape, r_guess, span):
    """Intensity profiles across the outline: (frames, samples, angles)."""
    t, ys, xs = _rays(frames.shape[1], shape, r_guess, span)
    return t, np.stack([ndimage.map_coordinates(f, [ys, xs], order=3) for f in frames])


def line_positions(t, profiles):
    """Position of the line on each profile: centroid of its brightness above
    the background (robust for thin and thick lines, and for ghosts)."""
    w = np.clip(profiles - BG, 0, None)
    return (t[None, :, None] * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)


def count_peaks(profiles, min_prominence=0.15):
    """Number of distinct bright peaks per profile (2+ = a doubled line).
    A peak must stand out by min_prominence * (FG - BG) from its surroundings,
    so a flat-topped thick line counts as one."""
    counts = np.zeros(profiles.shape[0:1] + profiles.shape[2:], dtype=int)
    for f in range(profiles.shape[0]):
        for a in range(profiles.shape[2]):
            peaks, _ = signal.find_peaks(profiles[f, :, a],
                                         prominence=min_prominence * (FG - BG))
            counts[f, a] = len(peaks)
    return counts


def analyse_outline(shape="circle", thickness=2.0, r0=30.0, r1=0.1, k=10.0, freq=1.5,
                    fps=30.0, n=240, noise=0.0, magnify=None, edge_sigma=0.8, trim=40):
    """Magnify a pulsating ring / hollow square and measure line fidelity.

    Returns gain (from the line's brightness centroid), the fraction of
    profiles showing a doubled line, the peak contrast relative to the ideal,
    and `ghost_energy`: energy of (output - ideal) across the line relative
    to the energy of the ideal line.
    """
    fps = float(fps)
    tt = np.arange(n) / fps
    wave = np.sin(2 * np.pi * freq * tt)
    clip = render(shape, r0 + r1 * wave, edge_sigma=edge_sigma, thickness=thickness)
    if noise:
        clip = clip + np.random.RandomState(0).normal(0, noise, clip.shape)
    if magnify is None:
        def magnify(x):
            return motion_mag.magnify_motions(x, magnification=k, width=80, nlevels=5)
    out = np.asarray(magnify(clip.astype(np.float32)), dtype=np.float64)
    ideal = render(shape, r0 + k * r1 * wave, edge_sigma=edge_sigma, thickness=thickness)

    keep = slice(trim, n - trim)
    span = k * abs(r1) + thickness + 6
    t, p_in = outline_profiles(clip[keep], shape, r0, span)
    _, p_out = outline_profiles(out[keep], shape, r0, span)
    _, p_ideal = outline_profiles(ideal[keep], shape, r0, span)
    pos_in = line_positions(t, p_in).mean(axis=1)
    pos_out = line_positions(t, p_out).mean(axis=1)
    a_in = fit_sine(pos_in, freq, fps)[0]
    a_out = fit_sine(pos_out, freq, fps)[0]
    line = (p_ideal - BG) ** 2
    return {
        "gain_over_k": a_out / a_in / k if a_in > 0 else np.nan,
        "doubled_fraction": float(np.mean(count_peaks(p_out) >= 2)),
        "doubled_fraction_ideal": float(np.mean(count_peaks(p_ideal) >= 2)),
        "peak_contrast": float(np.mean(p_out.max(axis=1) - BG) / np.mean(p_ideal.max(axis=1) - BG)),
        "ghost_energy": float(np.sum((p_out - p_ideal) ** 2) / np.sum(line)),
        "frames": (clip, out, ideal),
    }


def fit_sine(signal, freq, fps):
    """Least-squares amplitude, phase (radians) and residual of a sine at freq."""
    n = len(signal)
    tt = np.arange(n) / fps
    basis = np.stack([np.sin(2 * np.pi * freq * tt), np.cos(2 * np.pi * freq * tt),
                      np.ones(n)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, signal, rcond=None)
    fit = basis @ coef
    return np.hypot(coef[0], coef[1]), np.arctan2(coef[1], coef[0]), signal - fit


def analyse(shape="circle", r0=30.0, r1=0.1, k=10.0, freq=1.5, fps=30.0, n=240,
            noise=0.0, magnify=None, edge_sigma=0.8, trim=40):
    """Magnify a pulsating shape and measure it against the ideal.

    Args:
        magnify: function(frames float32) -> magnified frames. Default:
            magnify_motions with width 80 and nlevels 5.
        trim: frames dropped at each end (temporal filter edges).
    """
    fps = float(fps)
    tt = np.arange(n) / fps
    radii = r0 + r1 * np.sin(2 * np.pi * freq * tt)
    clip = render(shape, radii, edge_sigma=edge_sigma)
    if noise:
        clip = clip + np.random.RandomState(0).normal(0, noise, clip.shape)
    if magnify is None:
        def magnify(x):
            return motion_mag.magnify_motions(x, magnification=k, width=80, nlevels=5)
    out = np.asarray(magnify(clip.astype(np.float32)), dtype=np.float64)
    ideal = render(shape, r0 + k * r1 * np.sin(2 * np.pi * freq * tt), edge_sigma=edge_sigma)

    keep = slice(trim, n - trim)
    r_in = edge_radii(clip[keep], shape, r0)
    r_out = edge_radii(out[keep], shape, r0 + 0 * k * r1)
    mean_in, mean_out = np.nanmean(r_in, axis=1), np.nanmean(r_out, axis=1)
    a_in, ph_in, res_in = fit_sine(mean_in, freq, fps)
    a_out, ph_out, res_out = fit_sine(mean_out, freq, fps)
    # per-angle gain (isotropy)
    per_angle = np.array([fit_sine(r_out[:, j], freq, fps)[0] for j in range(r_out.shape[1])])
    # harmonic distortion of the mean radius
    spec = np.abs(np.fft.rfft(mean_out - mean_out.mean(), 8 * len(mean_out)))
    f_axis = np.fft.rfftfreq(8 * len(mean_out), 1 / fps)

    def peak(f):
        return spec[np.argmin(np.abs(f_axis - f))]

    lag = ((ph_in - ph_out + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi * freq) * fps
    diff = out[keep] - ideal[keep]
    return {
        "gain": a_out / r1,
        "gain_over_k": a_out / r1 / k,
        "input_gain": a_in / r1,
        "phase_lag_frames": lag,
        "harmonic2": peak(2 * freq) / peak(freq),
        "harmonic3": peak(3 * freq) / peak(freq),
        "angle_spread": np.std(per_angle) / np.mean(per_angle),
        "angle_min_max": (per_angle.min() / r1, per_angle.max() / r1),
        "jitter_in": np.nanstd(res_in),
        "jitter_out": np.nanstd(res_out),
        "mean_radius_offset": np.nanmean(r_out) - np.nanmean(r_in),
        "rmse_vs_ideal": float(np.sqrt(np.mean(diff ** 2))),
        "overshoot": float(max(out[keep].max() - FG, BG - out[keep].min(), 0)),
        "frames": (clip, out, ideal),
    }
