"""Generate the figures for docs/theory.md from the real pipeline code.

Needs matplotlib, which is not a runtime or dev dependency:
    pip install matplotlib
    python scripts/make_theory_figures.py            # writes docs/images/theory/*.png

It also prints the per-level numbers quoted in docs/theory.md (phase slope,
effective wavelength, and the displacement at which the phase change reaches
a quarter and a half turn).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import motion_mag  # noqa: E402

OUT = os.path.join(ROOT, "docs", "images", "theory")
FPS = 30.0
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
GREY = "#8a9097"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e9ecef", "axes.edgecolor": "#b9c0c7",
                     "figure.dpi": 110})


def transform():
    return motion_mag._transform("near_sym_b", "qshift_b")


def face_luma():
    channels, _, _ = motion_mag.load_video(os.path.join(ROOT, "face.mp4"))
    return motion_mag.luma(channels)


def shifted(image, dx):
    """Image shifted right by dx pixels (Fourier shift: exact sub-pixel)."""
    return np.real(np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(image), (0, dx))))


def fig_subbands(frame):
    """Magnitude of the six orientations at levels 2 and 3 of a face crop."""
    crop = frame[150:406, 136:392].astype(np.float64)
    p = transform().forward(crop, nlevels=3)
    # dtcwt orders the six sub-bands as +15, +45, +75, -75, -45, -15 degrees
    angles = ["+15°", "+45°", "+75°", "−75°", "−45°", "−15°"]
    fig, axes = plt.subplots(3, 7, figsize=(11, 5.2),
                             gridspec_kw={"width_ratios": [1.4] + [1] * 6})
    for row, level in enumerate((1, 2, 3)):
        hp = np.abs(p.highpasses[level - 1])
        ax = axes[row, 0]
        ax.imshow(crop, cmap="gray")
        ax.set_title("input (256×256)" if row == 0 else "", fontsize=9)
        ax.set_ylabel(f"level {level}\n{hp.shape[0]}×{hp.shape[1]}", fontsize=9)
        for o in range(6):
            a = axes[row, o + 1]
            a.imshow(hp[:, :, o], cmap="magma")
            if row == 0:
                a.set_title(angles[o], fontsize=9)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.suptitle("DTCWT magnitude |C| per level and orientation (face crop)", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dtcwt_subbands.png"), dpi=80)
    plt.close(fig)


def fig_phase_vs_shift(frame):
    """Amplitude-weighted phase change per level as the image shifts."""
    crop = frame[170:298, 150:278].astype(np.float64)
    nlevels = 5
    base = transform().forward(crop, nlevels=nlevels).highpasses
    shifts = np.round(np.arange(-4, 4.01, 0.1), 2)
    per_level = {lv: [] for lv in range(nlevels)}
    for dx in shifts:
        hp = transform().forward(shifted(crop, dx), nlevels=nlevels).highpasses
        for lv in range(nlevels):
            per_level[lv].append(hp[lv] * np.conj(base[lv]))
    rows = []
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for lv in range(nlevels):
        # ignore coefficients near the border, where the Fourier shift wraps
        size = base[lv].shape[0]
        m = size // 8 if size >= 16 else 0
        inner = slice(m, size - m)
        prod = np.stack(per_level[lv])[:, inner, inner, :]  # (shifts, H, W, 6)
        w = np.abs(base[lv][inner, inner, :]) ** 2
        # pick the orientation whose phase follows a horizontal shift fastest
        i0 = int(np.argmin(np.abs(shifts - 0.1)))
        slopes = [np.angle((prod[i0, :, :, o]).sum()) / 0.1 for o in range(6)]
        o = int(np.argmax(np.abs(slopes)))
        # mean phase change of that orientation, weighted by |C|^2
        dphi = np.angle((prod[:, :, :, o] / np.maximum(np.abs(prod[:, :, :, o]), 1e-12)
                         * w[None, :, :, o]).sum(axis=(1, 2)))
        slope = abs(slopes[o])
        rows.append((lv + 1, o, slope, 2 * np.pi / slope, (np.pi / 2) / slope, np.pi / slope))
        ax.plot(shifts, dphi, color=C[lv], lw=2, label=f"level {lv + 1}")
    ax.axhline(np.pi, color=GREY, lw=1, ls="--")
    ax.axhline(-np.pi, color=GREY, lw=1, ls="--")
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels(["−π", "−π/2", "0", "π/2", "π"])
    ax.set_xlabel("shift of the whole image (px)")
    ax.set_ylabel("phase change Δφ")
    ax.set_title("Phase follows position, then wraps (fine levels first)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "phase_vs_shift.png"))
    plt.close(fig)
    return rows


def _haar_energy(x, level):
    for _ in range(level - 1):
        x = (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2]) / 2
    a, b, c, d = x[0::2, 0::2], x[1::2, 0::2], x[0::2, 1::2], x[1::2, 1::2]
    details = [(a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2]
    return sum(float((q ** 2).sum()) for q in details)


def fig_shift_invariance():
    """Energy per level of a small blob as it moves by sub-pixel steps:
    DTCWT vs ordinary (Haar) DWT. The blob sits far from the borders, so
    only the transform's own shift dependence shows."""
    size = 128
    yy, xx = np.mgrid[:size, :size] - size / 2
    shifts = np.round(np.arange(0, 8.01, 0.125), 3)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, level in enumerate((2, 3)):
        dt, haar = [], []
        for dx in shifts:
            blob = np.exp(-((xx - dx) ** 2 + yy ** 2) / (2 * 1.5 ** 2))
            hp = transform().forward(blob, nlevels=level).highpasses[level - 1]
            dt.append(float((np.abs(hp) ** 2).sum()))
            haar.append(_haar_energy(blob, level))
        ax.plot(shifts, np.array(dt) / np.mean(dt), color=C[i], lw=2, label=f"DTCWT level {level}")
        ax.plot(shifts, np.array(haar) / np.mean(haar), color=C[i], lw=2, ls="--",
                label=f"Haar DWT level {level}")
    ax.set_xlabel("position of the blob (px)")
    ax.set_ylabel("energy / mean energy")
    ax.set_title("Shift dependence: DTCWT energy stays nearly constant, DWT energy swings",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "shift_invariance.png"))
    plt.close(fig)
    return shifts, dt, haar


def _response(width, f_hz):
    win = motion_mag._flattop_window(width)
    n = np.arange(len(win)) - len(win) // 2
    return np.array([np.real(np.sum(win * np.exp(-2j * np.pi * f / FPS * n))) for f in f_hz])


def fig_temporal_filters():
    f = np.logspace(-2, np.log10(FPS / 2), 400)
    k = 10
    h_low, h2 = _response(80, f), _response(2.0, f)
    gain = h2 * (1 + (k - 1) * (1 - h_low))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(f, h_low, color=C[0], lw=2, label="baseline low-pass Hlow (-w 80)")
    a1.plot(f, 1 - h_low, color=C[0], lw=2, ls="--", label="detail = 1 − Hlow")
    a1.plot(f, h2, color=C[1], lw=2, label="smoothing H2 (width 2)")
    a1.set_xscale("log")
    a1.set_xlabel("motion frequency (Hz, 30 fps)")
    a1.set_ylabel("response")
    a1.set_ylim(-0.1, 1.1)
    a1.set_title("Width mode: the two flat-top filters", loc="left", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    a2.plot(f, gain, color=C[2], lw=2, label="width mode, k = 10")
    band = np.where((f >= 0.8) & (f <= 2.0), k, 1)
    a2.plot(f, band, color=C[1], lw=2, ls="--", label="band mode 0.8–2 Hz, k = 10")
    a2.set_xscale("log")
    a2.set_xlabel("motion frequency (Hz, 30 fps)")
    a2.set_ylabel("motion gain (×)")
    a2.set_title("Resulting magnification", loc="left", fontsize=10)
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "temporal_filters.png"))
    plt.close(fig)


def fig_phase_trace(luma):
    """One coefficient's phase over time in face.mp4, through the pipeline steps."""
    level, k = 3, 10
    pyramids = [transform().forward(f.astype(np.float64), nlevels=level) for f in luma]
    coeffs = np.stack([p.highpasses[level - 1] for p in pyramids])  # (N, H, W, 6)
    amp = np.abs(coeffs).mean(axis=0)
    phase_all = motion_mag.temporal_phase(coeffs.reshape(len(luma), -1))
    detail_all = phase_all - motion_mag.flattop_filter_1d(phase_all, 80)
    # a strong coefficient with clear pulse motion
    score = amp.ravel() * detail_all[40:-40].std(axis=0)
    j = int(np.argmax(score))
    phase = phase_all[:, j].astype(np.float64)
    phi0 = motion_mag.flattop_filter_1d(phase[:, None], 80)[:, 0]
    magnified = motion_mag.flattop_filter_1d((phi0 + (phase - phi0) * k)[:, None], 2.0)[:, 0]
    t = np.arange(len(phase)) / FPS
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    a1.plot(t, phase - phase[0], color=C[0], lw=1.6, label="φ(t): cumulative phase")
    a1.plot(t, phi0 - phase[0], color=C[1], lw=2, label="φ0(t): slow baseline (low-pass)")
    a1.set_ylabel("phase (rad)")
    a1.legend(frameon=False, fontsize=8.5)
    a1.set_title(f"One level-{level} coefficient of face.mp4", loc="left", fontsize=10)
    a2.plot(t, phase - phi0, color=C[2], lw=1.6, label="detail φ − φ0")
    a2.plot(t, magnified - phi0, color=C[3], lw=1.6, label=f"after ×{k} and smoothing, minus φ0")
    a2.set_xlabel("time (s)")
    a2.set_ylabel("phase (rad)")
    a2.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "phase_trace.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    luma = face_luma()
    fig_subbands(luma[0])
    rows = fig_phase_vs_shift(luma[0])
    fig_shift_invariance()
    fig_temporal_filters()
    fig_phase_trace(luma)
    print("level  slope(rad/px)  wavelength(px)  quarter-turn shift(px)  half-turn shift(px)")
    for lv, _, slope, lam, quarter, half in rows:
        print(f"{lv:5d}  {slope:13.3f}  {lam:14.2f}  {quarter:22.2f}  {half:19.2f}")


if __name__ == "__main__":
    main()
