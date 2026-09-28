"""Benchmark noise and artefacts at high magnification (issue #39).

Two measurements, each run for a list of option sets:

1. Synthetic scene with ground truth: a textured object (a crop of
   face.mp4) moving by amp * sin(2 pi f t) pixels over a static background,
   plus Gaussian sensor noise. Reported: PSNR of the object area against the
   ideal (clean, k-times-magnified) scene, the measured motion gain / k,
   the noise ratio far from the object, and the "halo" ratio in the
   background next to the object. Run for a small (0.05 px) and a large
   (0.3 px, i.e. 3 px after k=10) motion.
2. face.mp4 luma. Temporal standard deviation of static background pixels
   (the wall on the right, the dark area on the left), output / input.
   Lower is less amplified noise.

Usage (inside the Docker image, from the repository root):
    python scripts/bench_high_k.py [--quick] [--out results.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import motion_mag  # noqa: E402

FPS = 30.0
# Static background regions of face.mp4 (rows, cols)
BACKGROUND = [(slice(60, 250), slice(492, 528)), (slice(300, 420), slice(0, 30))]
# Option sets to compare; each is passed to magnify_motions
CONFIGS = {
    "baseline": {},
    "sigma0.5": {"phase_sigma": 0.5},
    "sigma1": {"phase_sigma": 1.0},
    "sigma2": {"phase_sigma": 2.0},
}


def _psnr(a, b):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def _shift(image_fft, dx):
    return np.real(np.fft.ifft2(ndimage.fourier_shift(image_fft, (0, dx))))


def _shift_gain(clip, shifts, mid):
    """Regression slope of the clip's per-frame horizontal shift on `shifts`."""
    ref = clip[mid].mean(axis=0)
    gx = np.gradient(ref, axis=1)[8:-8, 8:-8]
    est = np.array([-(gx * (f - ref)[8:-8, 8:-8]).sum() / (gx * gx).sum()
                    for f in clip[mid]])
    s = shifts[mid] - shifts[mid].mean()
    return float((est - est.mean()) @ s / (s @ s))


def _scene(luma0, n, amp, freq, k=1.0):
    """Textured object (face crop, soft round mask) moving horizontally by
    k * amp * sin(2 pi freq t) over a static, low-contrast background."""
    size, obj = 192, 96
    rng = np.random.RandomState(1)
    # detail and contrast similar to face.mp4's background (high-pass std ~5)
    background = (110 + 30 * ndimage.gaussian_filter(rng.randn(size, size), 1.5)
                  + 400 * ndimage.gaussian_filter(rng.randn(size, size), 15))
    texture = np.full((size, size), 0.0)
    c0 = (size - obj) // 2
    texture[c0:c0 + obj, c0:c0 + obj] = luma0[170:170 + obj, 150:150 + obj]
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2
    mask = ndimage.gaussian_filter((np.hypot(yy, xx) < obj / 2 - 4).astype(float), 2)
    layer_fft, mask_fft = np.fft.fft2(texture * mask), np.fft.fft2(mask)
    shifts = amp * np.sin(2 * np.pi * freq * np.arange(n) / FPS)
    frames = np.stack([
        background * (1 - _shift(mask_fft, k * s)) + _shift(layer_fft, k * s)
        for s in shifts])
    dist = np.hypot(yy, xx) - obj / 2  # distance from the object's edge
    return frames, shifts, dist


def synthetic(luma0, k, amp, noise, options, n=150, freq=1.2):
    """Scene with ground truth. Returns:
    psnr_object: PSNR vs the ideal k-times-magnified clean scene, object area
    gain_over_k: measured object motion gain / k (1 = full magnification)
    noise_far: output / input temporal std in background far from the object
    halo_near: output / input temporal std in background 4-24 px from the edge
    """
    clean, shifts, dist = _scene(luma0, n, amp, freq)
    ideal, _, _ = _scene(luma0, n, amp, freq, k=k)
    noisy = clean + np.random.RandomState(0).normal(0, noise, clean.shape)
    out = motion_mag.magnify_motions(noisy.astype(np.float32), magnification=k,
                                     width=80, nlevels=5, **options)
    mid = slice(25, n - 25)
    obj = dist < 0
    near = (dist > 4) & (dist < 24)
    far = dist > 60

    def temporal_std(clip, region):
        return clip[mid][:, region].std(axis=0).mean()

    obj_box = (slice(56, 136), slice(56, 136))
    gain = (_shift_gain(out[:, obj_box[0], obj_box[1]], shifts, mid)
            / _shift_gain(clean[:, obj_box[0], obj_box[1]], shifts, mid) / k)
    return {
        "psnr_object": _psnr(out[mid][:, obj], ideal[mid][:, obj]),
        "gain_over_k": gain,
        "noise_far": temporal_std(out, far) / temporal_std(noisy, far),
        "halo_near": temporal_std(out, near) / temporal_std(noisy, near),
    }


def background_noise(luma, k, options, band=None):
    """Output / input temporal std in the static background of face.mp4."""
    out = motion_mag.magnify_motions(luma, magnification=k, nlevels=8, band=band,
                                     **options)
    ratios = []
    for rows, cols in BACKGROUND:
        before = luma[:, rows, cols].std(axis=0).mean()
        after = out[:, rows, cols].std(axis=0).mean()
        ratios.append(after / before)
    return float(np.mean(ratios)), out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="synthetic tests only")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--k", type=float, default=10)
    parser.add_argument("--out", default=None, help="write results as JSON")
    parser.add_argument("--save-video", default=None,
                        help="directory to save each config's face.mp4 luma as .npy")
    args = parser.parse_args()

    channels, _, _ = motion_mag.load_video(os.path.join(ROOT, "face.mp4"))
    luma = motion_mag.luma(channels)
    del channels

    results = {}
    for name in args.configs:
        options = CONFIGS[name]
        t0 = time.time()
        row = {}
        for amp in (0.05, 0.3):
            r = synthetic(luma[0], args.k, amp, noise=2.0, options=options)
            row.update({f"amp{amp}_{key}": v for key, v in r.items()})
        if not args.quick:
            row["bg_noise_ratio"], out = background_noise(luma, args.k, options)
            if args.save_video:
                np.save(os.path.join(args.save_video, f"{name}.npy"),
                        np.clip(np.rint(out), 0, 255).astype(np.uint8))
        row["seconds"] = time.time() - t0
        results[name] = row
        print(name, {key: round(v, 3) for key, v in row.items()}, flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
