"""Sanity-check a magnified video against its input (used by CI)."""

import sys

import cv2
import numpy as np


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        sys.exit(f"FAIL: no frames could be read from {path}")
    return np.stack(frames)


def main(input_path, output_path):
    src = read_frames(input_path)
    out = read_frames(output_path)
    errors = []
    if out.shape != src.shape:
        errors.append(f"shape {out.shape} != input {src.shape}")
    else:
        black = [i for i, f in enumerate(out) if f.max() == 0]
        if black:
            errors.append(f"{len(black)} all-black frames, first at {black[0]}")
        src_mean, out_mean = src.mean(), out.mean()
        if abs(out_mean - src_mean) > 0.1 * src_mean:
            errors.append(f"mean intensity {out_mean:.1f} vs input {src_mean:.1f} (>10% off)")
    if errors:
        sys.exit("FAIL: " + "; ".join(errors))
    print(f"OK: {out.shape[0]} frames, {out.shape[2]}x{out.shape[1]}, "
          f"mean {out.mean():.1f} (input {src.mean():.1f})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
