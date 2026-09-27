"""Regenerate tests/data/golden_face.npz (a crop of face.mp4 and its output).

Run only when an output change is intended, and note it in CHANGELOG.md:
    python scripts/make_golden.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import motion_mag  # noqa: E402

# 20 frames of a 48x48 crop around the eyes; small enough to keep in git
FRAMES, ROWS, COLS = slice(0, 20), slice(208, 256), slice(240, 288)
PARAMS = dict(magnification=3.0, width=10, nlevels=3)


def magnify(channels):
    return np.stack([
        np.clip(np.rint(motion_mag.magnify_motions(c, **PARAMS)), 0, 255).astype(np.uint8)
        for c in channels
    ])


if __name__ == "__main__":
    channels, _, _ = motion_mag.load_video(os.path.join(ROOT, "face.mp4"))
    inp = np.stack([c[FRAMES, ROWS, COLS] for c in channels])
    np.savez_compressed(os.path.join(ROOT, "tests", "data", "golden_face.npz"),
                        input=inp, output=magnify(inp))
