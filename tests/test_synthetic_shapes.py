"""Correctness checks on pulsating shapes with exact ground truth.

A circle, ring or square of radius r0 + r1 * sin(2 pi f t) magnified by k
should pulse k times as much. See scripts/synthetic_shapes.py and the
"Synthetic validation" section of the README.
"""

import contextlib
import io
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import synthetic_shapes as ss  # noqa: E402

import motion_mag  # noqa: E402

FPS, K = 30.0, 10.0


def quiet(func, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        result = func(**kwargs)
    result.pop("frames", None)
    return result


def model_gain(freq, k=K, width=80):
    """Gain of the flat-top pipeline for motion at `freq` Hz:
    H2(f) * (1 + (k - 1) * (1 - Hlow(f))), from the actual windows."""
    nu = freq / FPS

    def response(w):
        win = motion_mag._flattop_window(w)
        n = np.arange(len(win)) - len(win) // 2
        return float(np.real(np.sum(win * np.exp(-2j * np.pi * nu * n))))

    return response(2.0) * (1 + (k - 1) * (1 - response(width)))


@pytest.mark.parametrize("freq", [0.3, 1.5, 8.0])
def test_frequency_response_matches_filter_model(freq):
    """A thin ring (little energy in the unmagnified lowpass band) follows the
    temporal filter model; measured about 1.03-1.05x the model."""
    r = quiet(ss.analyse_outline, thickness=1.0, r1=0.05, k=K, freq=freq, n=200, trim=40)
    ratio = r["gain_over_k"] * K / model_gain(freq)
    assert 0.9 <= ratio <= 1.15, ratio


def test_no_phase_lag():
    r = quiet(ss.analyse, r1=0.02, freq=1.5, n=180, trim=30)
    assert abs(r["phase_lag_frames"]) < 0.05


def test_filled_shape_gain_is_limited_by_lowpass_residual():
    """Documents the known limitation: motion carried by the unmagnified
    lowpass band is not amplified, so a filled circle reaches ~0.88k at
    nlevels=5 (the ring above reaches ~1.0k)."""
    r = quiet(ss.analyse, r1=0.02, freq=1.5, n=180, trim=30)
    assert 0.8 <= r["gain_over_k"] <= 0.95, r["gain_over_k"]


@pytest.mark.parametrize("freq, expected", [(0.3, 1.0), (1.2, K), (5.0, 1.0)])
def test_band_mode(freq, expected):
    def magnify(x):
        return motion_mag.magnify_motions(x, magnification=K, nlevels=5,
                                          band=(0.8 / FPS, 2.0 / FPS))
    r = quiet(ss.analyse_outline, thickness=1.0, r1=0.05, freq=freq, n=240, trim=50,
              magnify=magnify)
    assert r["gain_over_k"] * K == pytest.approx(expected, rel=0.15)


def test_identity_at_k1():
    def magnify(x):
        return motion_mag.magnify_motions(x, magnification=1.0, width=80, nlevels=5)
    r = quiet(ss.analyse, r1=0.5, k=1.0, magnify=magnify, n=180, trim=30)
    assert r["gain"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("shape", ["circle", "square"])
def test_isotropic(shape):
    """The six DTCWT orientations magnify all edge directions alike."""
    r = quiet(ss.analyse, shape=shape, r1=0.05, n=180, trim=30)
    assert r["angle_spread"] < 0.05, r["angle_spread"]


def test_thin_line_stays_single_for_moderate_motion():
    """A 1 px ring magnified to 2 px of motion keeps one line (no ghost) and
    most of its contrast; doubling only starts around 5+ px."""
    r = quiet(ss.analyse_outline, thickness=1.0, r1=0.2, n=180, trim=30)
    assert r["doubled_fraction"] == 0
    assert r["peak_contrast"] > 0.9
