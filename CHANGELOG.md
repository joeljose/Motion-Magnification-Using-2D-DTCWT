# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `--jobs N`: the CPU path runs the DTCWT in worker processes over blocks of frames (shared memory) and the phase step in threads over coefficient chunks. Default 2 workers: face.mp4 takes 56 s instead of 94 s, with lower peak memory (2.3 GiB). Output is identical for any `--jobs` (#36)
- `--color-space yiq`: magnify luma only and keep the input's chroma. About 3.4x faster on the CPU and no colour fringing; the default stays `rgb`, so existing output is unchanged (#37)
- `--freq-low` / `--freq-high`: set the amplified band in Hz; uses an ideal temporal band-pass (CPU and GPU). The Parameters output shows the band in Hz in both modes. `-w` still works (and stays the default) but is deprecated (#38)
- `--phase-sigma`: amplitude-weighted spatial smoothing of the amplified phase (CPU), off by default. Synthetic validation with pulsating shapes (`scripts/synthetic_shapes.py`, `tests/test_synthetic_shapes.py`) and a high-k benchmark (`scripts/bench_high_k.py`); the README documents the results, including why output motion is somewhat below k and the ~3 px displacement limit (#39)

### Fixed
- CI failing on new ruff releases: dev tools pinned, lint rules set in `pyproject.toml` (#24)
- CI now runs the unit tests and checks the output video (#25)
- NaN / black regions from exact-zero wavelet coefficients on the CPU path (#22)
- Output write failures now exit 1 instead of reporting success (#23)
- CPU peak RAM on face.mp4 cut from 8.3 GiB to 2.9 GiB (float32/complex64, chunked phase filtering, uint8 frames); `estimate_memory()` uses the real coefficient count and is now called from `main()` (#26)
- GPU Docker image ran the CPU path by default; its entrypoint now passes `--gpu`, and the CLI prints the active backend (#27)
- Notebook failed on SciPy >= 1.13 and wrapped pixel values; it now calls `motion_mag` instead of keeping its own copy of the algorithm (#28)
- Temporal filter was half a frame off centre (even window lengths) and the FFT and direct paths handled clip edges differently; windows are now always odd and all paths use the same boundary rule. Output changes slightly, mostly near the first and last frames (#29)
- Odd frame sizes crashed the CPU path at the inverse DTCWT (#30)
- Unreadable input now exits 1 with a clear error; frames beyond an under-reported frame count are no longer dropped; a missing frame rate is an error unless the new `--fps` is given; clips shorter than 3 frames are rejected (#31)
- Tests: weak assertions tightened; added a golden regression test (`tests/data/golden_face.npz`, regenerated with `scripts/make_golden.py`), a test that motion is magnified about k times, and CLI runs on a real clip (#32)
- Band mode: a band containing no frequency bins silently left the video unmagnified; it is now rejected before processing (CPU, GPU and `bandpass_1d`), a band narrower than the clip's resolution warns, and the band's bin count and resolution are printed. The Nyquist check now runs before the Parameters output (#67)

### Changed
- Reproducible builds: Docker images install from hash-locked `requirements.lock` / `requirements-gpu.lock`, `pytorch_wavelets` is pinned to a commit, base images and GitHub Actions are pinned by digest/SHA, and Dependabot watches them. `opencv-python-headless` is used everywhere. Images use a fixed non-root user, so `docker build .` needs no build args; run with `--user "$(id -u):$(id -g)"` for bind mounts (#33)
- GPU path: batches and FFT chunks that run out of GPU memory are retried at half the size; an invalid `--device` gives a clean error; the GPU code runs on CPU tensors too, and CI runs its tests with CPU PyTorch (#34)

- Output pixels were truncated to uint8 instead of rounded (about 8% of pixels one level dark); non-finite `-k`/`-w` values are rejected; unknown `--biort`/`--qshift` names are rejected before loading; `magnify_motions_gpu` accepts float64 input (#43)

### Documentation
- README, docstrings and the GPU design doc now match the code: GPU padding, VRAM fractions, batch equivalence, the meaning of the 0.2327 constant (with the amplified band in Hz), CPU/GPU agreement, supported Python versions; a test checks `VERSION` matches `__version__` (#35)

## [2.0.0] - 2026-03-21

### Added
- GPU-accelerated motion magnification via `--gpu` flag (~5x speedup on RTX 4050)
- `--device` flag for CUDA GPU selection
- `--biort` and `--qshift` flags for wavelet filter selection
- `Dockerfile.gpu` based on PyTorch 2.1.2 + CUDA 12.1
- `docker-build-gpu.sh` build script
- `requirements-gpu.txt` (scipy, numpy<2, opencv, dtcwt, PyWavelets)
- GPU test suite (`tests/test_motion_mag_gpu.py`) — 14 tests, skips on CPU-only
- Pre-flight memory estimation (`estimate_memory()`)
- GPU design doc (`docs/design/gpu-acceleration.md`)

### Changed
- **BREAKING**: Default DTCWT filters changed from `near_sym_a`/`qshift_a` to `near_sym_b`/`qshift_b` (fewer block artifacts at higher magnification). Use `--biort near_sym_a --qshift qshift_a` to restore old behavior.
- CPU temporal filter uses FFT-based convolution for large windows (4x faster, 2x total speedup)
- `test.sh` supports `gpu` mode (`./test.sh gpu`)

## [1.1.0] - 2026-03-20

### Added
- Unit tests for core functions (normalize_phase, flattop_filter, extract_temporal_phases, magnify_motions)
- Input validation tests for all CLI error paths
- `VERSION` file as single source of truth for versioning
- Docker image version labels and tags
- `CHANGELOG.md`
- `requirements-dev.txt` for dev dependencies (pytest, ruff)
- `test.sh` for running lint and tests inside Docker
- Design doc (`docs/design/dtcwt-hardening.md`)
- Development section in README (testing, versioning, project structure)

### Fixed
- `load_video` buffer overflow when `CAP_PROP_FRAME_COUNT` underreports
- `fps` truncation: keep as float instead of casting to int (3.2% speed error on 29.97fps videos)
- `flattop_filter_1d` window size guard against zero
- Ruff F541 lint error (extraneous f-prefix)

### Changed
- CI modernized: runs inside Docker with ruff linting, updated to actions/checkout v6
- Dev dependencies (pytest, ruff) baked into Docker image
- Build script reads version from `VERSION` file, tags image accordingly
- `dtcwt` dependency pinned with upper bound (`<1`)

### Removed
- `np.int` monkey-patch — no longer needed with dtcwt 0.14.0
