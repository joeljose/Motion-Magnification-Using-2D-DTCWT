# Design Doc: DTCWT Motion Magnification — Project Hardening

**Status: APPROVED**

## Context

The Motion Magnification Using 2D DTCWT project is a Python implementation of phase-based motion magnification (Anfinogentov & Nakariakov, 2016) using the Dual-Tree Complex Wavelet Transform. It's a single-file CLI tool (`motion_mag.py`) shipped via Docker.

A code review identified the same gaps as the sibling EVM project: no unit tests, no versioning infrastructure, no design documentation, and minor bugs. This design doc covers retroactive architecture decisions and the hardening plan.

## Goals and Non-Goals

**Goals:**
- Add unit tests for core functions
- Fix bugs: `load_video` buffer overflow, `fps` truncation, remove dead `np.int` monkey-patch
- Establish versioning infrastructure (VERSION file, Docker labels, changelog)
- Modernize CI (Docker-based, add linting)
- Add `test.sh`, `requirements-dev.txt`, and dev workflow documentation
- Pin `dtcwt` upper bound
- Document architecture decisions

**Non-Goals:**
- Refactoring into a pip-installable package
- Integration tests with video output comparison
- Regenerating demo GIFs (deferred to after GPU implementation)

## Proposed Design

### A. Architecture Decisions (retroactive)

**Why 2D DTCWT over complex steerable pyramids?**
The original Wadhwa et al. (SIGGRAPH 2013) phase-based method uses complex steerable pyramids, which require custom filter bank implementations. The 2D DTCWT (Selesnick et al., 2005) provides similar properties — near shift-invariance, directional selectivity (6 orientations per level) — with an off-the-shelf Python library (`dtcwt`). Anfinogentov & Nakariakov (2016) demonstrated this substitution works for motion magnification with comparable quality. The tradeoff: DTCWT has fixed 6 orientations per level vs steerable pyramids' configurable orientation count, but 6 is sufficient for general video motion.

**Why flat-top window for temporal filtering?**
The flat-top window (scipy.signal.windows.flattop) has a very flat passband, meaning signals within the passband are preserved with minimal amplitude distortion. For motion magnification, this matters because amplitude distortion in the temporal filter would non-uniformly scale phase changes, creating visible artifacts. The tradeoff: flat-top has wider transition band (slower rolloff) than alternatives like Hann or Hamming, so frequency selectivity is lower. For the typical use case (separating slow base motion from subtle detail motion), this is acceptable.

**Why frame-to-frame complex division for phase extraction?**
Phase could be extracted by computing `np.angle()` of each frame's coefficients directly, then differencing. Instead, the code computes `curr / prev` (complex division) and takes `np.angle()` of the ratio. This is more numerically stable because: (1) it avoids phase wrapping issues at ±π boundaries (the ratio's angle is always the correct small delta), (2) it works correctly even when individual coefficients have very small magnitude (the ratio's angle is still meaningful). The cumulative sum of these deltas gives the total phase evolution.

**Why float64 throughout?**
Verified: dtcwt produces `complex128` highpass coefficients (float64 real/imaginary pairs). Float64 roundtrip error is `4.4e-16` vs `3.6e-07` for float32. Phase extraction via cumulative complex division compounds errors across frames — with 300+ frames and 8 pyramid levels, float32 errors could become visible. Memory cost is 2x, but correctness is more important for a research tool.

### B. Hardening Changes

All follow the same patterns established in EVM hardening:

**Bug fixes:**
- Remove `np.int` monkey-patch — verified dtcwt 0.14.0 doesn't need it (numpy 1.26.4, no `np.int` references in dtcwt source)
- Fix `fps` truncation: `int(cap.get(...))` → keep as float (verified `cv2.VideoWriter` accepts float fps, 3.2% speed error on 29.97fps videos)
- Add `load_video` buffer guard: `if i >= frame_count: break`
- `flattop_filter_1d`: guard against zero window size with `max(1, ...)`

**Versioning:**
- Add `VERSION` file (Approach B — build-time injection, same as EVM)
- Stamp `__version__` in `motion_mag.py` at release time
- `docker-build.sh` reads from `VERSION`, tags image, passes build arg for Docker label
- Dockerfile gets `ARG VERSION` + `LABEL version=${VERSION}`

**Testing:**
- `requirements-dev.txt` with pytest and ruff
- `tests/test_motion_mag.py` — unit tests:
  - Tier 1 (strict): `normalize_phase` (unit magnitude property, zero-magnitude safety), `format_duration`
  - Tier 2 (moderate): `flattop_filter_1d` (DC passthrough, smoothing effect), `extract_temporal_phases` (known constant phase → zero delta)
  - Tier 3 (smoke): `magnify_motions` on tiny synthetic data (shapes, dtypes, finite values)
  - Input validation error paths
- `test.sh` — build once, run lint + tests in Docker
- CI modernized: Docker-based, ruff lint, smoke tests only

**CI modernization:**
- Run inside Docker (matching dev workflow and EVM pattern)
- Add ruff linting
- Update `actions/checkout` to v6
- Remove `actions/setup-python` (not needed when running in Docker)

**Documentation:**
- `CHANGELOG.md` (Keep a Changelog, starting fresh)
- Design doc (this document)
- `dtcwt` upper bound pinned (`<1`)
- README updated with Development section (testing, versioning, project structure)
- CONTRIBUTING.md updated with `test.sh` instructions

### C. Future: GPU Acceleration (deferred, implement after hardening)

**Approach**: Same hybrid pattern as the Visual-Mic project (`joeljose/Visual-Mic`):
- CPU path: existing `dtcwt` library (unchanged)
- GPU path: `pytorch_wavelets` (`DTCWTForward`) with PyTorch CUDA backend
- `--gpu` flag switches between them, `--batch-size` controls GPU memory usage

**Key considerations for motion magnification vs Visual-Mic:**
- Visual-Mic does single-pass phase extraction (streaming) — motion magnification needs all pyramids in memory for temporal filtering
- GPU batching applies to forward/inverse DTCWT transforms (the expensive part)
- Temporal filtering (flat-top convolution on phase arrays) stays on CPU/numpy — it's already fast and operates on extracted phase angles, not full wavelet coefficients
- `pytorch_wavelets` uses float32 internally — phase extraction precision may differ from CPU float64 path. Needs validation.

**Dependencies**: `pytorch_wavelets` (installed from `git+https://github.com/fbcotter/pytorch_wavelets.git`), PyTorch with CUDA, `Dockerfile.gpu` based on PyTorch CUDA image.

**After GPU implementation**: Regenerate demo GIFs using the GPU path for faster processing.

## Alternatives Considered

### Temporal filter: Flat-top vs ideal bandpass (FFT-based, as in EVM)

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Flat-top window convolution | Minimal amplitude distortion in passband, simple | Wider transition band, less frequency selectivity | **Chosen** — amplitude fidelity matters for phase-based method |
| Ideal bandpass via FFT | Sharp frequency cutoff, matches EVM | Gibbs ringing, requires all frames for FFT | Rejected — ringing introduces phase artifacts |

### Phase extraction: Complex division vs direct angle subtraction

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Complex division + cumsum | No phase wrapping, numerically stable | Slightly more computation | **Chosen** — correctness |
| Direct `angle()` subtraction | Simpler code | Phase wrapping at ±π, needs unwrapping | Rejected — unwrapping adds complexity and failure modes |

### GPU framework: pytorch_wavelets vs custom CuPy DTCWT

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| pytorch_wavelets | Proven in Visual-Mic, maintained library, batched transforms | PyTorch is a large dependency, float32 only | **Chosen** (deferred) — minimal effort, proven approach |
| Custom CuPy DTCWT | No PyTorch dependency, could use float64 | Massive implementation effort, no existing library | Rejected — not worth writing from scratch |

## Tradeoffs and Risks

- **No GPU acceleration (current)**: DTCWT is CPU-bound and slow for large videos. Accepted for now, GPU path planned as next phase.
- **Memory intensive**: All frame pyramids stay in memory for temporal filtering. A 1080p 30s video at float64 needs ~42 GB for channels plus pyramid storage. Documented, not fixed.
- **Removing `np.int` patch**: If someone uses an older dtcwt (<0.14.0), it will break. Mitigated by pinning `dtcwt>=0.12.0,<1`.
- **GPU float32 vs CPU float64**: When GPU path is added, outputs may differ slightly from CPU. Will need validation and documentation.

## Open Questions

None for the hardening phase. GPU acceleration details to be resolved during that implementation.
