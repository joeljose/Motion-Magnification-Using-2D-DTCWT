# Design Doc: GPU-Accelerated DTCWT Motion Magnification

**Status: APPROVED**

## Context

CPU-based motion magnification takes ~2 minutes for a 301-frame 528×592 video (3 channels). Profiling showed the temporal filter (flat-top convolution) dominates at 54% of runtime, not the DTCWT transforms (26%). A CPU-only FFT filter optimization (already applied) cuts this to ~2 minutes by switching to `scipy.signal.fftconvolve` with chunked reflect-padding for 2x total speedup.

GPU acceleration via `pytorch_wavelets` + `torch.fft` delivers ~17x speedup per channel in prototype testing. This design doc covers adding `--gpu` support to the existing `motion_mag.py`.

**PRD**: https://github.com/joeljose/Motion-Magnification-Using-2D-DTCWT/issues/10

## Goals and Non-Goals

**Goals:**
- Add `--gpu` and `--device` flags to `motion_mag.py`
- GPU-accelerated forward/inverse DTCWT via `pytorch_wavelets`
- GPU-accelerated temporal filtering via `torch.fft` (cuFFT)
- Pre-flight memory check (CPU RAM + VRAM) before processing
- Auto-tuned batch sizes and filter chunk sizes based on available VRAM
- Switch default wavelet filters to `near_sym_b`/`qshift_b` (both paths)
- Add `--biort`/`--qshift` CLI flags for wavelet filter selection
- `Dockerfile.gpu`, `docker-build-gpu.sh`, `requirements-gpu.txt`
- GPU unit tests (`tests/test_motion_mag_gpu.py`)

**Non-Goals:**
- Multi-GPU support (single GPU only, `--device` selects which)
- Separate `motion_mag_gpu.py` file (GPU path lives in `motion_mag.py`)
- Matching CPU/GPU output exactly (different libraries, different precision — accepted)
- Mixed-precision (float16) or `torch.compile` support
- GPU-accelerated video I/O

## Proposed Design

### A. Single-File Architecture

The GPU path is added to `motion_mag.py` as an alternative `magnify_motions_gpu()` function, called when `--gpu` is passed. Shared code (`load_video`, `save_video`, `format_duration`, `normalize_phase`, `flattop_filter_1d`, CLI parsing) remains unchanged. PyTorch and `pytorch_wavelets` are imported only when `--gpu` is used, guarded by try/except.

```
main()
  ├── parse args (shared)
  ├── load_video() (shared — CPU, float64 for CPU path, float32 for GPU)
  ├── validate + pre-flight memory check
  ├── if --gpu:
  │     for each channel:
  │       magnify_motions_gpu(channel, ...)
  │   else:
  │     for each channel:
  │       magnify_motions(channel, ...)  # existing CPU path
  └── save_video() (shared)
```

### B. GPU Pipeline: Two-Pass Batched Architecture

The GPU pipeline processes each channel independently (3× C=1, not C=3), using two GPU-accelerated passes with CPU-resident phase storage between them. This was chosen because C=3 batching only speeds up the DTCWT transforms (22% of pipeline) by 2.2x while adding complexity — the temporal filter (largest bottleneck) processes channels independently regardless.

#### Pass 1: Forward DTCWT + Phase Extraction

Process frames in GPU batches of size B (auto-tuned):

```
for each batch of B frames:
    1. Transfer (B, 1, H, W) float32 to GPU
    2. DTCWTForward → Yl (discard), Yh[level] = (B, 1, 6, H_l, W_l, 2)
    3. For each level:
       - Extract real/imag: Yh[level][..., 0], Yh[level][..., 1]
       - Compute magnitude, normalize to unit magnitude
       - Vectorized phase deltas: conjugate multiply curr[1:] * conj(curr[:-1])
       - Cross-batch boundary: carry prev_normalized from last frame of prior batch
       - torch.atan2 → phase deltas
       - Transfer to CPU float32
    4. Free GPU memory: del batch, Yl, Yh
```

After all batches: `np.cumsum(deltas, axis=0)` on CPU gives cumulative phase.

**Key detail — cross-batch boundary**: The last frame's normalized coefficients are kept as a small GPU tensor (~7 MB for L0 at 528×592) and used as the reference for the first frame of the next batch. Verified: produces bitwise-identical results to single-batch processing.

**Why no amplitude storage**: Amplitudes are NOT stored during Pass 1. They are recomputed by re-running forward DTCWT in Pass 2. Verified: forward DTCWT is deterministic (0.00 diff between runs). This saves ~718 MB CPU RAM per channel.

**Why no Yl storage**: Yl (lowpass) carries 94.6% of image energy and is essential for reconstruction. But it's recovered for free by re-running forward DTCWT in Pass 2 (deterministic, 0.00 diff).

#### Temporal Filtering: Chunked cuFFT

Phase arrays are processed on GPU using `torch.fft`, chunked along the coefficient dimension to fit in VRAM:

```
for each level:
    phase_array shape: (num_frames, num_coeffs)  # CPU, float32

    chunk_size = auto_tune_chunk_size(num_frames, free_vram)

    for each chunk of chunk_size coefficients:
        1. Transfer chunk (num_frames, chunk_size) to GPU
        2. torch.fft.rfft along time axis (dim=0)
        3. Multiply by pre-computed FFT of flat-top window
        4. torch.fft.irfft → phase0 (base motion)
        5. detail = phase - phase0; phase = phase0 + detail * magnification
        6. Repeat steps 2-4 with width=2 smoothing window
        7. Transfer modified phase chunk back to CPU
        8. Free GPU memory
```

**Why chunking is necessary**: cuFFT VRAM overhead is ~20× the array size (FFT buffers + complex intermediates). For face.mp4 at 301 frames, L0 phase array is 538 MB → needs ~3.8 GB for whole-array FFT, which OOMs on 6 GB GPUs. Chunking with 2 chunks uses ~3.9 GB peak and works.

**Chunk size auto-tuning**: Query `torch.cuda.mem_get_info()` for free VRAM, divide by estimated per-coefficient FFT overhead (20× frame count × 4 bytes), use 70% of that as chunk size. This adapts to any GPU without user configuration.

**Boundary handling**: cuFFT uses zero-padding, not reflect-padding. This produces ~1.3% relative error at video boundaries vs the CPU reflect-padded approach. At 65+ dB PSNR, this is visually imperceptible. The GPU path does NOT attempt to replicate reflect-padding (would require padding data before FFT, increasing memory usage).

#### Pass 2: Coefficient Reconstruction + Inverse DTCWT

Re-run forward DTCWT to recover Yl and amplitudes, apply modified phases, then inverse:

```
for each batch of B frames:
    1. Transfer original (B, 1, H, W) frames to GPU
    2. DTCWTForward → Yl, Yh (recovers Yl + amplitudes)
    3. For each level:
       - Extract amplitude: sqrt(real² + imag²)
       - Load modified phase from CPU
       - Reconstruct: real = amp * cos(phase), imag = amp * sin(phase)
       - Stack to (B, 1, 6, H_l, W_l, 2)
    4. DTCWTInverse((Yl, Yh_modified)) → reconstructed frames
    5. Transfer to CPU, free GPU memory
```

### C. Wavelet Filter Selection

Both `dtcwt` (CPU) and `pytorch_wavelets` (GPU) support the same named filter banks. The default is changed from `near_sym_a`/`qshift_a` to `near_sym_b`/`qshift_b` for both paths, based on visual testing that showed fewer block artifacts with the longer `near_sym_b` filters.

New CLI flags:
- `--biort` (default: `near_sym_b`) — biorthogonal filter for level 1
- `--qshift` (default: `qshift_b`) — quarter-shift filter for levels 2+

Available options (both libraries support the same set):
- biort: `antonini`, `legall`, `near_sym_a`, `near_sym_b`
- qshift: `qshift_06`, `qshift_a`, `qshift_b`, `qshift_c`, `qshift_d`

Note: Changing the default from `near_sym_a` to `near_sym_b` is a **breaking change** for users who expect identical output. This is acceptable because: (1) we're bumping to v2.0.0, (2) the visual quality improvement justifies it, (3) users can pass `--biort near_sym_a --qshift qshift_a` to restore old behavior.

### D. Pre-Flight Memory Check

Before processing, estimate peak memory usage and compare against available resources:

```
CPU RAM estimate:
    frames:       num_frames × H × W × 3 × 8 bytes (float64, CPU path)
                  num_frames × H × W × 3 × 4 bytes (float32, GPU path)
    phase arrays: num_frames × total_coeffs × 4 bytes (float32, GPU path only)
    total ≈ num_frames × H × W × 28 bytes (GPU path)

VRAM estimate (GPU path):
    DTCWT batch:  batch_size × 13 MB (input + coefficients)
    cuFFT chunk:  chunk_coeffs × num_frames × 80 bytes (20× overhead)
    overhead:     ~300 MB (PyTorch, filter weights, buffers)

Threshold: 70% of total RAM / VRAM
```

If estimated usage exceeds threshold, print a clear error with the numbers and suggestions (reduce nlevels, resolution, or use CPU mode). No `--force` flag — just an informational warning that processing may be slow or fail.

### E. GPU Dependencies and Docker

**`requirements-gpu.txt`:**
```
scipy>=1.7.1,<2
numpy>=1.20.3,<2
opencv-python-headless>=4.7.0,<5
PyWavelets>=1.1.0
```

Note: PyTorch is NOT listed because it's provided by the Docker base image. `numpy<2` is required because `pytorch_wavelets` uses removed NumPy 2.0 APIs (`np.asfarray`, `np.issubsctype`).

**`Dockerfile.gpu`:**
```dockerfile
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
# Pin specific PyTorch version because pytorch_wavelets is unmaintained
# and uses old-style autograd.Function (compatible through 2.x but untested
# with future versions). numpy<2 required for same reason.
```

**`pytorch_wavelets`** installed from git: `pip install git+https://github.com/fbcotter/pytorch_wavelets.git`

### F. Error Handling

**OOM during processing:**
```python
try:
    Yl, Yh = xfm(batch)
except RuntimeError as e:
    if 'out of memory' in str(e).lower():
        print(f"Error: GPU out of memory during {step_name}.")
        print(f"  Suggestions:")
        print(f"  - Reduce --nlevels (current: {nlevels})")
        print(f"  - Close other GPU applications")
        print(f"  - Use --device to select a different GPU")
        print(f"  - Remove --gpu to use CPU mode")
        sys.exit(1)
    raise
```

**Import errors:** When `--gpu` is passed but PyTorch or `pytorch_wavelets` is not installed, print a clear error with install instructions rather than a traceback.

### G. Testing Strategy

**`tests/test_motion_mag_gpu.py`:**

All GPU tests wrapped in `@pytest.mark.skipif(not torch.cuda.is_available())`.

- **Tier 1 (strict):** Memory estimation arithmetic (pure math, exact equality)
- **Tier 2 (moderate):**
  - GPU DTCWT forward/inverse roundtrip (PSNR > 120 dB)
  - Phase extraction produces finite values, correct shapes
  - Batched processing matches single-batch (verify cross-batch boundary)
  - cuFFT filter output shape and finiteness
- **Tier 3 (smoke):**
  - Full GPU pipeline on `face.mp4` (finite output, correct dimensions, reasonable range)
  - OOM error message format

**CI:** GPU job runs lint only (no GPU on runners). Local testing via `./test.sh gpu`.

## Alternatives Considered

### Single file vs separate `motion_mag_gpu.py`

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Single file with `--gpu` flag | No code duplication, shared CLI/IO code, simpler maintenance | Longer file, GPU imports at call site | **Chosen** — DRY principle, ~200 lines of shared code would drift |
| Separate `motion_mag_gpu.py` | Clean separation, mirrors EVM project | Duplicates load_video, save_video, format_duration, arg parsing, validation | Rejected — EVM has 2 files because CuPy API differs from numpy; here pytorch_wavelets replaces only the DTCWT calls |

### C=3 multi-channel batching vs 3× C=1 sequential

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| C=3 (all channels in one DTCWT call) | 2.2x faster DTCWT, fewer kernel launches | Phases still filtered per-channel (no speedup on 95% of pipeline), more complex reshaping, 2.5x VRAM | Rejected — 1.2x total speedup not worth complexity |
| 3× C=1 sequential | Simple, matches CPU path, lower VRAM | 3 forward+inverse passes | **Chosen** — DTCWT is <5% of GPU pipeline time |

### Float64 cumsum vs float32 everywhere

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Float64 cumsum on CPU | Theoretically more precise | 0.3s overhead, max error only 2.4e-4 rad at 900 frames | Rejected — error is 1000x below visibility |
| Float32 everywhere | Simple, no dtype casting | Cumsum error grows with frames | **Chosen** — verified: 0.0002 rad max error at 900 frames with k=5, negligible |

### Whole-array cuFFT vs chunked cuFFT

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Whole-array cuFFT | Simplest code, single kernel launch | OOMs for >100 frames at 528×592 (cuFFT uses 20× array size) | Rejected — doesn't fit on consumer GPUs |
| Chunked along coefficients | Always fits, auto-tunable | CPU↔GPU transfer per chunk, slightly more code | **Chosen** — 3x faster than CPU FFT even with chunking |

### Wavelet filters: `near_sym_a` (current default) vs `near_sym_b`

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Keep `near_sym_a`/`qshift_a` | Backward compatible | Visible block artifacts at higher magnification | Rejected — quality matters more |
| Switch to `near_sym_b`/`qshift_b` | Fewer artifacts (longer filters, better directional selectivity), matches Visual-Mic | Breaking change for existing users | **Chosen** — v2.0.0 justifies the break, `--biort`/`--qshift` flags allow old behavior |

### Store amplitudes + Yl in Pass 1 vs re-run forward DTCWT in Pass 2

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Store amplitudes + Yl on CPU | One forward pass | +718 MB RAM per channel for amplitudes, +negligible for Yl | Rejected — RAM cost not worth it |
| Re-run forward in Pass 2 | Saves ~718 MB RAM, simpler data flow | Extra forward DTCWT pass (~1s for 301 frames on GPU) | **Chosen** — forward is deterministic (0.00 diff), <1s cost |

## Tradeoffs and Risks

- **`pytorch_wavelets` is unmaintained** (last commit 2022). Uses only stable PyTorch APIs (`F.conv2d`, `autograd.Function`), but `pkg_resources` usage will break on Python 3.14+. Mitigation: pin PyTorch and numpy versions in Dockerfile; if it breaks, fork or rewrite (~400 lines using same `F.conv2d` approach).

- **CPU and GPU outputs differ.** Different DTCWT implementations (dtcwt vs pytorch_wavelets), different precision (float64 vs float32). Both produce valid motion magnification; they are not cross-comparable. Documented, accepted.

- **cuFFT boundary handling differs from CPU.** GPU uses zero-padding, CPU uses reflect-padding for the temporal filter. ~1.3% relative error at video boundaries, 65+ dB PSNR. Visually imperceptible.

- **Default filter change is breaking.** Switching from `near_sym_a` to `near_sym_b` changes output for all users. Justified by visible quality improvement; old behavior restorable via `--biort near_sym_a --qshift qshift_a`.

- **Memory-intensive for large videos.** A 1080p 30s video needs ~5.8 GB CPU RAM (GPU path float32). Pre-flight check catches this before processing starts.

- **numpy<2 pin is a maintenance burden.** Required because `pytorch_wavelets` uses removed APIs. If a future dependency requires numpy 2+, we'll need to fork `pytorch_wavelets` and patch it.

## Open Questions

None — all questions from the PRD were resolved during the grill phase.
