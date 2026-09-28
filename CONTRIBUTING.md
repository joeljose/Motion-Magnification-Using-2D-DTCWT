# Contributing

Thanks for your interest in contributing!

## How to contribute

1. **Open an issue first** — describe the bug or feature you'd like to work on.
2. **Fork the repo** and create a branch from `main`.
3. **Keep PRs small** — one logical change per pull request.
4. **Follow PEP 8** for Python code style. We use [ruff](https://docs.astral.sh/ruff/) for linting.
5. **Test your changes** before opening a PR:
   ```bash
   ./test.sh
   ```
   Tests run inside Docker — no local Python dependencies needed. See [Development](README.md#development) in the README for details.
6. **Open a pull request** against `main` with a clear description of your changes.

## Dependencies

`requirements*.txt` hold the allowed version ranges. The Docker images install from the hash-locked `requirements.lock` (CPU, Python 3.11) and `requirements-gpu.lock` (GPU, Python 3.10), so builds are reproducible. After changing a `requirements*.txt` file, or to pick up new releases on purpose, regenerate both locks with [uv](https://docs.astral.sh/uv/):

```bash
uv pip compile requirements.txt requirements-dev.txt --python-version 3.11 \
    --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header -o requirements.lock
uv pip compile requirements-gpu.txt requirements-dev.txt --python-version 3.10 \
    --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header -o requirements-gpu.lock
```

`pytorch_wavelets` is pinned to a commit via `PYTORCH_WAVELETS_SHA` in `Dockerfile.gpu`. Base images and GitHub Actions are pinned by digest/SHA; Dependabot proposes updates for them monthly.

## Reporting bugs

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS
