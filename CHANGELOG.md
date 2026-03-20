# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
