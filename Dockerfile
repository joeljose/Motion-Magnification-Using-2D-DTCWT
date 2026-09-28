FROM python:3.14.6-slim@sha256:7bec7ddcddeff7975d6ba9b4be7dd6f6b2f55e7491539145e2978f7f97ce9144

# Fixed non-root user; for bind mounts run with --user "$(id -u):$(id -g)"
RUN useradd -m -u 1000 app

WORKDIR /app

# Hash-locked dependencies; regenerate requirements.lock as described in
# CONTRIBUTING.md when requirements*.txt change
COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY motion_mag.py .
COPY tests/ tests/
COPY scripts/ scripts/

RUN chown -R app:app /app

ARG VERSION
LABEL version=${VERSION}

USER app

ENTRYPOINT ["python", "-u", "motion_mag.py"]
