FROM python:3.11.15-slim@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff

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
