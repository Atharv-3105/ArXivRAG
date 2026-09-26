# syntax=docker/dockerfile:1

# ---- builder: resolve + install dependencies from the lockfile (reproducible) ----
# Pinned to Debian 12 (OpenSSL 3.0) ON PURPOSE. The unpinned python:3.12-slim moved to Debian 13
# (OpenSSL 3.5), whose default TLS handshake (post-quantum key share) gets HTTP 406 from
# export.arxiv.org. Verified: restricting TLS groups makes the same request succeed.
FROM python:3.12-slim-bookworm AS builder

# Pinned, not :latest (plan: immutable base images)
COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Only the dependency manifests, so this layer is cached until dependencies change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---- runtime: no compilers, no uv, non-root ----
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    MODEL_CACHE_DIR=/models

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /app/data /models \
    && chown -R app:app /app /models /home/app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src ./src

USER app

# Default job: ingest papers. The same image can run the API by overriding the command.
CMD ["python", "-m", "src.pipeline.runner"]
