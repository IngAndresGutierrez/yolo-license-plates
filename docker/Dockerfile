# syntax=docker/dockerfile:1
# Requires BuildKit (docker buildx) for the RUN --mount cache mounts.

ARG PYTHON_VERSION=3.13
ARG POETRY_VERSION=2.5.1

# --- Build stage: resolve dependencies from poetry.lock into /app/.venv ---
FROM python:${PYTHON_VERSION}-slim AS build
ARG POETRY_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "poetry==${POETRY_VERSION}"

# Dependencies first so source edits don't invalidate this (very large) layer
COPY pyproject.toml poetry.lock README.md ./
# The cache mount keeps downloaded wheels (torch included) between builds
# without storing them in the image
RUN --mount=type=cache,target=/tmp/poetry-cache \
    poetry install --only main --no-root

COPY src ./src
RUN poetry install --only-root

# --- Runtime stage ---
FROM python:${PYTHON_VERSION}-slim AS runtime

# opencv-python (pulled in by ultralytics) needs libGL and glib at runtime.
# docker-clean is removed so apt keeps its downloads in the cache mount.
RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0

RUN groupadd --system --gid 1001 appgroup \
    && useradd --system --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# The venv is not relocatable, so it keeps the same path as in the build stage
COPY --from=build --chown=1001:1001 /app/.venv /app/.venv
COPY --chown=1001:1001 src ./src
RUN mkdir -p /tmp/ultralytics /tmp/matplotlib \
    && chown 1001:1001 /app /tmp/ultralytics /tmp/matplotlib

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

LABEL org.opencontainers.image.title="yolo-license-plates-trainer"

USER appuser

ENTRYPOINT ["python", "-m", "yolo_license_plates.training.train"]
