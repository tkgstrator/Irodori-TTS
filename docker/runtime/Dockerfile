# syntax=docker/dockerfile:1.7
#
# Runtime image for the Irodori-TTS FastAPI server.
#
# The entrypoint installs the project into the container's system Python
# on start — no virtualenv, which the container already makes redundant, and
# nothing for a bind mount to shadow or for a throwaway `uv run` on the same
# host to delete mid-serve. UV_PROJECT_ENVIRONMENT is /usr/local rather than
# /usr because uv requires a bin/python in the target environment and Debian
# ships only python3 there. Mount the uv cache (/root/.cache/uv) for speed.
#
# Everything else is supplied by the user via bind mounts:
#   - /app/config.yaml              (server config YAML)
#   - /app/models/...               (base checkpoint and LoRA adapters)
#
# Build (from repo root):
#   docker build -f docker/runtime/Dockerfile -t ghcr.io/tkgstrator/irodori-tts .
# Run:
#   docker run --gpus all -p 8765:8765 \
#     -v ./configs/runtime.yaml:/app/config.yaml:ro \
#     -v tts_models:/app/models \
#     -v tts_hf_cache:/root/.cache/huggingface \
#     ghcr.io/tkgstrator/irodori-tts
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    TTS_CONFIG=/app/config.yaml \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8765

RUN \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    ffmpeg \
    ca-certificates \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config && \
  ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
  ln -sf /usr/bin/python3.12 /usr/local/bin/python

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY irodori_tts ./irodori_tts
COPY server.py infer.py ./
COPY docker/runtime/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 8765

ENTRYPOINT ["./entrypoint.sh"]
