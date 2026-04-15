# syntax=docker/dockerfile:1.7
#
# Runtime image for the Irodori-TTS FastAPI server.
#
# This image is intentionally slim. At runtime the entrypoint only runs
# `uv sync --frozen --no-dev` into /app/.venv (mount as a named volume so the
# venv persists across restarts) and then launches server.py.
#
# Everything else is supplied by the user via bind mounts:
#   - /app/config.yaml              (server config YAML)
#   - /app/models/...               (base checkpoint referenced from config)
#   - /app/outputs/...              (LoRA adapter directories referenced from config)
#
# Build (from repo root):
#   docker build -f docker/runtime/Dockerfile -t tkgling/irodori-tts .
# Run:
#   docker run --gpus all -p 8765:8765 \
#     -v ./configs/runtime.yaml:/app/config.yaml:ro \
#     -v ./outputs:/app/outputs:ro \
#     -v tts_models:/app/models \
#     -v tts_venv:/app/.venv \
#     -v tts_hf_cache:/root/.cache/huggingface \
#     tkgling/irodori-tts
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
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
    ffmpeg \
    ca-certificates \
    curl \
    git && \
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
