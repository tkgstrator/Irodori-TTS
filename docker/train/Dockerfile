# syntax=docker/dockerfile:1.7
# Training image for Irodori-TTS LoRA fine-tuning.
#
# This image only ships system dependencies and the project source. The
# Python virtual environment (.venv) and the base model checkpoint are
# created at first container start by scripts/entrypoint.sh — keep
# .venv on a named volume so subsequent runs reuse it.
#
# Build (from repo root):
#   docker build -f docker/train/Dockerfile -t irodori-tts-train .
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    BASE_MODEL_REPO=Aratako/Irodori-TTS-500M-v2 \
    BASE_CKPT=/app/models/Irodori-TTS-500M-v2/model.safetensors

RUN \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  --mount=type=cache,target=/var/cache/apt,sharing=locked \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    ffmpeg git ca-certificates curl \
    build-essential cmake pkg-config && \
  ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
  ln -sf /usr/bin/python3.12 /usr/local/bin/python

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY irodori_tts ./irodori_tts
COPY train.py prepare_manifest.py convert_checkpoint_to_safetensors.py ./
COPY configs ./configs
COPY scripts ./scripts
COPY docker/train/entrypoint.sh ./entrypoint.sh

RUN chmod +x entrypoint.sh scripts/train/train_multi_speaker.sh scripts/preprocess/batch_preprocess_voice.sh scripts/ops/backup_wandb_volume.sh

ENTRYPOINT ["./entrypoint.sh"]
