#!/usr/bin/env bash
# Irodori-TTS runtime container entrypoint.
#
# Environment variables:
#   TTS_CONFIG  - path to tts_server config YAML (bind-mounted).
#                 Default: /app/config.yaml
#   TTS_HOST    - listen host. Default: 0.0.0.0
#   TTS_PORT    - listen port. Default: 8765
set -euo pipefail

cd /app

log() { printf '[entrypoint] %s\n' "$*"; }

: "${TTS_CONFIG:=/app/config.yaml}"
: "${TTS_HOST:=0.0.0.0}"
: "${TTS_PORT:=8765}"

log "uv sync (venv=/app/.venv)"
uv sync --frozen --no-dev

if [ ! -f "${TTS_CONFIG}" ]; then
  echo "ERROR: TTS_CONFIG not found: ${TTS_CONFIG}" >&2
  echo "Mount your server config at ${TTS_CONFIG} (e.g. -v ./configs/runtime.yaml:${TTS_CONFIG}:ro)" >&2
  exit 1
fi

log "starting tts_server config=${TTS_CONFIG} host=${TTS_HOST} port=${TTS_PORT}"
exec uv run --no-sync python tts_server.py \
  --config "${TTS_CONFIG}" \
  --host "${TTS_HOST}" \
  --port "${TTS_PORT}"
