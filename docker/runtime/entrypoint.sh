#!/usr/bin/env bash
# Irodori-TTS runtime container entrypoint.
#
# Environment variables:
#   TTS_CONFIG  - path to server config YAML (bind-mounted).
#                 Default: /app/config.yaml
#   TTS_HOST    - listen host. Default: 0.0.0.0
#   TTS_PORT    - listen port. Default: 8765
set -euo pipefail

cd /app

log() { printf '[entrypoint] %s\n' "$*"; }

: "${TTS_CONFIG:=/app/config.yaml}"
: "${TTS_HOST:=0.0.0.0}"
: "${TTS_PORT:=8765}"

log "uv sync (env=${UV_PROJECT_ENVIRONMENT:-.venv})"
uv sync --frozen --no-dev

if [ ! -f "${TTS_CONFIG}" ]; then
  echo "ERROR: TTS_CONFIG not found: ${TTS_CONFIG}" >&2
  echo "Mount your server config at ${TTS_CONFIG} (e.g. -v ./configs/runtime.yaml:${TTS_CONFIG}:ro)" >&2
  exit 1
fi

# --- Pre-download model checkpoints if not present locally ----------------
# Reads hf_repo / local path pairs from the config YAML and downloads any
# missing checkpoints before the server starts.  This avoids a long blocking
# download inside the Python process (which has no progress output and holds
# the GPU lock).

download_if_missing() {
  local local_path="$1" hf_repo="$2" hf_filename="$3" label="$4"

  if [ -z "${hf_repo}" ]; then
    return
  fi
  if [ -n "${local_path}" ] && [ -f "${local_path}" ]; then
    log "${label}: found locally at ${local_path}"
    return
  fi
  log "${label}: downloading from ${hf_repo}/${hf_filename}"
  uv run --no-sync python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='${hf_repo}', filename='${hf_filename}')
print(f'  cached at {p}')
"
}

# Read the checkpoint paths through load_config rather than the raw YAML, so
# base_version resolves to a repo here exactly as it does in the server.
eval "$(uv run --no-sync python -c "
import shlex
from pathlib import Path
from irodori_tts.server.config import load_config
c = load_config(Path('${TTS_CONFIG}'))
def q(v):
    return shlex.quote(str(v)) if v else \"''\"
print(f'_BASE_LOCAL={q(c.base_checkpoint)}')
print(f'_BASE_REPO={q(c.base_hf_repo)}')
print(f'_BASE_FILE={q(c.base_hf_filename)}')
print(f'_CAP_LOCAL={q(c.caption_checkpoint)}')
print(f'_CAP_REPO={q(c.caption_hf_repo)}')
print(f'_CAP_FILE={q(c.caption_hf_filename)}')
print(f'_CODEC_REPO={q(c.codec_repo)}')
")"

download_if_missing "${_BASE_LOCAL}" "${_BASE_REPO}" "${_BASE_FILE}" "base model"
download_if_missing "${_CAP_LOCAL}" "${_CAP_REPO}" "${_CAP_FILE}" "caption model"

# Codec model (DACVAE) — downloaded by transformers/huggingface_hub at runtime,
# but we can warm the cache here so the server starts faster.
if [ -n "${_CODEC_REPO}" ]; then
  log "codec: ensuring ${_CODEC_REPO} is cached"
  uv run --no-sync python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='${_CODEC_REPO}')
print(f'  cached at {p}')
"
fi

log "starting server config=${TTS_CONFIG} host=${TTS_HOST} port=${TTS_PORT}"
exec uv run --no-sync python server.py \
  --config "${TTS_CONFIG}" \
  --host "${TTS_HOST}" \
  --port "${TTS_PORT}"
