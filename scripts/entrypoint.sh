#!/usr/bin/env bash
# Irodori-TTS training container entrypoint.
#
# Environment variables:
#   HF_TOKEN           - HF access token (required for private datasets / uploads).
#   WANDB_API_KEY      - W&B API key (optional; enables online logging).
#   WANDB_BASE_URL     - Custom W&B server URL (e.g. https://wandb.tkgstrator.work).
#                        Unset for public wandb.ai.
#   CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET
#                      - Cloudflare Access service-token credentials. Required
#                        when WANDB_BASE_URL points at a server behind CF Access;
#                        train.py injects them as CF-Access-* headers on wandb
#                        requests.
#   HF_DATASET         - HF dataset repo ID that holds all speakers as subdirs
#                        (e.g. ultemica/irodori-tts-voices). If unset, skips
#                        download and uses whatever is already mounted under
#                        /app/data.
#   SPEAKERS           - Comma-separated speaker names. When HF_DATASET is set,
#                        limits the download to those subdirs. When unset, the
#                        entire repo is pulled. At training time, SPEAKERS also
#                        filters which speakers to train (default: every
#                        speaker in /app/data that has a manifest.jsonl).
#   BASE_MODEL_REPO    - HF repo for the base checkpoint.
#                        Default: Aratako/Irodori-TTS-500M-v2
#   BASE_CKPT          - Path to base checkpoint inside the container.
#                        Default: /app/models/Irodori-TTS-500M-v2/model.safetensors
#   GPUS               - Space-separated GPU indices for train_multi_speaker.sh.
#                        Takes precedence over NUM_GPUS. Example: "0 1 2 3"
#   NUM_GPUS           - Integer number of GPUs to use. If set (and GPUS unset),
#                        selects indices [0..NUM_GPUS-1]. Default: all visible
#                        GPUs (round-robin).
#
# Hyperparameter overrides (all optional — leave unset to use the template
# configs/train_500m_v2_speaker_lora.yaml defaults). Each one, if set, is
# passed to train.py as a CLI flag and therefore overrides the config value.
#
#   MAX_EPOCHS                    -> --max-epochs
#   BATCH_SIZE                    -> --batch-size
#   GRADIENT_ACCUMULATION_STEPS   -> --gradient-accumulation-steps
#   LEARNING_RATE                 -> --lr
#   WEIGHT_DECAY                  -> --weight-decay
#   WARMUP_RATIO                  -> --warmup-ratio
#   DECAY_RATIO                   -> --decay-ratio
#   MIN_LR_SCALE                  -> --min-lr-scale
#   LORA_R                        -> --lora-r
#   LORA_ALPHA                    -> --lora-alpha
#   LORA_DROPOUT                  -> --lora-dropout
#   TEXT_CONDITION_DROPOUT        -> --text-condition-dropout
#   SPEAKER_CONDITION_DROPOUT     -> --speaker-condition-dropout
#   VALID_EVERY                   -> --valid-every
#   SAVE_EVERY                    -> --save-every
#   CHECKPOINT_BEST_N             -> --checkpoint-best-n
#   SEED                          -> --seed
#   EXTRA_TRAIN_ARGS              -> arbitrary extra flags appended as-is
#                                    (escape hatch for anything not listed)
set -euo pipefail

cd /app

log() { printf '[entrypoint] %s\n' "$*"; }

: "${BASE_MODEL_REPO:=Aratako/Irodori-TTS-500M-v2}"
: "${BASE_CKPT:=/app/models/Irodori-TTS-500M-v2/model.safetensors}"

# -----------------------------------------------------------------------------
# 0. Ensure Python venv is in place. The image ships only system deps and
#    project sources; .venv lives on a named volume so it persists across
#    runs.
# -----------------------------------------------------------------------------
log "uv sync (venv=/app/.venv)"
uv sync --frozen --no-dev

# -----------------------------------------------------------------------------
# 1. Ensure base checkpoint is present.
# -----------------------------------------------------------------------------
if [ ! -f "${BASE_CKPT}" ]; then
  log "base checkpoint missing -> downloading ${BASE_MODEL_REPO}"
  mkdir -p "$(dirname "${BASE_CKPT}")"
  uv run --no-sync python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["BASE_MODEL_REPO"],
    local_dir=os.path.dirname(os.environ["BASE_CKPT"]),
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN"),
    allow_patterns=["model.safetensors", "*.json", "*.md"],
)
PY
fi
if [ ! -f "${BASE_CKPT}" ]; then
  echo "ERROR: base checkpoint still not found at ${BASE_CKPT}" >&2
  exit 1
fi
log "base checkpoint: ${BASE_CKPT}"

# -----------------------------------------------------------------------------
# 2. Download dataset from HF, if requested.
# -----------------------------------------------------------------------------
if [ -n "${HF_DATASET:-}" ]; then
  if [ -n "${SPEAKERS:-}" ]; then
    log "download: ${HF_DATASET} (speakers=${SPEAKERS})"
    uv run --no-sync python scripts/hf_download_dataset.py \
      --repo-id "${HF_DATASET}" --speakers "${SPEAKERS}"
  else
    log "download: ${HF_DATASET} (all speakers)"
    uv run --no-sync python scripts/hf_download_dataset.py --repo-id "${HF_DATASET}"
  fi
fi

# -----------------------------------------------------------------------------
# 3. Discover speakers to train.
# -----------------------------------------------------------------------------
declare -a TRAIN_SPEAKERS=()

if [ -n "${SPEAKERS:-}" ]; then
  IFS=',' read -ra REQUESTED <<< "${SPEAKERS}"
  for s in "${REQUESTED[@]}"; do
    s="${s// /}"
    [ -z "${s}" ] && continue
    TRAIN_SPEAKERS+=("${s}")
  done
else
  # Auto-discover: any data/<speaker>/ with a manifest.jsonl.
  shopt -s nullglob
  for manifest in data/*/manifest.jsonl; do
    dir="$(dirname "${manifest}")"
    TRAIN_SPEAKERS+=("$(basename "${dir}")")
  done
  shopt -u nullglob
fi

if [ "${#TRAIN_SPEAKERS[@]}" -eq 0 ]; then
  echo "ERROR: no speakers to train. Set HF_REPOS / SPEAKERS, or mount data/ with manifest.jsonl." >&2
  exit 1
fi

log "training speakers: ${TRAIN_SPEAKERS[*]}"

# -----------------------------------------------------------------------------
# 4. Validate per-speaker data layout and generate config if missing.
# -----------------------------------------------------------------------------
for s in "${TRAIN_SPEAKERS[@]}"; do
  manifest="data/${s}/manifest.jsonl"
  latents="data/${s}/latents"
  if [ ! -f "${manifest}" ]; then
    echo "ERROR: ${manifest} not found" >&2
    exit 1
  fi
  if [ ! -d "${latents}" ]; then
    echo "ERROR: ${latents}/ not found" >&2
    exit 1
  fi

  cfg="configs/train_500m_v2_${s}_lora.yaml"
  if [ ! -f "${cfg}" ]; then
    log "generating ${cfg}"
    uv run --no-sync python scripts/make_speaker_config.py "${s}"
  fi
done

# -----------------------------------------------------------------------------
# 5. W&B login (optional).
# -----------------------------------------------------------------------------
if [ -n "${WANDB_API_KEY:-}" ]; then
  log "W&B key detected — runs will log online"
  if [ -n "${WANDB_BASE_URL:-}" ]; then
    log "W&B server: ${WANDB_BASE_URL}"
    if [ -n "${CF_ACCESS_CLIENT_ID:-}" ] && [ -n "${CF_ACCESS_CLIENT_SECRET:-}" ]; then
      log "CF Access service token detected — will be forwarded as CF-Access-* headers"
    else
      log "WARNING: WANDB_BASE_URL set but CF_ACCESS_CLIENT_ID/SECRET missing — requests may be blocked by Cloudflare Access"
    fi
  fi
else
  log "no WANDB_API_KEY — W&B runs may fall back to offline mode"
fi

# -----------------------------------------------------------------------------
# 6. GPU selection.
# -----------------------------------------------------------------------------
if [ -z "${GPUS:-}" ] && [ -n "${NUM_GPUS:-}" ]; then
  n="${NUM_GPUS}"
  if ! [[ "${n}" =~ ^[0-9]+$ ]] || [ "${n}" -lt 1 ]; then
    echo "ERROR: NUM_GPUS must be a positive integer, got '${n}'" >&2
    exit 1
  fi
  GPUS=""
  for ((i = 0; i < n; i++)); do
    GPUS="${GPUS}${GPUS:+ }${i}"
  done
  export GPUS
  log "NUM_GPUS=${n} -> GPUS='${GPUS}'"
elif [ -n "${GPUS:-}" ]; then
  log "GPUS='${GPUS}'"
else
  log "GPUS not set -> using all visible GPUs"
fi

# -----------------------------------------------------------------------------
# 7. Build EXTRA_TRAIN_ARGS from env overrides.
# -----------------------------------------------------------------------------
extra_args=()
append_flag() {
  local var_name="$1"
  local flag="$2"
  local val="${!var_name:-}"
  if [ -n "${val}" ]; then
    extra_args+=("${flag}" "${val}")
  fi
}
append_flag MAX_EPOCHS                   --max-epochs
append_flag BATCH_SIZE                   --batch-size
append_flag GRADIENT_ACCUMULATION_STEPS  --gradient-accumulation-steps
append_flag LEARNING_RATE                --lr
append_flag WEIGHT_DECAY                 --weight-decay
append_flag WARMUP_RATIO                 --warmup-ratio
append_flag DECAY_RATIO                  --decay-ratio
append_flag MIN_LR_SCALE                 --min-lr-scale
append_flag LORA_R                       --lora-r
append_flag LORA_ALPHA                   --lora-alpha
append_flag LORA_DROPOUT                 --lora-dropout
append_flag TEXT_CONDITION_DROPOUT       --text-condition-dropout
append_flag SPEAKER_CONDITION_DROPOUT    --speaker-condition-dropout
append_flag VALID_EVERY                  --valid-every
append_flag SAVE_EVERY                   --save-every
append_flag CHECKPOINT_BEST_N            --checkpoint-best-n
append_flag SEED                         --seed

# Append arbitrary escape-hatch flags verbatim.
if [ -n "${EXTRA_TRAIN_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  extra_args+=(${EXTRA_TRAIN_ARGS})
fi

if [ "${#extra_args[@]}" -gt 0 ]; then
  log "env overrides -> ${extra_args[*]}"
  export EXTRA_TRAIN_ARGS="${extra_args[*]}"
else
  export EXTRA_TRAIN_ARGS=""
fi

# -----------------------------------------------------------------------------
# 8. Launch training.
# -----------------------------------------------------------------------------
export BASE_CKPT
exec scripts/train_multi_speaker.sh "${TRAIN_SPEAKERS[@]}"
