#!/usr/bin/env bash
# Stream extractions into prep_manifest + train as each speaker finishes.
#
# Pre-conditions:
#   - rebuild_speaker_dataset.py invocations are running (or have completed)
#     for SPEAKERS, writing data/<sid>/metadata.jsonl when done.
#   - configs/train_500m_v3/lora/default.yaml exists.
#   - models/Irodori-TTS-500M-v3/model.safetensors exists.
#
# Each GPU runs a worker that claims one ready speaker at a time, runs
# prepare_manifest.py + train.py on it, then moves to the next speaker.
# This way extraction, prep, and train overlap; trainings start as soon
# as their inputs are ready instead of waiting for the whole batch.

set -uo pipefail
cd "$(dirname "$0")/../.."

DEFAULT_SPEAKERS=(mualani yae_miko lumine mavuika hu_tao ningguang raiden ayaka beidou jean sayu eula ganyu)

if [ $# -gt 0 ]; then
  SPEAKERS=("$@")
elif [ -n "${SPEAKERS:-}" ]; then
  # shellcheck disable=SC2206
  SPEAKERS=(${SPEAKERS//,/ })
else
  SPEAKERS=("${DEFAULT_SPEAKERS[@]}")
fi

GPUS=(${GPUS:-0 3 4 5 6 7})
CONFIG="${CONFIG:-configs/train_500m_v3/lora/default.yaml}"
BASE_CKPT="${BASE_CKPT:-models/Irodori-TTS-500M-v3/model.safetensors}"
LOCK_DIR="${LOCK_DIR:-/tmp/genshin_pipeline}"
mkdir -p "$LOCK_DIR"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

is_extraction_done() {
  local sid="$1"
  local meta="data/${sid}/metadata.jsonl"
  [ -f "$meta" ] || return 1
  # Still running if any rebuild process is targeting this speaker's output dir.
  pgrep -f "rebuild_speaker_dataset.*--output-dir data/${sid}\b" >/dev/null 2>&1 && return 1
  # Sanity: non-empty manifest.
  [ -s "$meta" ] || return 1
  return 0
}

claim_speaker() {
  for sid in "${SPEAKERS[@]}"; do
    [ -f "${LOCK_DIR}/${sid}.claimed" ] && continue
    [ -f "${LOCK_DIR}/${sid}.done" ] && continue
    is_extraction_done "$sid" || continue
    # noclobber-based atomic claim
    if (set -C; printf '%s' "$$" > "${LOCK_DIR}/${sid}.claimed") 2>/dev/null; then
      printf '%s' "$sid"
      return 0
    fi
  done
  return 1
}

all_speakers_done() {
  local n_done=0
  for s in "${SPEAKERS[@]}"; do
    [ -f "${LOCK_DIR}/${s}.done" ] && n_done=$((n_done + 1))
  done
  [ "$n_done" -eq "${#SPEAKERS[@]}" ]
}

find_latest_checkpoint() {
  local outdir="$1"
  local latest=""
  local latest_step=-1
  shopt -s nullglob
  for path in "${outdir}"/checkpoint_[0-9]*; do
    [ -d "${path}" ] || continue
    local step
    step="$(basename "${path}")"
    step="${step#checkpoint_}"
    step="${step%%[!0-9]*}"
    [ -z "${step}" ] && continue
    if [ "${step}" -gt "${latest_step}" ]; then
      latest_step="${step}"
      latest="${path}"
    fi
  done
  shopt -u nullglob
  printf '%s' "${latest}"
}

worker() {
  local gpu="$1"
  while true; do
    if all_speakers_done; then
      echo "[gpu=${gpu}] all speakers done; exit"
      return 0
    fi
    local sid
    sid="$(claim_speaker)"
    if [ -z "$sid" ]; then
      sleep 30
      continue
    fi
    echo "[gpu=${gpu}][${sid}] claimed"

    local out="outputs/${sid}_lora"
    local prep_log="data/${sid}/preprocess.log"
    mkdir -p "$out"

    # prep_manifest if latents not yet built.
    if [ ! -f "data/${sid}/manifest.jsonl" ] || [ ! -d "data/${sid}/latents" ] \
        || [ -z "$(ls -A "data/${sid}/latents" 2>/dev/null)" ]; then
      echo "=== prep_manifest $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=${gpu} ===" >> "$prep_log"
      echo "[gpu=${gpu}][${sid}] prep_manifest"
      CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python prepare_manifest.py \
        --dataset json \
        --data-files "train=data/${sid}/metadata.jsonl" \
        --split train \
        --audio-column audio --text-column text \
        --target-sample-rate 44100 \
        --output-manifest "data/${sid}/manifest.jsonl" \
        --latent-dir "data/${sid}/latents" \
        --device cuda \
        >> "$prep_log" 2>&1
      local rc=$?
      if [ "$rc" -ne 0 ]; then
        echo "[gpu=${gpu}][${sid}] prep_manifest FAILED rc=${rc}" >&2
        rm -f "${LOCK_DIR}/${sid}.claimed"
        continue
      fi
    fi

    # train
    local train_log="${out}/train.log"
    echo "=== train $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=${gpu} ===" >> "$train_log"
    local resume_path=""
    if [ "${NO_RESUME:-false}" != "true" ]; then
      resume_path="$(find_latest_checkpoint "$out")"
    fi
    local resume_args=()
    if [ -n "$resume_path" ]; then
      echo "[gpu=${gpu}][${sid}] resume from ${resume_path}"
      resume_args=(--resume "$resume_path" --init-checkpoint "$BASE_CKPT")
    else
      resume_args=(--init-checkpoint "$BASE_CKPT")
    fi
    echo "[gpu=${gpu}][${sid}] train"
    CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python train.py \
      --config "$CONFIG" \
      --manifest "data/${sid}/manifest.jsonl" \
      --output-dir "$out" \
      --wandb-run-name "${sid}_lora_v3" \
      "${resume_args[@]}" \
      >> "$train_log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "[gpu=${gpu}][${sid}] train FAILED rc=${rc}" >&2
    else
      echo "[gpu=${gpu}][${sid}] train DONE"
    fi
    touch "${LOCK_DIR}/${sid}.done"
  done
}

echo "=== stream_pipeline start: speakers=${#SPEAKERS[@]} gpus=${GPUS[*]} ==="
for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
done
wait
echo "=== stream_pipeline all done ==="
