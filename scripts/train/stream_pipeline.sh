#!/usr/bin/env bash
# Stream extractions into prep_manifest + train as each speaker finishes.
#
# Pre-conditions:
#   - data/<sid>/metadata.jsonl exists for every requested speaker
#     (rebuild_many_speakers.py / rebuild_speaker_dataset.py output).
#   - configs/train_v4_small_lora.yaml exists.
#   - models/Irodori-TTS-v4.1-Small/model.safetensors exists.
#
# Each GPU runs a worker that claims one ready speaker at a time, runs
# prepare_manifest.py on it, then hands it to train_multi_speaker.sh
# (which owns step budgeting, LR-schedule scaling, resume, and W&B
# naming). Prep, and training of already-prepped speakers, overlap.
#
# LOCK_DIR defaults to locks/<campaign> inside the repo, so on a shared
# NFS checkout several machines can run this concurrently and split the
# speaker list between them via atomic claim files.
#
# Environment knobs:
#   SPEAKERS   - comma/space-separated speaker ids (or pass as args).
#                Default: every data/*/ dir with a metadata.jsonl but no
#                manifest.jsonl-and-checkpoint yet is NOT assumed; the
#                default is every gi_*/hsr_*/wuwa_* dir with metadata.jsonl.
#   GPUS       - GPU indices for this machine's workers, e.g. "1 4 7".
#   CONFIG     - train config. Default: configs/train_v4_small_lora.yaml
#   BASE_CKPT  - base checkpoint. Default: models/Irodori-TTS-v4.1-Small/model.safetensors
#   LOCK_DIR   - claim dir shared across machines. Default: locks/stream_v4
#   TARGET_EPOCHS etc. pass through to train_multi_speaker.sh.

set -uo pipefail
cd "$(dirname "$0")/../.."

if [ $# -gt 0 ]; then
  SPEAKERS=("$@")
elif [ -n "${SPEAKERS:-}" ]; then
  # shellcheck disable=SC2206
  SPEAKERS=(${SPEAKERS//,/ })
else
  SPEAKERS=()
  for meta in data/gi_*/metadata.jsonl data/hsr_*/metadata.jsonl data/wuwa_*/metadata.jsonl; do
    [ -f "$meta" ] || continue
    SPEAKERS+=("$(basename "$(dirname "$meta")")")
  done
fi
if [ "${#SPEAKERS[@]}" -eq 0 ]; then
  echo "ERROR: no speakers (no args, no SPEAKERS env, no data/{gi,hsr,wuwa}_*/metadata.jsonl)" >&2
  exit 1
fi

GPUS=(${GPUS:-0 3 4 5 6 7})
CONFIG="${CONFIG:-configs/train_v4_small_lora.yaml}"
BASE_CKPT="${BASE_CKPT:-models/Irodori-TTS-v4.1-Small/model.safetensors}"
LOCK_DIR="${LOCK_DIR:-locks/stream_v4}"
mkdir -p "$LOCK_DIR"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

is_extraction_done() {
  local sid="$1"
  local meta="data/${sid}/metadata.jsonl"
  [ -s "$meta" ] || return 1
  pgrep -f "rebuild_(speaker_dataset|many_speakers).*${sid}" >/dev/null 2>&1 && return 1
  return 0
}

claim_speaker() {
  for sid in "${SPEAKERS[@]}"; do
    [ -f "${LOCK_DIR}/${sid}.claimed" ] && continue
    [ -f "${LOCK_DIR}/${sid}.done" ] && continue
    is_extraction_done "$sid" || continue
    # noclobber-based atomic claim (works across NFS clients)
    if (set -C; printf '%s' "$(hostname -s):$$" > "${LOCK_DIR}/${sid}.claimed") 2>/dev/null; then
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

    local prep_log="data/${sid}/preprocess.log"

    if [ ! -f "data/${sid}/manifest.jsonl" ] || [ ! -d "data/${sid}/latents" ] \
        || [ -z "$(ls -A "data/${sid}/latents" 2>/dev/null)" ]; then
      echo "=== prep_manifest $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname -s) gpu=${gpu} ===" >> "$prep_log"
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
        sleep 10
        continue
      fi
    fi

    echo "[gpu=${gpu}][${sid}] train"
    GPUS="$gpu" CONFIG="$CONFIG" BASE_CKPT="$BASE_CKPT" \
      scripts/train/train_multi_speaker.sh "$sid"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "[gpu=${gpu}][${sid}] train FAILED rc=${rc}" >&2
      # marked .done anyway so workers move on instead of retry-looping;
      # to retry later, remove the .claimed and .done files for this sid
    else
      echo "[gpu=${gpu}][${sid}] train DONE"
    fi
    touch "${LOCK_DIR}/${sid}.done"
  done
}

echo "=== stream_pipeline start: host=$(hostname -s) speakers=${#SPEAKERS[@]} gpus=${GPUS[*]} ==="
echo "config:    ${CONFIG}"
echo "base_ckpt: ${BASE_CKPT}"
echo "lock_dir:  ${LOCK_DIR}"
for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
done
wait
echo "=== stream_pipeline all done: host=$(hostname -s) ==="
