#!/usr/bin/env bash
# Train multiple speaker LoRA adapters in parallel, one per GPU.
#
# Usage:
#   scripts/train/train_multi_speaker.sh [speaker1 speaker2 ...]
#     If no args, reads SPEAKERS from env or uses DEFAULT_SPEAKERS below.
#
# Each speaker must have:
#   - configs/train_500m_v2_<speaker>_lora.yaml
#   - data/<speaker>/manifest.jsonl (+ latents/)
#
# Per-speaker run is pinned to a single GPU via CUDA_VISIBLE_DEVICES.
# stdout/stderr go to outputs/<speaker>_lora/train.log.
# Waits for all runs, then exits non-zero if any failed.

set -uo pipefail

cd "$(dirname "$0")/.."

DEFAULT_SPEAKERS=(
  margo leia coco alisa hanna meruru nanoka miria noah yuki anan
)

if [ $# -gt 0 ]; then
  SPEAKERS=("$@")
elif [ -n "${SPEAKERS:-}" ]; then
  # shellcheck disable=SC2206
  SPEAKERS=(${SPEAKERS})
else
  SPEAKERS=("${DEFAULT_SPEAKERS[@]}")
fi

# GPU detection (uses nvidia-smi). Override with GPUS="0 1 2 3".
if [ -z "${GPUS:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader | awk '{print $1}')
  else
    GPU_LIST=(0)
  fi
else
  # shellcheck disable=SC2206
  GPU_LIST=(${GPUS})
fi

BASE_CKPT="${BASE_CKPT:-models/Irodori-TTS-500M-v2/model.safetensors}"

if [ ! -f "${BASE_CKPT}" ]; then
  echo "ERROR: base checkpoint not found: ${BASE_CKPT}" >&2
  exit 1
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "=== multi-speaker train launch ==="
echo "GPUs: ${GPU_LIST[*]}"
echo "Speakers: ${SPEAKERS[*]}"
echo

fail_count=0

# Partition speakers into per-GPU queues using LPT (Longest Processing Time
# first) greedy balancing on manifest line counts. This keeps total training
# work per GPU roughly equal without needing an LP solver — LPT is a simple
# 4/3-approximation to the NP-hard multiway number partitioning problem.
# Falls back to size 0 for any missing manifest (error surfaces later in
# run_queue). At most one training job per GPU at a time.
n_gpu=${#GPU_LIST[@]}
declare -a QUEUES
declare -a GPU_LOAD
for ((g = 0; g < n_gpu; g++)); do
  QUEUES[$g]=""
  GPU_LOAD[$g]=0
done

declare -A SPEAKER_COUNT
for speaker in "${SPEAKERS[@]}"; do
  manifest="data/${speaker}/manifest.jsonl"
  if [ -f "${manifest}" ]; then
    SPEAKER_COUNT[$speaker]=$(wc -l < "${manifest}" | tr -d ' ')
  else
    echo "[${speaker}] WARN: missing manifest ${manifest}, assuming size 0 for balancing" >&2
    SPEAKER_COUNT[$speaker]=0
  fi
done

# Sort speakers by count desc (LPT)
mapfile -t SORTED_SPEAKERS < <(
  for s in "${SPEAKERS[@]}"; do
    printf '%d\t%s\n' "${SPEAKER_COUNT[$s]}" "$s"
  done | sort -rn -k1,1
)

for entry in "${SORTED_SPEAKERS[@]}"; do
  count="${entry%%$'\t'*}"
  speaker="${entry#*$'\t'}"
  min_g=0
  min_load=${GPU_LOAD[0]}
  for ((g = 1; g < n_gpu; g++)); do
    if [ "${GPU_LOAD[$g]}" -lt "${min_load}" ]; then
      min_load=${GPU_LOAD[$g]}
      min_g=$g
    fi
  done
  QUEUES[$min_g]+="${speaker} "
  GPU_LOAD[$min_g]=$((GPU_LOAD[$min_g] + count))
done

for ((g = 0; g < n_gpu; g++)); do
  gpu="${GPU_LIST[$g]}"
  queue="${QUEUES[$g]}"
  [ -z "${queue}" ] && continue
  echo "[gpu=${gpu}] load=${GPU_LOAD[$g]} queue: ${queue}"
done
echo

find_latest_checkpoint() {
  local outdir="$1"
  local manifest_size="$2"
  local latest=""
  local latest_step=-1
  shopt -s nullglob
  for path in "${outdir}"/checkpoint_[0-9]*; do
    [ -d "${path}" ] || continue
    local name step
    name="$(basename "${path}")"
    step="${name#checkpoint_}"
    step="${step%%[!0-9]*}"
    [ -z "${step}" ] && continue
    # Skip checkpoints whose recorded manifest size does not match the current
    # manifest — training data has changed, so optimizer/scheduler state is
    # stale.
    if [ -n "${manifest_size}" ] && [ -f "${path}/manifest_size.txt" ]; then
      local stored
      stored="$(tr -d '[:space:]' < "${path}/manifest_size.txt")"
      if [ -n "${stored}" ] && [ "${stored}" != "${manifest_size}" ]; then
        continue
      fi
    fi
    if [ "${step}" -gt "${latest_step}" ]; then
      latest_step="${step}"
      latest="${path}"
    fi
  done
  shopt -u nullglob
  printf '%s' "${latest}"
}

run_queue() {
  local gpu="$1"
  shift
  local speakers=("$@")
  local rc_any=0
  for speaker in "${speakers[@]}"; do
    local cfg="configs/train_500m_v2_${speaker}_lora.yaml"
    local manifest="data/${speaker}/manifest.jsonl"
    local outdir="outputs/${speaker}_lora"

    if [ ! -f "${cfg}" ]; then
      echo "[${speaker}] ERROR: missing config ${cfg}" >&2
      rc_any=1
      continue
    fi
    if [ ! -f "${manifest}" ]; then
      echo "[${speaker}] ERROR: missing manifest ${manifest}" >&2
      rc_any=1
      continue
    fi

    mkdir -p "${outdir}"
    local log="${outdir}/train.log"
    echo "=== launch: $(date +'%Y-%m-%d %H:%M') gpu=${gpu} ===" >> "${log}"
    echo "[${speaker}] -> GPU ${gpu}, log=${log}"

    local manifest_size
    manifest_size="$(wc -l < "${manifest}" | tr -d ' ')"
    local resume_path=""
    if [ "${NO_RESUME:-false}" != "true" ]; then
      resume_path="$(find_latest_checkpoint "${outdir}" "${manifest_size}")"
      if [ -z "${resume_path}" ] && compgen -G "${outdir}/checkpoint_[0-9]*" > /dev/null; then
        echo "[${speaker}] resume skipped: no checkpoint matches current manifest size (${manifest_size})"
      fi
    fi

    local init_args=()
    if [ -n "${resume_path}" ]; then
      echo "[${speaker}] resume: ${resume_path}"
      init_args=(--resume "${resume_path}" --init-checkpoint "${BASE_CKPT}")
    else
      init_args=(--init-checkpoint "${BASE_CKPT}")
    fi

    # shellcheck disable=SC2206
    local extra=(${EXTRA_TRAIN_ARGS:-})
    CUDA_VISIBLE_DEVICES="${gpu}" \
    uv run --no-sync python train.py \
      --config "${cfg}" \
      --manifest "${manifest}" \
      --output-dir "${outdir}" \
      --wandb-run-name "${speaker}_lora" \
      "${init_args[@]}" \
      "${extra[@]}" \
      >> "${log}" 2>&1
    local rc=$?
    if [ "${rc}" -ne 0 ]; then
      echo "[${speaker}] FAILED rc=${rc} (gpu=${gpu})" >&2
      rc_any=1
    else
      echo "[${speaker}] DONE (gpu=${gpu})"
    fi
  done
  return $rc_any
}

declare -A PID_GPU
for ((g = 0; g < n_gpu; g++)); do
  gpu="${GPU_LIST[$g]}"
  # shellcheck disable=SC2206
  queue=(${QUEUES[$g]})
  [ "${#queue[@]}" -eq 0 ] && continue
  run_queue "${gpu}" "${queue[@]}" &
  PID_GPU[$!]="${gpu}"
done

echo "=== waiting for ${#PID_GPU[@]} GPU worker(s) ==="
for pid in "${!PID_GPU[@]}"; do
  gpu="${PID_GPU[$pid]}"
  if ! wait "$pid"; then
    fail_count=$((fail_count + 1))
  fi
done

echo
if [ "${fail_count}" -gt 0 ]; then
  echo "=== multi-speaker train: ${fail_count} failure(s) ==="
  exit 1
fi
echo "=== multi-speaker train: all done ==="
