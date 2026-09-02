#!/usr/bin/env bash
# Train multiple v3 speaker LoRA adapters in parallel, one per GPU.
#
# Usage:
#   scripts/train/train_multi_speaker.sh [speaker1 speaker2 ...]
#     If no args, reads SPEAKERS from env or uses DEFAULT_SPEAKERS below.
#
# v3 flow: a single shared config (configs/train_500m_v3/lora/default.yaml)
# is used for every speaker. Per-speaker sample_generation prompts come from
# data/<speaker>/config.yaml:sample_texts (or, if absent, train.py auto-picks
# length-balanced samples from the manifest). No per-speaker yaml is needed.
#
# Each speaker must have:
#   - data/<speaker>/manifest.jsonl
#   - data/<speaker>/latents/
#
# Per-speaker run is pinned to a single GPU via CUDA_VISIBLE_DEVICES.
# stdout/stderr go to outputs/<speaker>_lora/train.log.
# Waits for all runs, then exits non-zero if any failed.
#
# Environment knobs:
#   GPUS                          - explicit GPU list, e.g. "0 3 4 5"
#                                   (overrides auto-detection)
#   FREE_GPU_MEM_THRESHOLD_MIB    - auto-detect threshold; GPUs with
#                                   memory.used >= this are skipped.
#                                   Default: 1000
#   BASE_CKPT                     - base v3 checkpoint path. Default:
#                                   models/Irodori-TTS-500M-v3/model.safetensors
#   NO_RESUME                     - "true" to ignore existing checkpoints
#   EXTRA_TRAIN_ARGS              - extra flags appended verbatim to train.py

set -uo pipefail

cd "$(dirname "$0")/../.."

CONFIG="${CONFIG:-configs/train_500m_v3/lora/default.yaml}"
BASE_CKPT="${BASE_CKPT:-models/Irodori-TTS-500M-v3/model.safetensors}"

DEFAULT_SPEAKERS=(
  ema hiro sherry margo leia coco alisa hanna meruru
  nanoka miria noah yuki anan cherry
)

if [ $# -gt 0 ]; then
  SPEAKERS=("$@")
elif [ -n "${SPEAKERS:-}" ]; then
  # shellcheck disable=SC2206
  SPEAKERS=(${SPEAKERS//,/ })
else
  SPEAKERS=("${DEFAULT_SPEAKERS[@]}")
fi

# GPU selection: GPUS env overrides auto-detection. Otherwise, query nvidia-smi
# and keep GPUs whose memory.used is below FREE_GPU_MEM_THRESHOLD_MIB (default
# 1000). This skips GPUs already running other services so we only land on
# idle ones.
: "${FREE_GPU_MEM_THRESHOLD_MIB:=1000}"
if [ -n "${GPUS:-}" ]; then
  # shellcheck disable=SC2206
  GPU_LIST=(${GPUS})
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_LIST < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', *' -v thr="${FREE_GPU_MEM_THRESHOLD_MIB}" '$2+0 < thr {print $1}'
  )
  if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: no GPU has memory.used < ${FREE_GPU_MEM_THRESHOLD_MIB} MiB." >&2
    echo "       Set GPUS=... to override, or lower FREE_GPU_MEM_THRESHOLD_MIB." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv >&2
    exit 1
  fi
else
  GPU_LIST=(0)
fi

if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi
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

echo "=== multi-speaker v3 LoRA train ==="
echo "config:    ${CONFIG}"
echo "base_ckpt: ${BASE_CKPT}"
echo "GPUs:      ${GPU_LIST[*]}"
echo "speakers:  ${SPEAKERS[*]}"
echo

fail_count=0

# LPT (Longest Processing Time first) greedy partition: keep total manifest
# rows roughly equal across GPUs without an LP solver.
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
    echo "[${speaker}] WARN: missing manifest ${manifest}, assuming size 0" >&2
    SPEAKER_COUNT[$speaker]=0
  fi
done

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
    local manifest="data/${speaker}/manifest.jsonl"
    local outdir="outputs/${speaker}_lora"

    if [ ! -f "${manifest}" ]; then
      echo "[${speaker}] ERROR: missing manifest ${manifest}" >&2
      rc_any=1
      continue
    fi

    mkdir -p "${outdir}"
    local log="${outdir}/train.log"
    echo "=== launch: $(date -u +'%Y-%m-%dT%H:%M:%SZ') gpu=${gpu} ===" >> "${log}"
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

    # Budget steps by dataset size. A fixed step count overtrains the small
    # speakers badly: at 74 clips a run is already 60 epochs deep after 30
    # steps. Scale by epochs instead, clamped so the tiny sets still get a
    # real run and the big ones do not run away. Set STEPS_PER_EPOCH_TARGET=0
    # to opt out and rely on --max-steps from EXTRA_TRAIN_ARGS.
    local epochs="${TARGET_EPOCHS:-40}"
    if [ "${epochs}" != "0" ] && [[ ! " ${extra[*]} " == *" --max-steps "* ]]; then
      local eff_batch="${EFFECTIVE_BATCH:-80}"
      local scaled=$(( manifest_size * epochs / eff_batch ))
      [ "${scaled}" -lt "${MIN_STEPS:-500}" ] && scaled="${MIN_STEPS:-500}"
      [ "${scaled}" -gt "${MAX_STEPS_CAP:-10000}" ] && scaled="${MAX_STEPS_CAP:-10000}"
      echo "[${speaker}] steps=${scaled} (${manifest_size} clips, ~${epochs} epochs at batch ${eff_batch})"
      extra+=(--max-steps "${scaled}")
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" \
    uv run --no-sync python train.py \
      --config "${CONFIG}" \
      --manifest "${manifest}" \
      --output-dir "${outdir}" \
      --wandb-run-name "${speaker}_lora_v4" \
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
