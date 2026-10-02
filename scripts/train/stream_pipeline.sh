#!/usr/bin/env bash
# Stream extractions into prep_manifest + train as each speaker finishes.
# Run with --cluster to deploy these two scripts and start Large training on
# g20/g21/g25 using the existing Dockerfile; --deploy only updates the scripts.
#
# Pre-conditions:
#   - data/<sid>/metadata.jsonl exists for every requested speaker
#     (rebuild_many_speakers.py / rebuild_speaker_dataset.py output).
#   - configs/train_v4_small_lora.yaml exists.
#   - models/Irodori-TTS-v4.1-Small/model.safetensors exists.
#
# Each GPU runs a worker that claims one ready speaker at a time, runs
# prepare_manifest.py on it, then hands it to train_multi_speaker.sh
# (which owns step budgeting, LR-schedule scaling, resume, and Atmos
# naming). Prep, and training of already-prepped speakers, overlap.
#
# LOCK_DIR defaults to a generation-specific directory inside the repo, so on a shared
# NFS checkout several machines can run this concurrently and split the
# speaker list between them via atomic claim files.
#
# Environment knobs:
#   SPEAKERS   - comma/space-separated speaker ids (or pass as args).
#                Default: every speaker directory under DATA_ROOT that has a
#                metadata.jsonl or a manifest.jsonl.
#   DATA_ROOT  - where the speaker directories live. Default: data
#   GPUS       - GPU indices for this machine's workers, e.g. "1 4 7", or
#                "auto" / unset for every visible GPU. A worker only starts a
#                speaker on a GPU that is free (see IDLE_GPU_MEM_MIB) and keeps
#                polling every GPU_POLL_SECONDS otherwise, so a GPU that frees
#                up later is picked up. Set WAIT_FOR_FREE_GPU=0 to skip the check.
#   CONFIG     - train config. Default: configs/train_v4_small_lora.yaml
#   BASE_CKPT  - base checkpoint. Default: models/Irodori-TTS-v4.1-Small/model.safetensors
#   LOCK_DIR   - claim dir shared across machines. Default: locks/stream_v4
#                (locks/stream_v4_large for a v4_large config)
#   TARGET_EPOCHS etc. pass through to train_multi_speaker.sh.
#
# A failed speaker gets <sid>.failed, never <sid>.done, and is not retried
# until you remove its .claimed and .failed files.

set -uo pipefail
cd "$(dirname "$0")/../.."

# Copy the two scripts to the shared source. Each file is written under a new
# name and moved into place, so shells that are running right now keep reading
# the old copy while the next speaker to start gets the new one. The old files
# are kept under outputs/script-backups.
deploy_scripts() {
  local node="$1" root="$2" audio="${3:-0}" command quoted
  local files=(scripts/train/stream_pipeline.sh scripts/train/train_multi_speaker.sh)
  if [ "$audio" = 1 ]; then
    files+=(irodori_tts/inference_runtime.py irodori_tts/server/config.py
      irodori_tts/server/registry.py pyproject.toml uv.lock requirements.txt)
  fi
  read -r -d '' command <<'DEPLOY'
set -e
backup=/app/outputs/script-backups/$(date -u +%Y%m%dT%H%M%S)-$$
mkdir -p "$backup" /tmp/new
cat > /tmp/deploy.tar
tar --no-same-owner -xf /tmp/deploy.tar -C /tmp/new
while IFS= read -r file; do
  mkdir -p "$backup/$(dirname "$file")"
  if test -f "/app/$file"; then cp -p "/app/$file" "$backup/$file"; fi
  cp -p "/tmp/new/$file" "/app/$file.new"
  if test -f "/app/$file"; then
    chmod --reference="/app/$file" "/app/$file.new"
    chown --reference="/app/$file" "/app/$file.new"
  fi
  mv -f "/app/$file.new" "/app/$file"
done < <(tar -tf /tmp/deploy.tar)
if [ "$DEPLOY_AUDIO" = 1 ]; then
  for file in configs/train_v4_large_lora.yaml configs/train_v4_small_lora.yaml; do
    if test -f "/app/$file"; then
      mkdir -p "$backup/configs"
      cp -p "/app/$file" "$backup/$file"
      sed -i 's/^  codec_device: cpu$/  codec_device: cuda/' "/app/$file"
    fi
  done
  if test -f /app/irodori_tts/watermark.py; then
    mkdir -p "$backup/irodori_tts"
    cp -p /app/irodori_tts/watermark.py "$backup/irodori_tts/watermark.py"
    rm /app/irodori_tts/watermark.py
  fi
fi
DEPLOY
  printf -v quoted '%q' "$command"
  tar -cf - "${files[@]}" | ssh -o BatchMode=yes "$node" \
    "docker run --rm -i --entrypoint bash -e DEPLOY_AUDIO=$audio -v $root:/app irodori-tts-train:v4.1 -lc $quoted"
}

# --deploy [NODE]: update the scripts only. Safe while training is running.
if [ "${1:-}" = --deploy ] || [ "${1:-}" = --deploy-audio ]; then
  node="${2:-g20}"
  root="${TRAIN_SOURCE_ROOT:-/home/smorimoto/Developer/Irodori-TTS-v4-large}"
  [[ "$node" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ && "$root" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 2
  audio=0
  if [ "$1" = --deploy-audio ]; then audio=1; fi
  deploy_scripts "$node" "$root" "$audio" || exit 1
  echo 'Scripts updated. Running speakers keep the old copy; speakers that start next use the new one.'
  exit 0
fi

# Deploy these two scripts and start detached training containers on each node.
# Credentials are read from the finished Large container and handed to the new
# containers through the SSH input stream: never printed, never written to disk.
if [ "${1:-}" = --cluster ]; then
  shift
  nodes=("$@")
  [ "${#nodes[@]}" -gt 0 ] || nodes=(g20 g21 g25)
  source_root="${TRAIN_SOURCE_ROOT:-/home/smorimoto/Developer/Irodori-TTS-v4-large}"
  legacy_root="${TRAIN_LEGACY_ROOT:-/home/smorimoto/Developer/Irodori-TTS-v4/data}"
  env_node="${TRAIN_ENV_NODE:-g20}"
  env_container="${TRAIN_ENV_CONTAINER:-irodori-v4-large-cherry}"
  [[ "$source_root" =~ ^/[a-zA-Z0-9_./-]+$ && "$legacy_root" =~ ^/[a-zA-Z0-9_./-]+$ ]] || exit 2
  for node in "${nodes[@]}"; do
    [[ "$node" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
    running="$(ssh -o BatchMode=yes "$node" "docker ps --filter name=irodori-large-train-$node --format '{{.Names}}'")" || exit 1
    if [ -n "$running" ]; then
      echo "Training already running on $node; no files were changed."
      exit 0
    fi
  done
  secrets="$(ssh -o BatchMode=yes "$env_node" "docker inspect $env_container --format '{{range .Config.Env}}{{println .}}{{end}}'" | grep -E '^(ATMOS_[A-Z_]+|HF_TOKEN)=')" || secrets=""
  if ! grep -q '^ATMOS_TOKEN=' <<< "$secrets"; then
    echo "No Atmos credentials found in $env_container on $env_node; nothing was started." >&2
    exit 1
  fi
  exports=""
  while IFS= read -r line; do
    exports+="$(printf 'export %s=%q' "${line%%=*}" "${line#*=}")"$'\n'
  done <<< "$secrets"
  unset secrets
  # The shared source is visible on every node. Stage it through the existing
  # setup node, so a newly added node does not need an image before its build.
  deploy_scripts "$env_node" "$source_root" || exit 1
  read -r -d '' remote_script <<'REMOTE'
set -euo pipefail
source_root=$1
legacy_root=$2
node=$3
cd "$source_root"
test -s models/Irodori-TTS-v4-Large/model.safetensors
test -s configs/train_v4_large_lora.yaml
# Reuse the existing Dockerfile; build only where the image is missing.
if ! docker image inspect irodori-tts-train:v4-large >/dev/null 2>&1; then
  docker build -f docker/train/Dockerfile -t irodori-tts-train:v4-large .
fi
env_args=()
for name in ATMOS_TOKEN ATMOS_API_URL ATMOS_BASE_URL ATMOS_VISIBILITY HF_TOKEN; do
  if [ -n "${!name:-}" ]; then env_args+=(-e "$name"); fi
done
docker run -d --name "irodori-large-train-$node" --gpus all --shm-size=16g --init \
  --entrypoint bash \
  -v "$source_root:/app" -v "$legacy_root:/app/data:ro" \
  -v irodori_v4_large_hf:/root/.cache/huggingface -v irodori_v4_large_uv:/root/.cache/uv \
  "${env_args[@]}" \
  -e CONFIG=configs/train_v4_large_lora.yaml \
  -e BASE_CKPT=models/Irodori-TTS-v4-Large/model.safetensors \
  -e OUTPUT_ROOT=outputs_v4_large -e LOCK_DIR=locks/stream_v4_large \
  -e METRICS_BACKEND=atmos -e GPUS=auto -e TARGET_EPOCHS=0 \
  irodori-tts-train:v4-large -lc 'set -e
    uv sync --quiet --frozen --no-dev --extra atmos
    python -c "import yaml; from irodori_tts.config import ModelConfig; from irodori_tts.training.model_init import build_text_tokenizer; from huggingface_hub import hf_hub_download; m=ModelConfig(**yaml.safe_load(open(\"configs/train_v4_large_lora.yaml\"))[\"model\"]); build_text_tokenizer(m, local_files_only=True); hf_hub_download(\"Aratako/Semantic-DACVAE-Japanese-32dim\", \"weights.pth\", local_files_only=True); print(\"Model cache verified\")"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    exec scripts/train/stream_pipeline.sh' >/dev/null
echo "started irodori-large-train-$node"
REMOTE
  pids=()
  for node in "${nodes[@]}"; do
    { printf '%s' "$exports"; printf '%s\n' "$remote_script"; } | \
      ssh -o BatchMode=yes "$node" bash -s -- "$source_root" "$legacy_root" "$node" &
    pids+=("$!")
  done
  unset exports
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [ "$failed" -ne 0 ]; then echo 'Some nodes failed to start.' >&2; exit 1; fi
  echo 'Training started in detached containers; idle GPUs are assigned automatically.'
  exit 0
fi

DATA_ROOT="${DATA_ROOT:-data}"

if [ $# -gt 0 ]; then
  SPEAKERS=("$@")
elif [ -n "${SPEAKERS:-}" ]; then
  # shellcheck disable=SC2206
  SPEAKERS=(${SPEAKERS//,/ })
else
  SPEAKERS=()
  declare -A discovered
  for meta in "${DATA_ROOT}"/*/metadata.jsonl "${DATA_ROOT}"/*/manifest.jsonl; do
    [ -f "$meta" ] || continue
    sid="$(basename "$(dirname "$meta")")"
    if [ -z "${discovered[$sid]:-}" ]; then
      SPEAKERS+=("$sid")
      discovered[$sid]=1
    fi
  done
fi
if [ "${#SPEAKERS[@]}" -eq 0 ]; then
  echo "ERROR: no speakers (no args, no SPEAKERS env, no ${DATA_ROOT}/*/{metadata,manifest}.jsonl)" >&2
  exit 1
fi

if [ -z "${GPUS:-}" ] || [ "$GPUS" = auto ]; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
else
  GPUS=(${GPUS//,/ })
fi
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo 'ERROR: no visible GPUs; nothing was started.' >&2
  exit 1
fi
CONFIG="${CONFIG:-configs/train_v4_small_lora.yaml}"
BASE_CKPT="${BASE_CKPT:-models/Irodori-TTS-v4.1-Small/model.safetensors}"
if [[ "$CONFIG" == *v4_large* ]]; then
  LOCK_DIR="${LOCK_DIR:-locks/stream_v4_large}"
else
  LOCK_DIR="${LOCK_DIR:-locks/stream_v4}"
fi
mkdir -p "$LOCK_DIR"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
if [[ "$CONFIG" == *v4_large* ]]; then
  METRICS_BACKEND="${METRICS_BACKEND:-atmos}"
  if [ "$METRICS_BACKEND" != atmos ] || [ -z "${ATMOS_TOKEN:-}" ]; then
    echo 'ERROR: V4 Large requires working Atmos credentials; no speakers were claimed.' >&2
    exit 1
  fi
  export METRICS_BACKEND
fi

is_extraction_done() {
  local sid="$1"
  local meta="${DATA_ROOT}/${sid}/metadata.jsonl"
  local manifest="${DATA_ROOT}/${sid}/manifest.jsonl"
  [ -s "$manifest" ] && [ -d "${DATA_ROOT}/${sid}/latents" ] && return 0
  [ -s "$meta" ] || return 1
  pgrep -f "rebuild_(speaker_dataset|many_speakers).*${sid}" >/dev/null 2>&1 && return 1
  return 0
}

claim_speaker() {
  for sid in "${SPEAKERS[@]}"; do
    [ -f "${LOCK_DIR}/${sid}.claimed" ] && continue
    [ -f "${LOCK_DIR}/${sid}.done" ] && continue
    [ -f "${LOCK_DIR}/${sid}.failed" ] && continue
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
    if [ -f "${LOCK_DIR}/${s}.done" ] || [ -f "${LOCK_DIR}/${s}.failed" ]; then
      n_done=$((n_done + 1))
    fi
  done
  [ "$n_done" -eq "${#SPEAKERS[@]}" ]
}

# A GPU is free when it holds little memory and no process has a compute context
# on it. Utilization alone is not enough: an idle-looking GPU may belong to
# someone else's job.
# Idle A100s show a few MiB; another user's process holds hundreds.
IDLE_GPU_MEM_MIB="${IDLE_GPU_MEM_MIB:-100}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
gpu_is_free() {
  local gpu="$1" uuid used
  IFS=', ' read -r uuid used < <(
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits \
      | awk -F', *' -v g="$gpu" '$1 == g {print $2, $3}'
  )
  [ -n "${uuid:-}" ] || return 1
  [ "${used:-999999}" -lt "$IDLE_GPU_MEM_MIB" ] || return 1
  local processes
  processes="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)" || return 1
  printf '%s\n' "$processes" | grep -Fq "$uuid" && return 1
  return 0
}

worker() {
  local gpu="$1"
  while true; do
    if all_speakers_done; then
      echo "[gpu=${gpu}] all speakers done; exit"
      return 0
    fi
    # Only take on a speaker when this GPU is free; otherwise keep polling so a
    # GPU that frees up later is picked up automatically.
    if [ "${WAIT_FOR_FREE_GPU:-1}" = "1" ] && ! gpu_is_free "$gpu"; then
      sleep "$GPU_POLL_SECONDS"
      continue
    fi
    local sid
    sid="$(claim_speaker)"
    if [ -z "$sid" ]; then
      sleep "$GPU_POLL_SECONDS"
      continue
    fi
    echo "[gpu=${gpu}][${sid}] claimed"

    local prep_log="${DATA_ROOT}/${sid}/preprocess.log"

    if [ ! -f "${DATA_ROOT}/${sid}/manifest.jsonl" ] || [ ! -d "${DATA_ROOT}/${sid}/latents" ] \
        || [ -z "$(ls -A "${DATA_ROOT}/${sid}/latents" 2>/dev/null)" ]; then
      echo "=== prep_manifest $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname -s) gpu=${gpu} ===" >> "$prep_log"
      echo "[gpu=${gpu}][${sid}] prep_manifest"
      CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python prepare_manifest.py \
        --dataset json \
        --data-files "train=${DATA_ROOT}/${sid}/metadata.jsonl" \
        --split train \
        --audio-column audio --text-column text \
        --target-sample-rate 44100 \
        --output-manifest "${DATA_ROOT}/${sid}/manifest.jsonl" \
        --latent-dir "${DATA_ROOT}/${sid}/latents" \
        --device cuda \
        >> "$prep_log" 2>&1
      local rc=$?
      if [ "$rc" -ne 0 ]; then
        echo "[gpu=${gpu}][${sid}] prep_manifest FAILED rc=${rc}" >&2
        touch "${LOCK_DIR}/${sid}.failed"
          continue
      fi
    fi

    echo "[gpu=${gpu}][${sid}] train"
    GPUS="$gpu" CONFIG="$CONFIG" BASE_CKPT="$BASE_CKPT" DATA_ROOT="$DATA_ROOT" \
      scripts/train/train_multi_speaker.sh "$sid"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "[gpu=${gpu}][${sid}] train FAILED rc=${rc}" >&2
      # Marked .failed (not .done) so workers move on instead of retry-looping
      # and a failure is never counted as finished; to retry later, remove the
      # .claimed and .failed files for this sid
      touch "${LOCK_DIR}/${sid}.failed"
    else
      echo "[gpu=${gpu}][${sid}] train DONE"
      touch "${LOCK_DIR}/${sid}.done"
    fi
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
failed=0
for sid in "${SPEAKERS[@]}"; do
  if [ -f "${LOCK_DIR}/${sid}.failed" ]; then failed=$((failed + 1)); fi
done
echo "=== stream_pipeline finished: host=$(hostname -s) failures=${failed} ==="
[ "$failed" -eq 0 ]
