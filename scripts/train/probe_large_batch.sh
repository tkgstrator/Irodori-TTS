#!/usr/bin/env bash
# Probe a single GPU in an isolated container without touching the live run.
set -euo pipefail

usage() {
    echo 'usage: scripts/train/probe_large_batch.sh BATCH_SIZE [GPU_INDEX] [HOST] [--accumulation N] [--persistent-workers true|false] [--workers N] [--checkpoint true|false] [--compile true|false] [--steps N] [--dynamic-padding true|false] [--speaker NAME]'
}

positionals=()
accumulation=
persistent_workers=false
workers=16
checkpoint=true
compile=false
steps=30
dynamic_padding=false
speaker=cherry
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --accumulation|--persistent-workers|--workers|--checkpoint|--compile|--steps|--dynamic-padding|--speaker)
            option=$1
            if (( $# < 2 )); then
                echo "Missing value for $option" >&2
                exit 2
            fi
            case "$option" in
                --accumulation) accumulation=$2 ;;
                --persistent-workers) persistent_workers=$2 ;;
                --workers) workers=$2 ;;
                --checkpoint) checkpoint=$2 ;;
                --compile) compile=$2 ;;
                --steps) steps=$2 ;;
                --dynamic-padding) dynamic_padding=$2 ;;
                --speaker) speaker=$2 ;;
            esac
            shift 2 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) positionals+=("$1"); shift ;;
    esac
done
if (( ${#positionals[@]} < 1 || ${#positionals[@]} > 3 )); then
    usage >&2
    exit 2
fi
batch_size=${positionals[0]}
gpu=${positionals[1]:-1}
host=${positionals[2]:-g20}
if [[ ! $batch_size =~ ^[1-9][0-9]*$ || ! $gpu =~ ^[0-9]+$ || ! $workers =~ ^(0|[1-9][0-9]*)$ || ! $steps =~ ^[1-9][0-9]*$ ]]; then
    echo 'BATCH_SIZE and steps must be positive; GPU_INDEX and workers must be nonnegative integers' >&2
    exit 2
fi
legacy_root=${PROBE_LEGACY_ROOT:-/home/smorimoto/Developer/Irodori-TTS-v4/data}
if [[ ! $speaker =~ ^[a-z0-9_]+$ || ! $legacy_root =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
    echo 'Invalid speaker name or PROBE_LEGACY_ROOT' >&2
    exit 2
fi
for value in "$persistent_workers" "$checkpoint" "$compile" "$dynamic_padding"; do
    if [[ $value != true && $value != false ]]; then
        echo 'Boolean options require true or false' >&2
        exit 2
    fi
done
if [[ -z $accumulation ]]; then
    accumulation=1
    if (( batch_size <= 80 && 80 % batch_size == 0 )); then
        accumulation=$((80 / batch_size))
    fi
fi
if [[ ! $accumulation =~ ^[1-9][0-9]*$ ]]; then
    echo 'accumulation must be a positive integer' >&2
    exit 2
fi
effective_batch=$((batch_size * accumulation))
echo "Trial: speaker=$speaker batch=$batch_size accumulation=$accumulation effective_batch=$effective_batch workers=$workers persistent=$persistent_workers checkpoint=$checkpoint compile=$compile steps=$steps dynamic_padding=$dynamic_padding"
echo 'Data: full speaker manifest, validation disabled, seed=42. Discard at least the first 10 optimizer steps; compare multiple complete epochs after warmup.'
echo 'Microbatch-dependent length grouping and drop_last change sample order/coverage. This is an end-to-end trial, not a shape-controlled GPU benchmark.'
if (( effective_batch != 80 )); then
    echo 'Capacity trial: effective batch differs from the baseline 80; do not compare its throughput as an equivalent training configuration.'
fi
if (( steps <= 10 )); then
    echo 'Short capacity probe: too few steps for a warmed-up throughput comparison.'
fi

ssh -o BatchMode=yes "$host" bash -s -- "$gpu" "$batch_size" "$accumulation" "$persistent_workers" "$workers" "$checkpoint" "$compile" "$steps" "$dynamic_padding" "$speaker" "$legacy_root" <<'REMOTE'
set -euo pipefail
gpu=$1
batch_size=$2
accumulation=$3
persistent_workers=$4
workers=$5
checkpoint=$6
compile=$7
steps=$8
dynamic_padding=$9
speaker=${10}
legacy_root=${11}
manifest=data/cherry/manifest.jsonl
legacy_mount=()
if [[ $speaker != cherry ]]; then
    manifest=/legacy/$speaker/manifest.jsonl
    legacy_mount=(-v "$legacy_root:/legacy:ro")
fi
container=irodori-batch-probe-"${gpu}"-"${batch_size}"
source_root=$(docker inspect irodori-v4-large-cherry --format '{{range .Mounts}}{{if eq .Destination "/app"}}{{.Source}}{{end}}{{end}}')
image=$(docker inspect irodori-v4-large-cherry --format '{{.Config.Image}}')
if [[ -z $source_root || -z $image ]]; then
    echo 'Cannot locate the source and image of the running Large container' >&2
    exit 1
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((gpu + 1))p" | tr -d ' ')
if [[ ! $used =~ ^[0-9]+$ || $used -gt 4096 ]]; then
    echo "GPU $gpu is occupied or unavailable (used: ${used:-unknown} MiB); stopping" >&2
    exit 1
fi

peak_file=$(mktemp)
monitor_pid=
cleanup() {
    if [[ -n $monitor_pid ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    rm -f "$peak_file"
}
trap cleanup EXIT
(
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
            sed -n "$((gpu + 1))p" | tr -d ' ' >> "$peak_file"
        sleep 1
    done
) &
monitor_pid=$!

echo "Probing batch=$batch_size on $(hostname) GPU=$gpu (initial memory=${used} MiB)"
set +e
docker run --rm --name "$container" --gpus "device=$gpu" --shm-size=8g \
    --entrypoint bash -e UV_PROJECT_ENVIRONMENT=/usr/local \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$source_root:/app:ro" "${legacy_mount[@]}" \
    -v irodori_v4_large_uv:/root/.cache/uv \
    -v irodori_v4_large_hf:/root/.cache/huggingface \
    "$image" -lc '
        set -euo pipefail
        batch_size=$1
        accumulation=$2
        persistent_workers=$3
        workers=$4
        checkpoint=$5
        compile=$6
        steps=$7
        dynamic_padding=$8
        manifest=$9
        uv sync --quiet --frozen --no-dev
        uv run --no-sync python - "$persistent_workers" "$checkpoint" "$compile" "$dynamic_padding" <<PY
import sys
from dataclasses import fields
import yaml
from irodori_tts.config import TrainConfig
dynamic_padding = sys.argv[4] == "true"
supports_dynamic_padding = any(f.name == "dynamic_condition_padding" for f in fields(TrainConfig))
if dynamic_padding and not supports_dynamic_padding:
    raise SystemExit("Dynamic padding is unsupported by the container source; sync the opt-in code before probing. Live source was not changed.")
p = yaml.safe_load(open("configs/train_v4_large_lora.yaml"))
p["sample_generation"]["enabled"] = False
for key, value in zip(
    ("dataloader_persistent_workers", "gradient_checkpointing", "compile_model"),
    sys.argv[1:4],
    strict=True,
):
    p["train"][key] = value == "true"
if supports_dynamic_padding:
    p["train"]["dynamic_condition_padding"] = dynamic_padding
else:
    p["train"].pop("dynamic_condition_padding", None)
yaml.safe_dump(p, open("/tmp/batch-probe-config.yaml", "w"))
PY
        uv run --no-sync python train.py \
            --config /tmp/batch-probe-config.yaml \
            --manifest "$manifest" \
            --output-dir /tmp/batch-probe \
            --init-checkpoint models/Irodori-TTS-v4-Large/model.safetensors \
            --metrics-backend none \
            --batch-size "$batch_size" \
            --gradient-accumulation-steps "$accumulation" \
            --num-workers "$workers" --seed 42 \
            --max-steps "$steps" --valid-ratio 0 --valid-every 0 \
            --save-every 100000 --checkpoint-best-n 0 --log-every 1
    ' probe "$batch_size" "$accumulation" "$persistent_workers" "$workers" "$checkpoint" "$compile" "$steps" "$dynamic_padding" "$manifest"
status=$?
set -e
peak=$(sort -nr "$peak_file" | head -1)
echo "Probe result: batch=$batch_size accumulation=$accumulation exit=$status peak_gpu_used=${peak:-unknown} MiB (initial=${used} MiB)"
exit "$status"
REMOTE
