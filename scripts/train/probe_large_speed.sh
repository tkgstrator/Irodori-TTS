#!/usr/bin/env bash
# Compare one change at a time, retaining an effective batch of 80.
set -euo pipefail

gpu=${1:-1}
host=${2:-g20}
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
output_dir="$repo_root/outputs/batch-probes/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$output_dir"
printf 'case\texit\tlog\n' > "$output_dir/results.tsv"

probe() {
    local name=$1
    local batch=$2
    shift 2
    local log="$output_dir/$name.log"
    local status=0
    echo "=== $name ==="
    if bash "$repo_root/scripts/train/probe_large_batch.sh" "$batch" "$gpu" "$host" \
        --steps 30 "$@" 2>&1 | tee "$log"; then
        status=0
    else
        status=$?
    fi
    printf '%s\t%s\t%s\n' "$name" "$status" "$log" >> "$output_dir/results.tsv"
    if (( status != 0 )) && ! grep -Eqi 'out of memory|OutOfMemoryError' "$log"; then
        echo "Non-OOM failure in $name; stopping. Results: $output_dir" >&2
        return "$status"
    fi
}

probe baseline-80 80 --persistent-workers false --workers 16 --checkpoint true
probe persistent-16 80 --persistent-workers true --workers 16 --checkpoint true
probe persistent-8 80 --persistent-workers true --workers 8 --checkpoint true
probe persistent-4 80 --persistent-workers true --workers 4 --checkpoint true
probe no-checkpoint-80 80 --persistent-workers true --workers 8 --checkpoint false
probe no-checkpoint-40 40 --accumulation 2 --persistent-workers true --workers 8 --checkpoint false

echo "Comparison complete. Logs and exit codes: $output_dir"
echo 'Compare the steady portion after the first 10 updates, not startup or final saving.'
