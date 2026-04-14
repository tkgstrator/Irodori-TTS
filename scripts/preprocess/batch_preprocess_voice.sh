#!/usr/bin/env bash
# Batch preprocess (extract + trim/normalize/filter + transcribe) for
# multiple Voice.zip speakers, sequentially. Each speaker gets its own
# data/<name>/ dir and a dedicated preprocess.log.
#
# Usage: scripts/preprocess/batch_preprocess_voice.sh [speaker1 speaker2 ...]
#   If no args, runs the full default list below.
#
# This script only runs through Whisper transcription. LLM cleaning and
# DACVAE latent encoding are handled separately once per-speaker rules
# are ready.

set -euo pipefail

cd "$(dirname "$0")/.."

ZIP=data/Voice.zip
MIN_SEC=1.5
MAX_SEC=30.0
SILENCE_DB=-40
LUFS=-23
WHISPER_MODEL=large-v3
WORKERS=12

# name_in_zip:output_dir
DEFAULT_SPEAKERS=(
  "Margo:margo"
  "Leia:leia"
  "Coco:coco"
  "Alisa:alisa"
  "Hanna:hanna"
  "Meruru:meruru"
  "Nanoka:nanoka"
  "Miria:miria"
  "Noah:noah"
  "Yuki:yuki"
  "AnAn:anan"
)

if [ $# -gt 0 ]; then
  SPEAKERS=("$@")
else
  SPEAKERS=("${DEFAULT_SPEAKERS[@]}")
fi

for entry in "${SPEAKERS[@]}"; do
  zip_name="${entry%%:*}"
  out_name="${entry##*:}"
  dst="data/${out_name}"
  log="${dst}/preprocess.log"
  src_dir="data/_${out_name}_raw"

  mkdir -p "$dst" "$src_dir"
  echo "=== [$(date +%H:%M)] === speaker=${out_name} zip_name=${zip_name}" | tee -a "$log"

  if [ ! -d "${src_dir}" ] || [ -z "$(ls -A "${src_dir}" 2>/dev/null)" ]; then
    echo "--- step: extract ---" | tee -a "$log"
    unzip -j -q "${ZIP}" "Voice/*_${zip_name}[0-9]*.ogg" -d "${src_dir}" 2>&1 | tee -a "$log" || true
    echo "extracted $(ls "${src_dir}" | wc -l) files" | tee -a "$log"
  else
    echo "[skip extract] ${src_dir} already populated" | tee -a "$log"
  fi

  echo "--- step: preprocess (trim/normalize/filter) ---" | tee -a "$log"
  uv run --no-sync python scripts/preprocess/preprocess_audio.py \
    --src "${src_dir}" \
    --dst "${dst}/wavs" \
    --tmp "${dst}/_tmp" \
    --min-seconds "${MIN_SEC}" \
    --max-seconds "${MAX_SEC}" \
    --silence-db "${SILENCE_DB}" \
    --normalize-lufs "${LUFS}" \
    --workers "${WORKERS}" 2>&1 | tee -a "$log"

  echo "--- step: transcribe ---" | tee -a "$log"
  uv run --no-sync python scripts/preprocess/transcribe_dir.py \
    --audio-dir "${dst}/wavs" \
    --output "${dst}/metadata_wts.jsonl" \
    --model "${WHISPER_MODEL}" 2>&1 | tee -a "$log"

  echo "--- step: heuristic filter ---" | tee -a "$log"
  uv run --no-sync python scripts/preprocess/filter_metadata_voice.py \
    --src "${dst}/metadata_wts.jsonl" \
    --out "${dst}/metadata_filtered.jsonl" \
    --rejected "${dst}/metadata_rejected.jsonl" 2>&1 | tee -a "$log"

  echo "=== [$(date +%H:%M)] done: ${out_name} ===" | tee -a "$log"
done

echo "all speakers done."
