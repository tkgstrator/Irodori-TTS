#!/usr/bin/env python3
"""Extract a single speaker's data from a HF parquet dataset and save as
wavs + metadata.jsonl suitable for the Irodori-TTS preprocessing pipeline.

Usage:
    python scripts/dataset/extract_hf_speaker.py \
        --repo-id ultemica/genshin-impact-voices \
        --data-files 'data/ja/train-*.parquet' \
        --speaker '神里绫华' \
        --output-dir data/ayaka
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int, num_channels: int = 1, sample_width: int = 2) -> None:
    """Write raw PCM bytes as a WAV file."""
    data_size = len(pcm_bytes)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * num_channels * sample_width))
        f.write(struct.pack("<H", num_channels * sample_width))
        f.write(struct.pack("<H", sample_width * 8))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a speaker from HF parquet dataset")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--data-files", required=True, help="Glob pattern for parquet files")
    parser.add_argument("--speaker", required=True, help="Exact speaker name to filter")
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. data/ayaka)")
    parser.add_argument("--token", default=None, help="HF token (default: $HF_TOKEN)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # List matching parquet files in the repo
    api = HfApi(token=token)
    all_files = list(api.list_repo_tree(args.repo_id, repo_type="dataset", recursive=True))
    import fnmatch
    parquet_files = sorted([
        f.path for f in all_files
        if hasattr(f, "path") and fnmatch.fnmatch(f.path, args.data_files)
    ])
    print(f"Found {len(parquet_files)} parquet shards matching '{args.data_files}'")

    import pyarrow.parquet as pq

    written = 0
    metadata_rows = []

    for shard_path in tqdm(parquet_files, desc="Scanning shards"):
        local_path = hf_hub_download(
            args.repo_id, shard_path, repo_type="dataset", token=token,
        )
        table = pq.read_table(local_path)
        speakers = table.column("speaker").to_pylist()

        for row_idx, spk in enumerate(speakers):
            if spk != args.speaker:
                continue

            audio_col = table.column("audio")[row_idx].as_py()
            transcription = table.column("transcription")[row_idx].as_py()

            if not audio_col or not audio_col.get("bytes"):
                continue
            if not transcription:
                continue

            wav_name = f"{written:06d}.wav"
            wav_path = wav_dir / wav_name

            audio_bytes = audio_col["bytes"]
            # Check if already a WAV (starts with RIFF header)
            if audio_bytes[:4] == b"RIFF":
                with open(wav_path, "wb") as f:
                    f.write(audio_bytes)
            else:
                # Raw PCM — shouldn't happen for this dataset but handle just in case
                _write_wav(wav_path, audio_bytes, sample_rate=44100)

            metadata_rows.append({
                "file_name": f"wavs/{wav_name}",
                "text": transcription,
            })
            written += 1

    # Write metadata.jsonl
    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone. Extracted {written} samples for speaker '{args.speaker}'")
    print(f"  wavs:     {wav_dir}")
    print(f"  metadata: {meta_path}")


if __name__ == "__main__":
    main()
