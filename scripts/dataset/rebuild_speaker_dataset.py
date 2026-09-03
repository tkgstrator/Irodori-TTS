#!/usr/bin/env python3
"""Re-extract a speaker from a HF parquet dataset, trim silence, filter by
duration — single-pass pipeline that keeps audio↔text pairing intact.

Why this exists: the older flow (extract_hf_speaker.py → preprocess_audio.py)
renumbers wavs by surviving order at the end of preprocess_audio
(`new_name = f"{seq:05d}.ogg"`) without updating metadata. Anything that later
joins `metadata.jsonl` with `wavs/<i>.wav` by file_name or list-position
ends up reading the WRONG audio for each text — which we observed in
practice (cps distribution 0.4–80).

This script does it in one pass, per parquet row:
  1. read (audio bytes, transcription) from the parquet row
  2. write the audio to a temp wav
  3. ffmpeg silenceremove + loudnorm
  4. probe duration; drop if outside [min, max]
  5. write the kept wav with a fresh sequential index AND the matching
     transcription line in the SAME iteration — so list-position pairing
     in the output is guaranteed correct.

Usage:
    PYTHONPATH=. uv run python scripts/dataset/rebuild_speaker_dataset.py \
        --repo-id ultemica/genshin-impact-voices \
        --data-files 'data/ja/train-*.parquet' \
        --speaker '神里绫华' \
        --output-dir data/ayaka \
        --min-seconds 1.5 --max-seconds 30.0 \
        --silence-db -40 --normalize-lufs -23
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int = 44100) -> None:
    data_size = len(pcm_bytes)
    with path.open("wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_bytes)


# A clip is at most a few tens of seconds, so a conversion that has not
# finished by now is wedged. Without a bound one stuck ffmpeg blocks every
# other worker, since the caller consumes results in order.
_FFMPEG_TIMEOUT_SECONDS = 120


def _trim_and_normalize(src: Path, dst: Path, silence_db: float, lufs: float) -> float | None:
    af = (
        f"silenceremove=1:0:{silence_db}dB,"
        f"areverse,silenceremove=1:0:{silence_db}dB,areverse,"
        f"loudnorm=I={lufs}:TP=-1.5:LRA=11"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-af",
            af,
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        check=True,
        timeout=_FFMPEG_TIMEOUT_SECONDS,
    )
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(dst)],
        capture_output=True,
        text=True,
        check=True,
        timeout=_FFMPEG_TIMEOUT_SECONDS,
    )
    out = r.stdout.strip()
    if not out or out == "N/A":
        return None
    try:
        return float(out)
    except ValueError:
        return None


def _process_row(
    table,
    row_idx: int,
    args: argparse.Namespace,
    tmp_dir: Path,
    dest: Path,
) -> tuple[str, dict | None]:
    """Decode, trim and normalise one row into `dest`.

    `tmp_dir` must be private to the caller: concurrent callers sharing one
    would overwrite each other's scratch files.
    """
    audio_col = table.column("audio")[row_idx].as_py()
    transcription = table.column("transcription")[row_idx].as_py()

    if not transcription or not transcription.strip():
        return "no_text", None
    if not audio_col or not audio_col.get("bytes"):
        return "no_audio", None

    raw_path = tmp_dir / "raw.wav"
    audio_bytes = audio_col["bytes"]
    if audio_bytes[:4] == b"RIFF":
        raw_path.write_bytes(audio_bytes)
    else:
        _write_wav(raw_path, audio_bytes, sample_rate=44100)

    trimmed_path = tmp_dir / "trimmed.wav"
    try:
        dur = _trim_and_normalize(
            raw_path,
            trimmed_path,
            silence_db=args.silence_db,
            lufs=args.normalize_lufs,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        dur = None
    if dur is None:
        return "err", None
    if dur < args.min_seconds:
        return "short", None
    if dur > args.max_seconds:
        return "long", None

    trimmed_path.replace(dest)
    record = {
        "audio": str(dest),
        "text": transcription.strip(),
        "duration": round(dur, 3),
    }
    return "kept", record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--data-files", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-seconds", type=float, default=1.5)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--silence-db", type=float, default=-40.0)
    ap.add_argument("--normalize-lufs", type=float, default=-23.0)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    # Stage temp files inside the output dir so the final rename stays on the
    # same filesystem (Path.replace fails with EXDEV across /tmp ↔ /home).
    tmp_root = out_dir / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.jsonl"

    api = HfApi(token=token)
    all_files = list(api.list_repo_tree(args.repo_id, repo_type="dataset", recursive=True))
    parquet_files = sorted(
        f.path for f in all_files if hasattr(f, "path") and fnmatch.fnmatch(f.path, args.data_files)
    )
    print(f"matched {len(parquet_files)} parquet shards for '{args.data_files}'")

    import pyarrow.parquet as pq

    written = 0
    counts = {"short": 0, "long": 0, "err": 0, "no_text": 0, "no_audio": 0}

    with (
        tempfile.TemporaryDirectory(dir=tmp_root) as tmp_str,
        metadata_path.open("w", encoding="utf-8") as meta_f,
    ):
        tmp_dir = Path(tmp_str)
        for shard_path in tqdm(parquet_files, desc="shards"):
            local_path = hf_hub_download(
                args.repo_id,
                shard_path,
                repo_type="dataset",
                token=token,
            )
            table = pq.read_table(local_path)
            speakers = table.column("speaker").to_pylist()
            for row_idx, spk in enumerate(speakers):
                if spk != args.speaker:
                    continue
                status, record = _process_row(
                    table, row_idx, args, tmp_dir, wav_dir / f"{written:06d}.wav"
                )
                if status == "kept":
                    meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                else:
                    counts[status] += 1

    with contextlib.suppress(OSError):
        tmp_root.rmdir()

    print(
        f"kept={written} short={counts['short']} long={counts['long']} "
        f"ffmpeg_err={counts['err']} no_text={counts['no_text']} no_audio={counts['no_audio']}"
    )
    print(f"wavs:     {wav_dir}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
