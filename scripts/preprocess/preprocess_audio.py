#!/usr/bin/env python3
"""Trim leading/trailing silence, loudness-normalize, filter by duration, sequence-rename."""
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def process_one(src: Path, dst: Path, silence_db: float, lufs: float) -> float | None:
    af = (
        f"silenceremove=1:0:{silence_db}dB,"
        f"areverse,silenceremove=1:0:{silence_db}dB,areverse,"
        f"loudnorm=I={lufs}:TP=-1.5:LRA=11"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src), "-af", af,
        "-c:a", "libvorbis", "-q:a", "5", str(dst),
    ]
    subprocess.run(cmd, check=True)
    try:
        return probe_duration(dst)
    except Exception:
        return None


def _worker(args):
    idx, src_str, tmp_str, silence_db, lufs = args
    src = Path(src_str)
    tmp = Path(tmp_str)
    try:
        dur = process_one(src, tmp, silence_db, lufs)
        return idx, src.name, tmp_str, dur
    except Exception:
        return idx, src.name, tmp_str, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--tmp", required=True)
    parser.add_argument("--min-seconds", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--silence-db", type=float, default=-40.0)
    parser.add_argument("--normalize-lufs", type=float, default=-23.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--glob", default="*.ogg")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob(args.glob))
    print(f"found {len(files)} source files in {src}")

    jobs = [
        (i, str(f), str(tmp / f.name), args.silence_db, args.normalize_lufs)
        for i, f in enumerate(files)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="preprocess"):
            results.append(fut.result())

    results.sort(key=lambda r: r[0])

    kept = 0
    dropped_short = 0
    dropped_long = 0
    dropped_err = 0
    seq = 0
    mapping_path = dst / "_source_map.tsv"
    with mapping_path.open("w", encoding="utf-8") as mf:
        mf.write("seq\tsource\tduration\n")
        for idx, orig, tmp_path, dur in results:
            if dur is None:
                dropped_err += 1
                continue
            if dur < args.min_seconds:
                dropped_short += 1
                Path(tmp_path).unlink(missing_ok=True)
                continue
            if dur > args.max_seconds:
                dropped_long += 1
                Path(tmp_path).unlink(missing_ok=True)
                continue
            new_name = f"{seq:05d}.ogg"
            Path(tmp_path).rename(dst / new_name)
            mf.write(f"{seq:05d}\t{orig}\t{dur:.3f}\n")
            seq += 1
            kept += 1

    print(f"kept={kept} short={dropped_short} long={dropped_long} err={dropped_err}")


if __name__ == "__main__":
    main()
