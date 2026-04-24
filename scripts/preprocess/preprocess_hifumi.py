#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["openpyxl", "tqdm"]
# ///
"""Preprocess data/hifumi nested-dir dataset into data/hifumi/wavs/*.wav + metadata.jsonl.

- Flattens 12 Japanese-named subdirectories
- Trims leading/trailing silence + loudness-normalizes via ffmpeg
- Outputs pcm_s16le .wav with naming `<romaji>_<stem>.wav`
- Builds metadata.jsonl by joining xlsx (col K, col N) → col P
- Reports unmatched files for manual fill
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import openpyxl
from tqdm import tqdm

DIR_MAP = {
    "勝利": "win",
    "敗北": "lose",
    "引き分け時": "draw",
    "囲い": "kakoi",
    "戦法": "senpou",
    "手筋": "tesuji",
    "開始時": "game",
    "終局時": "end",
    "秒読み": "byoyomi",
    "あと何分_編集版": "remain",
    "特殊": "special",
    "新規戦法": "new_senpou",
}


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
        "-c:a", "pcm_s16le", str(dst),
    ]
    subprocess.run(cmd, check=True)
    try:
        return probe_duration(dst)
    except Exception:
        return None


def _worker(args):
    src_str, tmp_str, silence_db, lufs = args
    src = Path(src_str)
    tmp = Path(tmp_str)
    try:
        dur = process_one(src, tmp, silence_db, lufs)
        return str(src), tmp_str, dur
    except Exception:
        return str(src), tmp_str, None


def load_xlsx_maps(xlsx_path: Path) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Return ((folder, stem) -> text, stem -> text) lookup maps from 'all' sheet."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["all"]
    by_pair: dict[tuple[str, str], str] = {}
    by_stem: dict[str, str] = {}
    for row in ws.iter_rows(min_row=3, values_only=True):
        folder = row[10]  # col K
        name = row[13]    # col N
        text = row[15]    # col P
        if name is None or text is None:
            continue
        name_s = str(name).strip()
        text_s = str(text).strip()
        if folder:
            by_pair[(str(folder).strip(), name_s)] = text_s
        # first-wins for stem-only fallback
        by_stem.setdefault(name_s, text_s)
    return by_pair, by_stem


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/hifumi")
    p.add_argument("--dst", default="data/hifumi")
    p.add_argument("--xlsx", default=None,
                   help="Path to xlsx; if omitted, glob data/hifumi/*.xlsx")
    p.add_argument("--min-seconds", type=float, default=0.5)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--silence-db", type=float, default=-40.0)
    p.add_argument("--normalize-lufs", type=float, default=-23.0)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    wavs_dir = dst / "wavs"
    tmp_dir = dst / "_tmp_preprocess"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate source files
    jobs: list[tuple[str, str, str, str, float, float]] = []
    items: list[tuple[str, str, str, Path]] = []  # (dir_jp, dir_romaji, stem, tmp_path)
    for dir_jp, romaji in DIR_MAP.items():
        d = src / dir_jp
        if not d.is_dir():
            print(f"WARN: missing {d}")
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() != ".wav":
                continue
            stem = f.stem
            out_name = f"{romaji}_{stem}.wav"
            tmp_path = tmp_dir / out_name
            items.append((dir_jp, romaji, stem, tmp_path))
            jobs.append((str(f), str(tmp_path), args.silence_db, args.normalize_lufs))

    print(f"found {len(items)} source wav files across {len(DIR_MAP)} dirs")

    # Process in parallel
    results: dict[str, float | None] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="preprocess"):
            src_s, tmp_s, dur = fut.result()
            results[src_s] = dur

    # Filter by duration, move survivors, build metadata
    if args.xlsx:
        xlsx_path = Path(args.xlsx)
    else:
        xlsx_candidates = sorted(src.glob("*.xlsx"))
        if not xlsx_candidates:
            raise SystemExit(f"no xlsx found under {src}")
        xlsx_path = xlsx_candidates[0]
    print(f"xlsx: {xlsx_path}")
    by_pair, by_stem = load_xlsx_maps(xlsx_path)

    kept = 0
    dropped_short = 0
    dropped_long = 0
    dropped_err = 0
    metadata: list[dict] = []
    unmatched: list[str] = []
    source_map: list[tuple[str, str, str, float]] = []

    for dir_jp, romaji, stem, tmp_path in items:
        src_full = str(src / dir_jp / f"{stem}.wav")
        dur = results.get(src_full)
        out_name = f"{romaji}_{stem}.wav"
        if dur is None:
            dropped_err += 1
            Path(tmp_path).unlink(missing_ok=True)
            continue
        if dur < args.min_seconds:
            dropped_short += 1
            Path(tmp_path).unlink(missing_ok=True)
            continue
        if dur > args.max_seconds:
            dropped_long += 1
            Path(tmp_path).unlink(missing_ok=True)
            continue
        final_path = wavs_dir / out_name
        shutil.move(str(tmp_path), str(final_path))
        source_map.append((out_name, dir_jp, stem, dur))

        text = by_pair.get((romaji, stem)) or by_stem.get(stem) or ""
        if not text:
            unmatched.append(out_name)
        metadata.append({"file_name": out_name, "text": text})
        kept += 1

    # Clean tmp
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    # Write metadata.jsonl
    meta_path = dst / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for rec in metadata:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write _source_map.tsv
    map_path = dst / "_source_map.tsv"
    with map_path.open("w", encoding="utf-8") as f:
        f.write("out_name\tsource_dir\tsource_stem\tduration\n")
        for out_name, dir_jp, stem, dur in source_map:
            f.write(f"{out_name}\t{dir_jp}\t{stem}\t{dur:.3f}\n")

    # Report
    print()
    print(f"=== preprocess done ===")
    print(f"kept={kept} short={dropped_short} long={dropped_long} err={dropped_err}")
    print(f"metadata: {len(metadata)} rows, unmatched (empty text): {len(unmatched)}")
    print(f"wrote: {meta_path}")
    print(f"wrote: {map_path}")
    if unmatched:
        print()
        print(f"=== unmatched files ({len(unmatched)}) — fill in metadata.jsonl manually ===")
        for n in unmatched:
            print(f"  {n}")


if __name__ == "__main__":
    main()
