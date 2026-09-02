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
import contextlib
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(r.stdout.strip())


def process_one(src: Path, dst: Path, silence_db: float, lufs: float) -> float | None:
    af = (
        f"silenceremove=1:0:{silence_db}dB,"
        f"areverse,silenceremove=1:0:{silence_db}dB,areverse,"
        f"loudnorm=I={lufs}:TP=-1.5:LRA=11"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-af",
        af,
        "-c:a",
        "pcm_s16le",
        str(dst),
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
        name = row[13]  # col N
        text = row[15]  # col P
        if name is None or text is None:
            continue
        name_s = str(name).strip()
        text_s = str(text).strip()
        if folder:
            by_pair[(str(folder).strip(), name_s)] = text_s
        # first-wins for stem-only fallback
        by_stem.setdefault(name_s, text_s)
    return by_pair, by_stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/hifumi")
    p.add_argument("--dst", default="data/hifumi")
    p.add_argument("--xlsx", default=None, help="Path to xlsx; if omitted, glob data/hifumi/*.xlsx")
    p.add_argument("--min-seconds", type=float, default=0.5)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--silence-db", type=float, default=-40.0)
    p.add_argument("--normalize-lufs", type=float, default=-23.0)
    p.add_argument("--workers", type=int, default=12)
    return p.parse_args()


def discover_jobs(
    src: Path, tmp_dir: Path, args: argparse.Namespace
) -> tuple[list[tuple[str, str, str, float]], list[tuple[str, str, str, Path]]]:
    """Enumerate source wav files. Returns (worker jobs, (dir_jp, romaji, stem, tmp_path) items)."""
    jobs: list[tuple[str, str, str, float]] = []
    items: list[tuple[str, str, str, Path]] = []
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
    return jobs, items


def run_preprocess_jobs(jobs: list[tuple], workers: int) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="preprocess"):
            src_s, _tmp_s, dur = fut.result()
            results[src_s] = dur
    return results


def resolve_xlsx_path(src: Path, xlsx_arg: str | None) -> Path:
    if xlsx_arg:
        return Path(xlsx_arg)
    xlsx_candidates = sorted(src.glob("*.xlsx"))
    if not xlsx_candidates:
        raise SystemExit(f"no xlsx found under {src}")
    return xlsx_candidates[0]


@dataclass
class _FilterInputs:
    src: Path
    wavs_dir: Path
    args: argparse.Namespace
    by_pair: dict[tuple[str, str], str]
    by_stem: dict[str, str]


def filter_and_collect(
    items: list[tuple[str, str, str, Path]],
    results: dict[str, float | None],
    inputs: _FilterInputs,
) -> tuple[list[dict], list[str], list[tuple[str, str, str, float]], dict[str, int]]:
    """Filter by duration, move survivors into wavs_dir, and build metadata rows."""
    src, wavs_dir, args = inputs.src, inputs.wavs_dir, inputs.args
    by_pair, by_stem = inputs.by_pair, inputs.by_stem
    counts = {"kept": 0, "short": 0, "long": 0, "err": 0}
    metadata: list[dict] = []
    unmatched: list[str] = []
    source_map: list[tuple[str, str, str, float]] = []

    for dir_jp, romaji, stem, tmp_path in items:
        src_full = str(src / dir_jp / f"{stem}.wav")
        dur = results.get(src_full)
        out_name = f"{romaji}_{stem}.wav"
        if dur is None:
            counts["err"] += 1
            Path(tmp_path).unlink(missing_ok=True)
            continue
        if dur < args.min_seconds:
            counts["short"] += 1
            Path(tmp_path).unlink(missing_ok=True)
            continue
        if dur > args.max_seconds:
            counts["long"] += 1
            Path(tmp_path).unlink(missing_ok=True)
            continue
        final_path = wavs_dir / out_name
        shutil.move(str(tmp_path), str(final_path))
        source_map.append((out_name, dir_jp, stem, dur))

        text = by_pair.get((romaji, stem)) or by_stem.get(stem) or ""
        if not text:
            unmatched.append(out_name)
        metadata.append({"file_name": out_name, "text": text})
        counts["kept"] += 1

    return metadata, unmatched, source_map, counts


def write_metadata(meta_path: Path, metadata: list[dict]) -> None:
    with meta_path.open("w", encoding="utf-8") as f:
        for rec in metadata:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_source_map(map_path: Path, source_map: list[tuple[str, str, str, float]]) -> None:
    with map_path.open("w", encoding="utf-8") as f:
        f.write("out_name\tsource_dir\tsource_stem\tduration\n")
        for out_name, dir_jp, stem, dur in source_map:
            f.write(f"{out_name}\t{dir_jp}\t{stem}\t{dur:.3f}\n")


def print_report(
    counts: dict[str, int],
    metadata: list[dict],
    unmatched: list[str],
    meta_path: Path,
    map_path: Path,
) -> None:
    print()
    print("=== preprocess done ===")
    print(
        f"kept={counts['kept']} short={counts['short']} long={counts['long']} err={counts['err']}"
    )
    print(f"metadata: {len(metadata)} rows, unmatched (empty text): {len(unmatched)}")
    print(f"wrote: {meta_path}")
    print(f"wrote: {map_path}")
    if unmatched:
        print()
        print(f"=== unmatched files ({len(unmatched)}) — fill in metadata.jsonl manually ===")
        for n in unmatched:
            print(f"  {n}")


def main() -> None:
    args = parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    wavs_dir = dst / "wavs"
    tmp_dir = dst / "_tmp_preprocess"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    jobs, items = discover_jobs(src, tmp_dir, args)
    print(f"found {len(items)} source wav files across {len(DIR_MAP)} dirs")

    results = run_preprocess_jobs(jobs, args.workers)

    xlsx_path = resolve_xlsx_path(src, args.xlsx)
    print(f"xlsx: {xlsx_path}")
    by_pair, by_stem = load_xlsx_maps(xlsx_path)

    metadata, unmatched, source_map, counts = filter_and_collect(
        items,
        results,
        _FilterInputs(src=src, wavs_dir=wavs_dir, args=args, by_pair=by_pair, by_stem=by_stem),
    )

    with contextlib.suppress(OSError):
        tmp_dir.rmdir()

    meta_path = dst / "metadata.jsonl"
    write_metadata(meta_path, metadata)

    map_path = dst / "_source_map.tsv"
    write_source_map(map_path, source_map)

    print_report(counts, metadata, unmatched, meta_path, map_path)


if __name__ == "__main__":
    main()
