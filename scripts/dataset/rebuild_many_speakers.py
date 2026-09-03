#!/usr/bin/env python3
"""Extract many speakers from one HF parquet dataset in a single pass.

rebuild_speaker_dataset.py walks every shard to collect one speaker. That is
fine for one or two, but the Genshin corpus is 203 shards and reading it once
takes about half an hour, so pulling 112 speakers out of it one at a time means
reading the same 203 shards 112 times. This reads them once and fans each row
out to whichever speaker it belongs to.

Row handling — silence trim, duration filter, loudness normalisation, and
keeping audio paired with its own transcription — is imported from that script
rather than reimplemented, so the two cannot drift apart.

Usage:
  uv run --no-sync python scripts/dataset/rebuild_many_speakers.py \\
    --repo-id ultemica/genshin-impact-voices \\
    --data-files 'data/ja/*.parquet' \\
    --speaker-map /app/plan.tsv \\
    --output-root data
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_speaker_dataset import _process_row


@dataclass
class _Speaker:
    sid: str
    out_dir: Path
    wav_dir: Path
    tmp_dir: Path
    handle: object
    prefix: str = ""
    written: int = 0
    skipped: int = 0


def _open_speaker(sid: str, root: Path, part: int | None) -> _Speaker:
    """Open one speaker for writing.

    `part` is the shard-offset of this process when several are splitting one
    corpus across machines. They share the speaker directory, so a part must
    not wipe it, and its files carry the part number to stay disjoint. Each
    part writes its own manifest; concatenating them yields metadata.jsonl.
    """
    out_dir = root / sid
    if part is None and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    wav_dir = out_dir / "wavs"
    tmp_dir = out_dir / "_tmp" if part is None else out_dir / f"_tmp{part:02d}"
    wav_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    manifest = "metadata.jsonl" if part is None else f"metadata.part{part:02d}.jsonl"
    return _Speaker(
        sid=sid,
        out_dir=out_dir,
        wav_dir=wav_dir,
        tmp_dir=tmp_dir,
        handle=(out_dir / manifest).open("w", encoding="utf-8"),
        prefix="" if part is None else f"p{part:02d}_",
    )


def _read_speaker_map(path: Path) -> dict[str, str]:
    """Source speaker name -> local speaker id."""
    wanted: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not line.strip():
            continue
        wanted[parts[1].strip()] = parts[0].strip()
    if not wanted:
        sys.exit(f"no speakers in {path}")
    return wanted


def _list_shards(repo_id: str, pattern: str, token: str | None) -> list[str]:
    api = HfApi(token=token)
    return sorted(
        f.path
        for f in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
        if hasattr(f, "path") and fnmatch.fnmatch(f.path, pattern)
    )


def _drain_shard(
    table,
    wanted: dict[str, str],
    speakers: dict[str, _Speaker],
    args: argparse.Namespace,
    pool: ThreadPoolExecutor,
) -> None:
    col = args.speaker_column or ("speaker" if "speaker" in table.column_names else "character")
    names: list[str] = table.column(col).to_pylist()
    jobs: list[tuple[int, _Speaker]] = []
    for row_idx, raw_name in enumerate(names):
        sid = wanted.get((raw_name or "").strip())
        if sid is None:
            continue
        spk = speakers.get(sid)
        if spk is None:
            spk = speakers[sid] = _open_speaker(sid, args.output_root, args.part)
        jobs.append((row_idx, spk))
    if not jobs:
        return

    def _run(job: tuple[int, _Speaker]) -> tuple[str, dict | None]:
        row_idx, spk = job
        scratch = spk.tmp_dir / f"r{row_idx:08d}"
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            return _process_row(
                table,
                row_idx,
                args,
                scratch,
                spk.wav_dir / f".stage_{row_idx:08d}.wav",
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    # ffmpeg is a subprocess, so threads overlap cleanly. map yields in input
    # order, which keeps the numbering and manifest order identical to a
    # serial run however the workers happen to interleave.
    for (_, spk), (status, record) in zip(jobs, pool.map(_run, jobs), strict=True):
        if status == "kept" and record is not None:
            final = spk.wav_dir / f"{spk.prefix}{spk.written:06d}.wav"
            Path(record["audio"]).replace(final)
            record["audio"] = str(final)
            spk.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            spk.written += 1
        else:
            spk.skipped += 1


def _close_all(speakers: dict[str, _Speaker], part: int | None) -> None:
    for spk in speakers.values():
        spk.handle.close()
        shutil.rmtree(spk.tmp_dir, ignore_errors=True)
        # A speaker that produced nothing would otherwise leave an empty
        # metadata.jsonl behind, which later stages read as "already done".
        # Under a shard split the other parts own the same directory, so drop
        # only this part's manifest and leave their work alone.
        if spk.written == 0:
            if part is None:
                shutil.rmtree(spk.out_dir, ignore_errors=True)
            else:
                (spk.out_dir / f"metadata.part{part:02d}.jsonl").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--data-files", required=True)
    ap.add_argument(
        "--speaker-map",
        required=True,
        type=Path,
        help="TSV of <local sid><tab><source speaker name>",
    )
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--speaker-column", default=None, help="default: speaker, else character")
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--silence-db", type=float, default=-40.0)
    ap.add_argument("--normalize-lufs", type=float, default=-20.0)
    ap.add_argument("--token", default=None)
    ap.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 1)),
        help="Concurrent ffmpeg conversions.",
    )
    ap.add_argument(
        "--shard-stride",
        type=int,
        default=1,
        help="Split the shards across this many machines (1 disables splitting).",
    )
    ap.add_argument(
        "--shard-offset",
        type=int,
        default=0,
        help="Which slice of the split this process takes, 0..stride-1.",
    )
    args = ap.parse_args()

    wanted = _read_speaker_map(args.speaker_map)
    print(f"{len(wanted)} speakers requested from {args.repo_id}")

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shards = _list_shards(args.repo_id, args.data_files, args.token)
    print(f"matched {len(shards)} shards")

    if args.workers < 1:
        sys.exit("--workers must be >= 1")
    if args.shard_stride < 1:
        sys.exit("--shard-stride must be >= 1")
    if not 0 <= args.shard_offset < args.shard_stride:
        sys.exit("--shard-offset must be in 0..stride-1")
    args.part = None if args.shard_stride == 1 else args.shard_offset
    if args.part is not None:
        shards = shards[args.shard_offset :: args.shard_stride]
        print(f"part {args.shard_offset}/{args.shard_stride}: {len(shards)} shards")
    print(f"{args.workers} conversion workers")

    args.output_root.mkdir(parents=True, exist_ok=True)
    speakers: dict[str, _Speaker] = {}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for shard in tqdm(shards, desc="shards"):
                local = hf_hub_download(args.repo_id, shard, repo_type="dataset", token=args.token)
                _drain_shard(pq.read_table(local), wanted, speakers, args, pool)
    finally:
        _close_all(speakers, args.part)

    done = {sid: spk for sid, spk in speakers.items() if spk.written > 0}
    print(f"\nextracted {len(done)} / {len(wanted)} speakers")
    for sid, spk in sorted(done.items(), key=lambda kv: -kv[1].written):
        print(f"  {sid:26} kept={spk.written:6} skipped={spk.skipped}")
    missing = sorted(set(wanted.values()) - set(done))
    if missing:
        print(f"no rows matched for {len(missing)}: {missing[:10]}")


if __name__ == "__main__":
    main()
