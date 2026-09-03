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
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_speaker_dataset import _process_row, _RowPaths


@dataclass
class _Speaker:
    sid: str
    out_dir: Path
    wav_dir: Path
    tmp_dir: Path
    handle: object
    written: int = 0
    skipped: int = 0


def _open_speaker(sid: str, root: Path) -> _Speaker:
    out_dir = root / sid
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    wav_dir = out_dir / "wavs"
    tmp_dir = out_dir / "_tmp"
    wav_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return _Speaker(
        sid=sid,
        out_dir=out_dir,
        wav_dir=wav_dir,
        tmp_dir=tmp_dir,
        handle=(out_dir / "metadata.jsonl").open("w", encoding="utf-8"),
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
    names: list[str],
    wanted: dict[str, str],
    speakers: dict[str, _Speaker],
    args: argparse.Namespace,
) -> None:
    for row_idx, raw_name in enumerate(names):
        sid = wanted.get((raw_name or "").strip())
        if sid is None:
            continue
        spk = speakers.get(sid)
        if spk is None:
            spk = speakers[sid] = _open_speaker(sid, args.output_root)
        status, record = _process_row(
            table,
            row_idx,
            args,
            _RowPaths(tmp_dir=spk.tmp_dir, wav_dir=spk.wav_dir),
            spk.written,
        )
        if status == "kept" and record is not None:
            spk.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            spk.written += 1
        else:
            spk.skipped += 1


def _close_all(speakers: dict[str, _Speaker]) -> None:
    for spk in speakers.values():
        spk.handle.close()
        shutil.rmtree(spk.tmp_dir, ignore_errors=True)
        # A speaker that produced nothing would otherwise leave an empty
        # metadata.jsonl behind, which later stages read as "already done".
        if spk.written == 0:
            shutil.rmtree(spk.out_dir, ignore_errors=True)


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
    args = ap.parse_args()

    wanted = _read_speaker_map(args.speaker_map)
    print(f"{len(wanted)} speakers requested from {args.repo_id}")

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shards = _list_shards(args.repo_id, args.data_files, args.token)
    print(f"matched {len(shards)} shards")

    args.output_root.mkdir(parents=True, exist_ok=True)
    speakers: dict[str, _Speaker] = {}
    try:
        for shard in tqdm(shards, desc="shards"):
            local = hf_hub_download(args.repo_id, shard, repo_type="dataset", token=args.token)
            table = pq.read_table(local)
            col = args.speaker_column or (
                "speaker" if "speaker" in table.column_names else "character"
            )
            _drain_shard(table, table.column(col).to_pylist(), wanted, speakers, args)
    finally:
        _close_all(speakers)

    done = {sid: spk for sid, spk in speakers.items() if spk.written > 0}
    print(f"\nextracted {len(done)} / {len(wanted)} speakers")
    for sid, spk in sorted(done.items(), key=lambda kv: -kv[1].written):
        print(f"  {sid:26} kept={spk.written:6} skipped={spk.skipped}")
    missing = sorted(set(wanted.values()) - set(done))
    if missing:
        print(f"no rows matched for {len(missing)}: {missing[:10]}")


if __name__ == "__main__":
    main()
