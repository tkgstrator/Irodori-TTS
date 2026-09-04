#!/usr/bin/env python3
"""Write the recovered transcripts back into the source parquet dataset.

The source corpus ships audio without text, so the curated transcripts only
existed inside the encoded archives, unpaired. `recover_source_pairing.py`
rebuilds the pairing; this writes it back as a `transcription` column so the
corpus stands on its own and latents become a cache that can be regenerated.

Rows preprocessing discarded get an empty string rather than being dropped, so
the parquet keeps its original row count and order and stays comparable with
the untouched shards.

Usage:
  uv run --no-sync python scripts/dataset/add_transcription_column.py \\
    --speaker anan --character AnAn \\
    --pairing data/anan/pairing.jsonl \\
    --out-dir data/_parquet_with_text
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TRANSCRIPTION = "transcription"


def _shard_names(repo_id: str, character: str, token: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request) as response:
        meta = json.load(response)
    names = sorted(
        s["rfilename"]
        for s in meta["siblings"]
        if s["rfilename"].startswith(f"data/{character}/") and s["rfilename"].endswith(".parquet")
    )
    if not names:
        sys.exit(f"no parquet shards under data/{character}/ in {repo_id}")
    return names


def _fetch(repo_id: str, name: str, token: str) -> pa.Table:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{name}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request) as response:
        return pq.read_table(io.BytesIO(response.read()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--repo-id", default="ultemica/magical-girl-witch-trials-voice")
    parser.add_argument("--pairing", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--min-ncc",
        type=float,
        default=0.0,
        help="skip pairings scoring below this instead of writing their text",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is required to read the source dataset")

    pairs = [json.loads(line) for line in args.pairing.read_text().splitlines() if line.strip()]
    by_index = {
        p["source_index"]: p["text"] for p in pairs if float(p.get("ncc", 1.0)) >= args.min_ncc
    }
    skipped = len(pairs) - len(by_index)

    shards = _shard_names(args.repo_id, args.character, token)
    out_root = args.out_dir / "data" / args.character
    out_root.mkdir(parents=True, exist_ok=True)

    row_base = 0
    filled = 0
    total = 0
    for name in shards:
        table = _fetch(args.repo_id, name, token)
        if TRANSCRIPTION in table.column_names:
            sys.exit(f"{name} already has a '{TRANSCRIPTION}' column; refusing to overwrite")
        # source_index in the pairing counts rows across the whole character,
        # concatenated in shard-name order, which is how it was built.
        texts = [by_index.get(row_base + i, "") for i in range(table.num_rows)]
        filled += sum(1 for t in texts if t)
        total += table.num_rows
        row_base += table.num_rows

        # Sit transcription right after audio so it reads like the HSR corpus.
        names = table.column_names
        insert_at = names.index("audio") + 1 if "audio" in names else len(names)
        table = table.add_column(insert_at, TRANSCRIPTION, pa.array(texts, type=pa.string()))
        destination = out_root / Path(name).name
        pq.write_table(table, destination)
        print(f"  {destination}  rows={table.num_rows}")

    print(f"{args.speaker}: filled {filled}/{total} rows ({total - filled} left empty)")
    if skipped:
        print(f"  {skipped} pairings skipped below --min-ncc {args.min_ncc}")
    if filled != len(by_index):
        print(f"  warning: {len(by_index) - filled} pairings did not land in any shard")


if __name__ == "__main__":
    main()
