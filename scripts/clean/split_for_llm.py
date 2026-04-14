#!/usr/bin/env python
"""Split metadata_filtered.jsonl into fixed-size batches for parallel LLM cleaning."""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=150)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for existing in args.out_dir.glob("batch_*.jsonl"):
        existing.unlink()

    with args.src.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    n = 0
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        out = args.out_dir / f"batch_{n:03d}.jsonl"
        with out.open("w") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
    print(f"wrote {n} batches ({len(rows)} rows) -> {args.out_dir}")


if __name__ == "__main__":
    main()
