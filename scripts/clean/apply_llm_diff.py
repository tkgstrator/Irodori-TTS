#!/usr/bin/env python
"""Apply aggregated LLM diff to metadata_filtered.jsonl -> metadata.jsonl.

Diff rows: {file_name, original, cleaned, reason}. If `cleaned` is an empty
string or null, the row is dropped from the output.
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--diff", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    diff_map: dict[str, dict] = {}
    with args.diff.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            diff_map[d["file_name"]] = d

    kept = 0
    changed = 0
    dropped = 0
    with args.src.open() as f, args.out.open("w") as out:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            fn = row["file_name"]
            if fn in diff_map:
                d = diff_map[fn]
                cleaned = d.get("cleaned")
                reason = (d.get("reason") or "").lstrip().lower()
                if cleaned is None or cleaned == "" or reason.startswith("suspect"):
                    dropped += 1
                    continue
                if cleaned != row.get("text"):
                    row["text"] = cleaned
                    changed += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept={kept} changed={changed} dropped={dropped}")


if __name__ == "__main__":
    main()
