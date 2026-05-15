#!/usr/bin/env python3
"""Join _source_map.tsv (post-trim seq -> original filename) with the raw
metadata.jsonl (original filename -> text) to produce metadata.jsonl/metadata_wts.jsonl
for the trimmed dataset.

Skips records whose original file was dropped during preprocess.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-map", required=True, help="path to _source_map.tsv")
    p.add_argument("--raw-metadata", required=True, help="path to original metadata.jsonl with {file_name, text, ...}")
    p.add_argument("--out", required=True, help="output metadata jsonl")
    p.add_argument("--audio-dir", required=True, help="directory the trimmed ogg files live in (used to build the 'audio' path)")
    args = p.parse_args()

    raw = {}
    with open(args.raw_metadata, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            raw[rec["file_name"]] = rec.get("text", "")

    audio_dir = args.audio_dir.rstrip("/")
    n_out = 0
    n_missing = 0
    with open(args.source_map, encoding="utf-8") as fmap, open(args.out, "w", encoding="utf-8") as fo:
        reader = csv.DictReader(fmap, delimiter="\t")
        for row in reader:
            seq = row["seq"]
            src = row["source"]
            text = raw.get(src)
            if text is None:
                n_missing += 1
                continue
            audio_path = f"{audio_dir}/{seq}.ogg"
            fo.write(json.dumps({"audio": audio_path, "text": text}, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"wrote {n_out} records to {args.out}; missing={n_missing}")


if __name__ == "__main__":
    main()
