#!/usr/bin/env python3
"""Heuristic cleanup for transcribed metadata jsonl."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

NON_VERBAL_RE = re.compile(r"^[あぁはひふへほっー…ぁぃぅぇぉっ、。！？!?\s]+$")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("!", "！").replace("?", "？")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def repeated_char_ratio(text: str) -> float:
    stripped = re.sub(r"[、。！？\s]", "", text)
    if not stripped:
        return 1.0
    counts = Counter(stripped)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(stripped)


def ascii_letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = len(ASCII_LETTER_RE.findall(text))
    return letters / len(text)


def judge(text: str, min_chars: int, rep_threshold: float, ascii_threshold: float) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if NON_VERBAL_RE.match(text):
        return "non_verbal"
    if repeated_char_ratio(text) >= rep_threshold:
        return "repeated_char"
    if ascii_letter_ratio(text) >= ascii_threshold:
        return "ascii_heavy"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="input metadata_wts.jsonl")
    parser.add_argument("--out", required=True, help="output metadata.jsonl (cleaned)")
    parser.add_argument("--rejected", default=None,
                        help="output metadata_rejected.jsonl (default: alongside --out)")
    parser.add_argument("--min-chars", type=int, default=3)
    parser.add_argument("--rep-threshold", type=float, default=0.5)
    parser.add_argument("--ascii-threshold", type=float, default=0.3)
    parser.add_argument("--drop-duplicates", action="store_true", default=True)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    rejected = Path(args.rejected) if args.rejected else out.with_name("metadata_rejected.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    seen_texts: dict[str, str] = {}
    kept: list[dict] = []
    rejects: list[dict] = []

    for rec in records:
        text = normalize(rec.get("text", ""))
        rec["text"] = text
        reason = judge(text, args.min_chars, args.rep_threshold, args.ascii_threshold)
        if reason is None and args.drop_duplicates:
            if text in seen_texts:
                reason = f"duplicate_of_{seen_texts[text]}"
            else:
                seen_texts[text] = rec["file_name"]
        if reason is None:
            kept.append(rec)
        else:
            r = dict(rec)
            r["reject_reason"] = reason
            rejects.append(r)

    with out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with rejected.open("w", encoding="utf-8") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"total={len(records)} kept={len(kept)} rejected={len(rejects)}")
    if rejects:
        reason_counts = Counter(r["reject_reason"].split("_of_")[0] for r in rejects)
        for k, v in reason_counts.most_common():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
