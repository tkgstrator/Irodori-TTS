#!/usr/bin/env python3
"""Transcribe a flat directory of audio with faster-whisper + word-level punctuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import subprocess

from faster_whisper import WhisperModel
from tqdm import tqdm


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())

TRAILING_PUNCT = set("、。,.!！?？…")


def build_text(words: list, short_gap: float, long_gap: float) -> str:
    out: list[str] = []
    prev_end: float | None = None
    for w in words:
        token = w.word
        if token is None:
            continue
        token = token.lstrip()
        if not token:
            continue
        if prev_end is not None:
            gap = max(0.0, w.start - prev_end)
            last_char = out[-1][-1] if out and out[-1] else ""
            if last_char not in TRAILING_PUNCT:
                if gap >= long_gap:
                    out.append("。")
                elif gap >= short_gap:
                    out.append("、")
        out.append(token)
        prev_end = w.end
    text = "".join(out).strip()
    if text and text[-1] not in TRAILING_PUNCT:
        text += "。"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output", default=None,
                        help="Default: <audio-dir>/metadata_wts.jsonl")
    parser.add_argument("--glob", default="*.ogg")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--short-gap", type=float, default=0.15)
    parser.add_argument("--long-gap", type=float, default=0.40)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output = Path(args.output) if args.output else audio_dir / "metadata_wts.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(audio_dir.glob(args.glob))
    print(f"Found {len(files)} files in {audio_dir}")

    done: set[str] = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                done.add(json.loads(line)["file_name"])
        print(f"Resuming: {len(done)} already transcribed")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    with output.open("a", encoding="utf-8") as f:
        for wav in tqdm(files, desc="transcribe"):
            name = wav.name
            if name in done:
                continue
            segments, _ = model.transcribe(
                str(wav),
                language=args.language,
                beam_size=args.beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=True,
            )
            words = []
            for seg in segments:
                if seg.words:
                    words.extend(seg.words)
            text = build_text(words, args.short_gap, args.long_gap)
            duration = round(probe_duration(wav), 3)
            rec = {"file_name": name, "text": text, "duration": duration}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
