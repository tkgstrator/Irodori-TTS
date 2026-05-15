#!/usr/bin/env python3
"""Extract per-character WAVs + transcriptions from the ultemica/honkai-star-rail-voices
ja parquet shards into `data/_raw_<speaker>/`.

Layout produced per speaker:
    data/_raw_<speaker>/wavs/000000.wav
    data/_raw_<speaker>/metadata.jsonl   # { file_name, text, speaker_zh, source_path }
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path("data")
SHARDS = sorted(glob.glob("data/honkai-star-rail-voices/data/ja/train-*.parquet"))

TARGETS: dict[str, list[str]] = {
    "cyrene":  ["昔涟", "往昔的涟漪"],  # 井上麻里奈
    "yanqing": ["彦卿"],                # 井上麻里奈
    "hanya":   ["寒鸦"],                # 鈴代紗弓
    "bailu":   ["白露"],                # 加藤英美里
    "sparkle": ["花火"],                # 上田麗奈
    "clara":   ["克拉拉"],              # 日高里菜
    "seele":   ["希儿"],                # 中原麻衣
    "robin":   ["知更鸟"],              # 名塚佳織
    "fuxuan":  ["符玄"],                # 伊藤美来
}

speaker_to_folder: dict[str, str] = {}
for folder, names in TARGETS.items():
    for n in names:
        speaker_to_folder[n] = folder

for folder in TARGETS:
    (ROOT / f"_raw_{folder}" / "wavs").mkdir(parents=True, exist_ok=True)

counters: dict[str, int] = {f: 0 for f in TARGETS}
meta_fps = {
    folder: (ROOT / f"_raw_{folder}" / "metadata.jsonl").open("w", encoding="utf-8")
    for folder in TARGETS
}

try:
    for shard in tqdm(SHARDS, desc="shards"):
        table = pq.read_table(
            shard, columns=["audio", "transcription", "speaker", "source_path"]
        )
        speakers = table.column("speaker").to_pylist()
        audios = table.column("audio").to_pylist()
        texts = table.column("transcription").to_pylist()
        sources = table.column("source_path").to_pylist()
        for sp, au, tx, src in zip(speakers, audios, texts, sources):
            folder = speaker_to_folder.get(sp)
            if folder is None:
                continue
            idx = counters[folder]
            counters[folder] += 1
            name = f"{idx:06d}.wav"
            (ROOT / f"_raw_{folder}" / "wavs" / name).write_bytes(au["bytes"])
            rec = {
                "file_name": name,
                "text": tx or "",
                "speaker_zh": sp,
                "source_path": src,
            }
            meta_fps[folder].write(json.dumps(rec, ensure_ascii=False) + "\n")
finally:
    for fp in meta_fps.values():
        fp.close()

print("\nDone:")
for folder, n in counters.items():
    print(f"  {folder:8s}  {n:6d} clips")
