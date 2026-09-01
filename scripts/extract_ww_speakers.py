#!/usr/bin/env python3
"""Extract per-character WAVs + transcriptions from the ultemica/wuthering-waves-voices
ja parquet shards into `data/_raw_<speaker>/`.

Layout produced per speaker:
    data/_raw_<speaker>/wavs/000000.wav
    data/_raw_<speaker>/metadata.jsonl   # { file_name, text, speaker_zh, source_path }
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path("data")
SHARDS = sorted(Path("data/wuthering-waves-voices/data/ja").glob("train-*.parquet"))

TARGETS: dict[str, list[str]] = {
    "carlotta": ["珂莱塔"],
    "yangyang": ["秧秧"],
    "cantarella": ["坎特蕾拉"],
    "encore": ["安可"],
    "changli": ["长离"],
    "chixia": ["炽霞"],
    "zhezhi": ["折枝"],
    "roccia": ["洛可可"],
    "shorekeeper": ["守岸人"],
    "phoebe": ["菲比"],
    "yinlin": ["吟霖"],
    "jinhsi": ["今汐"],
    "camellya": ["椿"],
    "cartethyia": ["卡提希娅"],
}

speaker_to_folder: dict[str, str] = {}
for folder, names in TARGETS.items():
    for n in names:
        speaker_to_folder[n] = folder

for folder in TARGETS:
    (ROOT / f"_raw_{folder}" / "wavs").mkdir(parents=True, exist_ok=True)

counters: dict[str, int] = dict.fromkeys(TARGETS, 0)
meta_fps = {
    folder: (ROOT / f"_raw_{folder}" / "metadata.jsonl").open("w", encoding="utf-8")
    for folder in TARGETS
}

try:
    for shard in tqdm(SHARDS, desc="shards"):
        table = pq.read_table(shard, columns=["audio", "transcription", "speaker", "source_path"])
        speakers = table.column("speaker").to_pylist()
        audios = table.column("audio").to_pylist()
        texts = table.column("transcription").to_pylist()
        sources = table.column("source_path").to_pylist()
        for sp, au, tx, src in zip(speakers, audios, texts, sources, strict=True):
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
    print(f"  {folder:12s}  {n:6d} clips")
