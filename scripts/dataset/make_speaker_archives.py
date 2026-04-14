#!/usr/bin/env python3
"""Build one tar.gz per speaker containing manifest.jsonl + referenced latents.

Output: data/_archives/<speaker>.tar.gz

Each archive root layout:
    manifest.jsonl
    latents/<file>.pt
    latents/<file>.pt
    ...

Only latents referenced from the manifest are included (ema has a subsetted
manifest that points to ~500 of 4681 latents — we ship just those).

Usage:
    uv run python scripts/dataset/make_speaker_archives.py \
        --speakers ema,sherry,hiro,margo,leia,coco,alisa,hanna,meruru,nanoka,miria,noah,yuki,anan,cherry
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def resolve_manifest(speaker_dir: Path) -> Path:
    for name in ("manifest.jsonl", "train_manifest.jsonl"):
        p = speaker_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no manifest in {speaker_dir}")


def build_archive(speaker: str, data_root: Path, out_dir: Path) -> Path:
    speaker_dir = data_root / speaker
    manifest_path = resolve_manifest(speaker_dir)
    latents_dir = speaker_dir / "latents"
    if not latents_dir.is_dir():
        raise FileNotFoundError(f"no latents dir in {speaker_dir}")

    referenced: list[str] = []
    missing: list[str] = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lp = rec.get("latent_path")
            if not lp:
                raise ValueError(f"{manifest_path}: record without latent_path")
            abs_lp = (speaker_dir / lp).resolve()
            if not abs_lp.is_file():
                missing.append(lp)
                continue
            referenced.append(lp)

    if missing:
        raise FileNotFoundError(
            f"{speaker}: {len(missing)} latents referenced but missing, e.g. {missing[:3]}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{speaker}.tar.gz"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with tarfile.open(tmp_path, "w:gz", compresslevel=6) as tar:
        tar.add(manifest_path, arcname="manifest.jsonl")
        for lp in referenced:
            tar.add(speaker_dir / lp, arcname=lp)

    tmp_path.rename(out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[{speaker}] manifest={len(referenced)} size={size_mb:.1f}MB -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", required=True, help="Comma-separated speaker list.")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="data/_archives")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    names = [s.strip() for s in args.speakers.split(",") if s.strip()]

    for speaker in names:
        build_archive(speaker, data_root, out_dir)


if __name__ == "__main__":
    main()
