#!/usr/bin/env python3
"""Download a speaker dataset from HuggingFace Hub into data/<speaker>/.

Uses HF_TOKEN for private repos. Run this on the GPU server.

Usage:
  uv run python scripts/dataset/hf_download_speaker.py <speaker> \\
    --repo-id ultemica/irodori-tts-<speaker>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--revision", default=None, help="tag/branch/commit (default: main)")
    args = ap.parse_args()

    target = Path(f"data/{args.speaker}")
    target.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"downloaded to {path}")


if __name__ == "__main__":
    main()
