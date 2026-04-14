#!/usr/bin/env python3
"""Upload a preprocessed speaker dataset to HuggingFace Hub (private dataset repo).

Uploads data/<speaker>/ with the build artifacts (_src, _tmp, llm_batches,
punct_batches, *.log) excluded. Reads HF_TOKEN from env.

Usage:
  uv run python scripts/dataset/hf_upload_speaker.py <speaker> \\
    --repo-id ultemica/irodori-tts-<speaker> \\
    [--private] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_IGNORE = [
    "_src/*",
    "_tmp/*",
    "_*_raw/*",
    "llm_batches/*",
    "llm_diffs/*",
    "punct_batches/*",
    "punct_diffs/*",
    "*.log",
    "metadata_rejected.jsonl",
    "metadata_wts.jsonl",
    "metadata_filtered.jsonl",
    "metadata_llm_diff.jsonl",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker")
    ap.add_argument("--repo-id", required=True, help="e.g. ultemica/irodori-tts-sherry")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--include-wavs",
        action="store_true",
        help="Include raw wavs/. Off by default — usually only manifest + latents are needed on the GPU server.",
    )
    args = ap.parse_args()

    folder = Path(f"data/{args.speaker}")
    if not folder.exists():
        raise SystemExit(f"missing folder: {folder}")

    ignore = list(DEFAULT_IGNORE)
    if not args.include_wavs:
        ignore.append("wavs/*")

    print(f"repo: {args.repo_id} (private={args.private})")
    print(f"folder: {folder}")
    print(f"ignore: {ignore}")
    if args.dry_run:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                rel = p.relative_to(folder).as_posix()
                skip = any(_match(rel, pat) for pat in ignore)
                if not skip:
                    print(f"  + {rel}")
        return

    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (put it in .env)")

    create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=token)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
        ignore_patterns=ignore,
        commit_message=f"upload {args.speaker} dataset",
    )
    print("done.")


def _match(rel: str, pat: str) -> bool:
    from fnmatch import fnmatch

    return fnmatch(rel, pat)


if __name__ == "__main__":
    main()
