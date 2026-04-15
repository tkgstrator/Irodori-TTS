#!/usr/bin/env python3
"""Upload per-speaker tar.gz archives to a HF dataset repo.

Remote layout after upload:
    speakers/<speaker>.tar.gz

Usage:
    uv run python scripts/dataset/hf_upload_archives.py \
        --repo-id <org>/irodori-tts-voices \
        --archive-dir data/_archives
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--archive-dir", default="data/_archives")
    ap.add_argument(
        "--commit-message",
        default="chore: upload per-speaker tar.gz archives",
    )
    args = ap.parse_args()

    archive_dir = Path(args.archive_dir)
    archives = sorted(archive_dir.glob("*.tar.gz"))
    if not archives:
        raise SystemExit(f"no archives found in {archive_dir}")

    from huggingface_hub import CommitOperationAdd, HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    operations: list = []

    for arc in archives:
        path_in_repo = f"speakers/{arc.name}"
        size_mb = arc.stat().st_size / (1024 * 1024)
        print(f"adding {path_in_repo} ({size_mb:.1f} MB)")
        operations.append(
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(arc))
        )

    print(f"committing {len(operations)} operations to {args.repo_id}")
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=args.commit_message,
    )
    print("done")


if __name__ == "__main__":
    main()
