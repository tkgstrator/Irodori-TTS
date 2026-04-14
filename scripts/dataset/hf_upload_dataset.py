#!/usr/bin/env python3
"""Upload a multi-speaker dataset to a single HuggingFace Hub repo.

Layout in the repo:
    <repo root>/
        speakers/
            <speaker1>/
                manifest.jsonl
                latents/*.pt
            <speaker2>/
                ...

Usage:
    uv run python scripts/dataset/hf_upload_dataset.py \\
        --repo-id ultemica/irodori-tts-voices \\
        [--speakers margo,leia,coco] [--private] [--dry-run]

If --speakers is omitted, every data/<speaker>/ with a manifest.jsonl is
uploaded.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DATA_ROOT = Path("data")

IGNORE = [
    "*/wavs/*",
    "*/_src/*",
    "*/_tmp/*",
    "*/_*_raw/*",
    "*/llm_batches/*",
    "*/llm_diffs/*",
    "*/punct_batches/*",
    "*/punct_diffs/*",
    "*/*.log",
    "*/metadata_rejected.jsonl",
    "*/metadata_wts.jsonl",
    "*/metadata_filtered.jsonl",
    "*/metadata_llm_diff.jsonl",
    "*/metadata.jsonl",
]


def discover_speakers() -> list[str]:
    speakers = []
    for sub in sorted(DATA_ROOT.iterdir()):
        if (sub / "manifest.jsonl").exists() and (sub / "latents").is_dir():
            speakers.append(sub.name)
    return speakers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument(
        "--speakers",
        default=None,
        help="Comma-separated speaker names. If omitted, auto-discover all.",
    )
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.speakers:
        speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    else:
        speakers = discover_speakers()

    if not speakers:
        raise SystemExit("no speakers found under data/")

    allow = []
    for s in speakers:
        if not (DATA_ROOT / s / "manifest.jsonl").exists():
            raise SystemExit(f"missing data/{s}/manifest.jsonl")
        allow.append(f"{s}/manifest.jsonl")
        allow.append(f"{s}/latents/*")

    print(f"repo:     {args.repo_id} (private={args.private})")
    print(f"speakers: {speakers}")
    print(f"allow:    {allow}")
    print(f"ignore:   {IGNORE}")

    if args.dry_run:
        from fnmatch import fnmatch

        for s in speakers:
            for p in sorted((DATA_ROOT / s).rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(DATA_ROOT).as_posix()
                if any(fnmatch(rel, pat) for pat in IGNORE):
                    continue
                if not any(fnmatch(rel, pat) for pat in allow):
                    continue
                print(f"  + {rel}")
        return

    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (put it in .env)")

    create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=token)
    api = HfApi(token=token)

    # Clean up legacy root-level layout (<speaker>/...) so the repo only keeps
    # the new speakers/<speaker>/ tree.
    try:
        existing = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    except Exception:
        existing = []
    stale = [
        f for f in existing
        if not f.startswith("speakers/") and not f.startswith(".") and "/" in f
    ]
    if stale:
        print(f"deleting {len(stale)} legacy file(s) at repo root")
        api.delete_files(
            repo_id=args.repo_id,
            repo_type="dataset",
            delete_patterns=stale,
            commit_message="cleanup legacy root layout",
        )

    api.upload_folder(
        folder_path=str(DATA_ROOT),
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo="speakers",
        allow_patterns=allow,
        ignore_patterns=IGNORE,
        commit_message=f"upload {len(speakers)} speaker(s): {', '.join(speakers)}",
    )
    print("done.")


if __name__ == "__main__":
    main()
