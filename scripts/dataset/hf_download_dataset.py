#!/usr/bin/env python3
"""Download per-speaker tar.gz archives from a HF dataset repo and extract.

Remote layout (as uploaded by scripts/dataset/hf_upload_archives.py):
    archives/<speaker>.tar.gz

Local layout after extraction:
    data/<speaker>/manifest.jsonl
    data/<speaker>/latents/*.pt

Already-present speakers (manifest.jsonl + latents/ exist) are skipped. The
single-archive-per-speaker layout means one HF xet-read-token call per
speaker, avoiding the 1000 req / 5 min rate limit that hits us when we fetch
thousands of individual .pt files.

Usage:
    uv run python scripts/dataset/hf_download_dataset.py \\
        --repo-id ultemica/irodori-tts-voices \\
        [--speakers margo,leia]
"""
from __future__ import annotations

import argparse
import os
import tarfile
import time
from pathlib import Path


def _is_speaker_present(speaker_dir: Path) -> bool:
    return (speaker_dir / "manifest.jsonl").is_file() and (speaker_dir / "latents").is_dir()


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            # Guard against path traversal.
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"{archive}: unsafe path {name!r}")
        tar.extractall(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--speakers", default=None, help="Comma-separated speaker names to fetch.")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--data-root", default="data")
    ap.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("HF_DOWNLOAD_MAX_RETRIES", "10")),
    )
    ap.add_argument(
        "--keep-archives",
        action="store_true",
        help="Do not remove the downloaded .tar.gz after extraction.",
    )
    args = ap.parse_args()

    target = Path(args.data_root)
    target.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    remote_files = api.list_repo_files(
        repo_id=args.repo_id, repo_type="dataset", revision=args.revision
    )
    remote_archives = {
        Path(f).stem.removesuffix(".tar"): f
        for f in remote_files
        if f.startswith("archives/") and f.endswith(".tar.gz")
    }
    if not remote_archives:
        raise SystemExit(f"{args.repo_id}: no archives/*.tar.gz found")

    if args.speakers:
        requested = [s.strip() for s in args.speakers.split(",") if s.strip()]
        missing_remote = [s for s in requested if s not in remote_archives]
        if missing_remote:
            raise SystemExit(f"not on HF: {missing_remote}")
        wanted = requested
    else:
        wanted = sorted(remote_archives.keys())

    to_fetch = [s for s in wanted if not _is_speaker_present(target / s)]
    skipped = len(wanted) - len(to_fetch)
    print(f"[hf_download] wanted={len(wanted)} skip-existing={skipped} to-fetch={len(to_fetch)}")

    for speaker in to_fetch:
        path_in_repo = remote_archives[speaker]
        attempt = 0
        while True:
            attempt += 1
            try:
                local_archive = hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    revision=args.revision,
                    filename=path_in_repo,
                    token=token,
                )
                break
            except HfHubHTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 429 and attempt < args.max_retries:
                    backoff = min(300, 30 * attempt)
                    print(f"[hf_download] {speaker}: 429 (attempt {attempt}); sleeping {backoff}s")
                    time.sleep(backoff)
                    continue
                raise

        speaker_dir = target / speaker
        if speaker_dir.exists():
            # Partial state — wipe before extracting to avoid stale files.
            import shutil
            shutil.rmtree(speaker_dir)
        print(f"[hf_download] extracting {speaker}")
        _extract(Path(local_archive), speaker_dir)

        if not args.keep_archives:
            try:
                Path(local_archive).unlink()
            except OSError:
                pass

    print(f"[hf_download] done; data under {target}")


if __name__ == "__main__":
    main()
