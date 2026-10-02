#!/usr/bin/env python3
"""Resolve published speaker adapters to already prepared V4 Large training data.

The Hub listing is the target inventory, not a source of training audio. This
command reads existing manifests and latents without modifying either data root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ID = "ultemica/irodori-tts"
SOURCE_PREFIX = "v4.1-small/"
CATEGORY_PREFIXES = {
    "genshin_impact": "gi_",
    "honkai_star_rail": "hsr_",
    "wuthering_waves": "wuwa_",
    "magical_girl_witch_trials": "mgwt_",
    "vtuber": "vtuber_",
}

# Prepared legacy datasets predate the globally unique published speaker IDs.
# No other implicit prefix stripping is allowed: it could pick up the wrong voice.
LEGACY_ALIASES = {
    "gi_ayaka": "ayaka",
    "gi_beidou": "beidou",
    "gi_ganyu": "ganyu",
    "gi_hu_tao": "hu_tao",
    "gi_raiden": "raiden",
    "gi_sayu": "sayu",
    "gi_yae_miko": "yae_miko",
    "hsr_bailu": "bailu",
    "hsr_fuxuan": "fuxuan",
    "hsr_sparkle": "sparkle",
    "wuwa_chixia": "chixia",
    "wuwa_jinhsi": "jinhsi",
    "mgwt_alisa": "alisa",
    "mgwt_anan": "anan",
    "mgwt_coco": "coco",
    "mgwt_ema": "ema",
    "mgwt_hanna": "hanna",
    "mgwt_hiro": "hiro",
    "mgwt_leia": "leia",
    "mgwt_margo": "margo",
    "mgwt_meruru": "meruru",
    "mgwt_miria": "miria",
    "mgwt_nanoka": "nanoka",
    "mgwt_noah": "noah",
    "mgwt_sherry": "sherry",
    "mgwt_yuki": "yuki",
    "vtuber_cherry": "cherry",
    "vtuber_vivi": "vivi",
}

_ID = re.compile(r"[a-z0-9_]+\Z")


def published_targets(paths: list[str]) -> list[tuple[str, str]]:
    """Extract (speaker ID, category), rejecting ambiguous Hub layouts."""
    targets: dict[str, str] = {}
    duplicates: set[str] = set()
    errors: list[str] = []
    for path in paths:
        if not path.startswith(SOURCE_PREFIX) or not path.endswith(".safetensors"):
            continue
        parts = path.removeprefix(SOURCE_PREFIX).split("/")
        if len(parts) != 2:
            errors.append(f"invalid adapter path: {path}")
            continue
        category, filename = parts
        speaker = filename.removesuffix(".safetensors")
        prefix = CATEGORY_PREFIXES.get(category)
        if prefix is None or _ID.fullmatch(speaker) is None or not speaker.startswith(prefix):
            errors.append(f"invalid category or speaker ID: {path}")
            continue
        if speaker in targets:
            duplicates.add(speaker)
        targets[speaker] = category
    if errors or duplicates or not targets:
        raise ValueError(
            f"published inventory invalid: targets={len(targets)} "
            f"duplicate_ids={len(duplicates)} malformed={len(errors)}; "
            f"examples={(sorted(duplicates) + errors)[:10]}"
        )
    return [(speaker, targets[speaker]) for speaker in sorted(targets)]


def _latent_path(record: Any, manifest: Path, line_number: int) -> str:
    if not isinstance(record, dict):
        raise TypeError(f"{manifest}:{line_number}: expected an object")
    raw = record.get("latent_path")
    if not isinstance(raw, str):
        raise TypeError(f"{manifest}:{line_number}: missing latent_path")
    relative = Path(raw)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "latents"
        or relative.suffix != ".pt"
        or ".." in relative.parts
    ):
        raise ValueError(f"{manifest}:{line_number}: unsafe latent_path {raw!r}")
    if not (manifest.parent / relative).is_file():
        raise ValueError(f"{manifest}:{line_number}: missing latent {raw!r}")
    return raw


def _manifest_clips(manifest: Path) -> int:
    latent_dir = manifest.parent / "latents"
    if not latent_dir.is_dir():
        raise ValueError(f"missing latents directory: {latent_dir}")
    referenced: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest}:{line_number}: invalid JSON") from exc
            raw = _latent_path(record, manifest, line_number)
            if raw in referenced:
                raise ValueError(f"{manifest}:{line_number}: duplicate latent_path {raw!r}")
            referenced.add(raw)
    clips = len(referenced)
    if clips == 0:
        raise ValueError(f"empty manifest: {manifest}")
    latent_count = sum(1 for latent in latent_dir.glob("*.pt") if latent.is_file())
    if latent_count != clips:
        raise ValueError(f"{manifest}: manifest={clips} latent_files={latent_count}")
    return clips


def build_inventory(
    paths: list[str], *, v4_data_root: Path, legacy_data_root: Path, revision: str
) -> dict[str, Any]:
    """Validate every published target before returning a deterministic inventory."""
    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    for speaker, category in published_targets(paths):
        direct = v4_data_root / speaker / "manifest.jsonl"
        alias = LEGACY_ALIASES.get(speaker)
        manifest = direct if direct.is_file() else None
        if manifest is None and alias is not None:
            legacy = legacy_data_root / alias / "manifest.jsonl"
            manifest = legacy if legacy.is_file() else None
        if manifest is None:
            problems.append(f"{speaker}: missing manifest")
            continue
        try:
            clips = _manifest_clips(manifest)
        except (TypeError, ValueError) as exc:
            problems.append(f"{speaker}: {exc}")
            continue
        entries.append(
            {
                "id": speaker,
                "category": category,
                "manifest_path": str(manifest.resolve()),
                "clips": clips,
            }
        )
    if problems:
        raise ValueError(
            f"prepared data invalid: targets={len(entries) + len(problems)} "
            f"valid={len(entries)} invalid={len(problems)}; examples={problems[:10]}"
        )
    return {
        "source_repo": REPO_ID,
        "source_revision": revision,
        "source_prefix": SOURCE_PREFIX,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-data-root", required=True, type=Path)
    parser.add_argument("--legacy-data-root", required=True, type=Path)
    parser.add_argument("--revision", default="main", help="Hub revision to resolve and pin")
    parser.add_argument(
        "--output", type=Path, help="JSON inventory path; required unless --dry-run"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is set")

    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.repo_info(REPO_ID, repo_type="model", revision=args.revision)
    revision = info.sha
    if not isinstance(revision, str) or not revision:
        raise SystemExit("Hub did not return a pinned revision")
    paths = api.list_repo_files(REPO_ID, repo_type="model", revision=revision)
    inventory = build_inventory(
        paths,
        v4_data_root=args.v4_data_root,
        legacy_data_root=args.legacy_data_root,
        revision=revision,
    )
    counts = Counter(entry["category"] for entry in inventory["entries"])
    total_clips = sum(entry["clips"] for entry in inventory["entries"])
    print(f"{REPO_ID}@{revision}: {len(inventory['entries'])} speakers, {total_clips} clips")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    if args.dry_run:
        return

    output: Path = args.output
    serialized = json.dumps(inventory, ensure_ascii=False, indent=2) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != serialized:
            raise SystemExit(f"existing inventory differs; refusing to overwrite: {output}")
        print(f"unchanged: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
