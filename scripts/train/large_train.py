#!/usr/bin/env python3
"""Run prepared v4-Large speaker LoRAs across independent cluster GPUs.

Run this inside a training container with the prepared source data mounted at
exactly the absolute paths recorded in the targets file. Share --lock-root and
--output-root between nodes, but never share a Python environment volume.

Example (after `uv sync --frozen --no-dev --extra atmos` in the container)::

    python scripts/train/large_train.py --targets /app/large-targets.json \
        --gpus "0 1" --config configs/train_v4_large_lora.yaml \
        --base models/Irodori-TTS-v4-Large/model.safetensors \
        --output-root outputs_v4_large --lock-root locks/v4_large \
        --exclude vtuber_cherry

Claims are never removed automatically. Inspect failed.json and the output
before manually releasing a failed or abandoned claim for an explicit retry.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SPEAKER_ID = re.compile(r"^(?:gi|hsr|wuwa|mgwt|vtuber)_[a-z0-9_]+$")
_CHECKPOINT = re.compile(r"^checkpoint_([0-9]+)$")
_REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "config.json",
    "manifest_size.txt",
    "trainer_state.pt",
)


@dataclass(frozen=True)
class Target:
    id: str
    category: str
    manifest: Path
    clips: int


@dataclass(frozen=True)
class TrainSettings:
    config: Path
    base: Path
    output_root: Path
    lock_root: Path
    project: str
    cwd: Path


def parse_gpus(value: str) -> list[int]:
    parts = value.replace(",", " ").split()
    if not parts or any(not part.isdecimal() for part in parts):
        raise ValueError("--gpus must contain one or more nonnegative GPU indices")
    gpus = [int(part) for part in parts]
    if len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must not repeat a GPU index")
    return gpus


def load_targets(path: Path) -> list[Target]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("targets must be a nonempty list or an object with nonempty entries")
    targets: list[Target] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("each target must be an object")
        speaker_id = row.get("id")
        category = row.get("category")
        manifest_path = row.get("manifest_path")
        clips = row.get("clips")
        if not isinstance(speaker_id, str) or _SPEAKER_ID.fullmatch(speaker_id) is None:
            raise ValueError(f"invalid canonical speaker id: {speaker_id!r}")
        if speaker_id in seen:
            raise ValueError(f"duplicate speaker id: {speaker_id}")
        if not isinstance(category, str) or not category:
            raise ValueError(f"invalid category for {speaker_id}")
        if not isinstance(manifest_path, str) or not Path(manifest_path).is_absolute():
            raise ValueError(f"manifest_path must be absolute for {speaker_id}")
        if isinstance(clips, bool) or not isinstance(clips, int) or clips <= 0:
            raise ValueError(f"clips must be a positive integer for {speaker_id}")
        seen.add(speaker_id)
        targets.append(Target(speaker_id, category, Path(manifest_path), clips))
    return sorted(targets, key=lambda target: (-target.clips, target.id))


def validate_target(target: Target) -> None:
    if not target.manifest.is_file():
        raise ValueError(f"missing manifest for {target.id}: {target.manifest}")
    with target.manifest.open(encoding="utf-8") as handle:
        actual = sum(1 for _ in handle)
    if actual != target.clips:
        raise ValueError(f"{target.id}: manifest has {actual} rows, expected {target.clips}")
    if not (target.manifest.parent / "latents").is_dir():
        raise ValueError(f"{target.id}: latents directory is missing beside manifest")


def latest_valid_checkpoint(output_dir: Path, manifest_size: int) -> Path | None:
    if not output_dir.is_dir():
        return None
    numeric = [
        child
        for child in output_dir.iterdir()
        if child.is_dir()
        and (match := _CHECKPOINT.fullmatch(child.name)) is not None
        and int(match.group(1)) > 0
    ]
    valid: list[Path] = []
    for checkpoint in numeric:
        if any(not (checkpoint / name).is_file() for name in _REQUIRED_CHECKPOINT_FILES):
            continue
        try:
            stored = int((checkpoint / "manifest_size.txt").read_text(encoding="utf-8").strip())
        except ValueError:
            continue
        if stored == manifest_size and all(
            (checkpoint / name).stat().st_size > 0 for name in _REQUIRED_CHECKPOINT_FILES
        ):
            valid.append(checkpoint)
    if numeric and not valid:
        raise ValueError(
            f"no complete checkpoint matches manifest size {manifest_size}: {output_dir}"
        )
    return (
        max(valid, key=lambda item: int(item.name.removeprefix("checkpoint_"))) if valid else None
    )


def build_command(
    target: Target,
    settings: TrainSettings,
    output_dir: Path,
    resume: Path | None,
) -> list[str]:
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "train.py",
        "--config",
        str(settings.config),
        "--manifest",
        str(target.manifest),
        "--output-dir",
        str(output_dir),
        "--init-checkpoint",
        str(settings.base),
        "--metrics-backend",
        "atmos",
        "--metrics-project",
        settings.project,
        "--metrics-run-name",
        f"{target.id}_lora_v4_large",
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_output(target: Target, settings: TrainSettings) -> tuple[Path, Path | None]:
    output_dir = settings.output_root / f"{target.id}_lora"
    if (output_dir / "checkpoint_final").exists():
        raise ValueError(
            "final checkpoint already exists without a done marker; reconcile manually"
        )
    resume = latest_valid_checkpoint(output_dir, target.clips)
    if resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output exists without a resumable checkpoint; reconcile manually")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, resume


def _train_once(target: Target, gpu: int, claim: Path, settings: TrainSettings) -> None:
    output_dir, resume = _prepare_output(target, settings)
    command = build_command(target, settings, output_dir, resume)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    with (output_dir / "train.log").open("a", encoding="utf-8") as log:
        log.write(
            f"=== launch {datetime.now(UTC).isoformat()} gpu={gpu} resume={resume or 'none'} ===\n"
        )
        log.flush()
        result = subprocess.run(
            command, cwd=settings.cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode != 0:
        raise RuntimeError(f"trainer exited with status {result.returncode}")
    final = output_dir / "checkpoint_final"
    if not all(
        (final / name).is_file() and (final / name).stat().st_size > 0
        for name in _REQUIRED_CHECKPOINT_FILES
    ):
        raise RuntimeError("trainer returned success without a complete checkpoint_final")
    _write_status(claim / "done.json", {"id": target.id, "gpu": gpu, "step": "final"})
    print(f"[{gpu}] {target.id}: done", flush=True)


def _run_claimed(target: Target, gpu: int, claim: Path, settings: TrainSettings) -> bool:
    try:
        _train_once(target, gpu, claim, settings)
    except Exception as exc:
        _write_status(claim / "failed.json", {"id": target.id, "gpu": gpu, "error": str(exc)})
        print(f"[{gpu}] {target.id}: failed: {exc}", file=sys.stderr, flush=True)
        return False
    else:
        return True


def run_training(
    targets: list[Target],
    gpus: list[int],
    settings: TrainSettings,
    excluded: set[str] | None = None,
) -> int:
    if not os.environ.get("ATMOS_TOKEN") or not os.environ.get("ATMOS_BASE_URL"):
        raise ValueError(
            "ATMOS_TOKEN and ATMOS_BASE_URL are required before launching any GPU work"
        )
    if not settings.config.is_file() or not settings.base.is_file():
        raise ValueError("Large config and base checkpoint must both exist before GPU work")
    if settings.output_root.resolve() == (settings.cwd / "outputs").resolve():
        raise ValueError("Large outputs must not overwrite the Small outputs directory")
    candidates = [target for target in targets if target.id not in (excluded or set())]
    for target in candidates:
        validate_target(target)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    settings.lock_root.mkdir(parents=True, exist_ok=True)

    def worker(gpu: int) -> int:
        failed = 0
        for target in candidates:
            claim = settings.lock_root / target.id
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            _write_status(
                claim / "claim.json",
                {"id": target.id, "host": socket.gethostname(), "pid": os.getpid(), "gpu": gpu},
            )
            if not _run_claimed(target, gpu, claim, settings):
                failed += 1
        return failed

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        failures = sum(pool.map(worker, gpus))
    unfinished = [
        target.id
        for target in candidates
        if not (settings.lock_root / target.id / "done.json").is_file()
    ]
    if unfinished:
        print(
            f"{len(unfinished)} targets remain unfinished ({failures} failed here): "
            + ", ".join(unfinished[:10]),
            file=sys.stderr,
        )
    return 1 if unfinished else 0


def _validate_excluded(excluded: set[str], targets: list[Target]) -> None:
    unknown = excluded - {target.id for target in targets}
    if unknown:
        raise ValueError(f"unknown excluded speaker IDs: {', '.join(sorted(unknown))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--gpus", default=os.environ.get("GPUS", ""))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--metrics-project", default="irodori-tts-v4-large")
    parser.add_argument(
        "--exclude", default="", help="Comma-separated canonical IDs already running"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        targets = load_targets(args.targets)
        gpus = parse_gpus(args.gpus)
        excluded = {value.strip() for value in args.exclude.split(",") if value.strip()}
        _validate_excluded(excluded, targets)
        candidates = [target for target in targets if target.id not in excluded]
        if args.dry_run:
            for target in candidates:
                validate_target(target)
            print(f"ready: {len(candidates)} speakers on {len(gpus)} GPU workers")
            return
        raise SystemExit(
            run_training(
                targets,
                gpus,
                TrainSettings(
                    config=args.config,
                    base=args.base,
                    output_root=args.output_root,
                    lock_root=args.lock_root,
                    project=args.metrics_project,
                    cwd=Path.cwd(),
                ),
                excluded=excluded,
            )
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
