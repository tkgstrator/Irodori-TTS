#!/usr/bin/env python3
"""Generate configs/train_500m_v2_<speaker>_lora.yaml from the base template.

Reads two inputs:
  1. configs/train_500m_v2_speaker_lora.yaml  (shared LoRA template)
  2. data/<speaker>/config.yaml               (per-speaker config: name,
     cleaning hints, sample_texts for checkpoint A/B listening)

Writes configs/train_500m_v2_<speaker>_lora.yaml, tuning:
  - train.save_every / train.valid_every so the run emits ~10 checkpoints
    and ~30 val points regardless of dataset size
  - sample_generation.prompts from the per-speaker sample_texts

The W&B run name is NOT written into the per-speaker yaml; it is passed
at launch time by scripts/train/train_multi_speaker.sh via --wandb-run-name.

Usage:
  uv run python scripts/train/make_speaker_config.py <speaker>
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import yaml

TEMPLATE = Path("configs/train_500m_v2_speaker_lora.yaml")

# Targets for how many ckpts / val points a run should emit. These drive
# save_every / valid_every derivation below. Everything else (batch size,
# epochs, valid_ratio) is read from the template so there is no drift.
TARGET_CKPTS = 10
TARGET_VAL_POINTS = 30


def count_rows(path: Path) -> int:
    n = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def compute_schedule(n_rows: int, template_cfg: dict) -> tuple[int, int, int]:
    """Return (save_every, valid_every, total_steps) derived from the template."""
    train_cfg = template_cfg.get("train", {})
    batch_size = int(train_cfg["batch_size"])
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    max_epochs = int(train_cfg["max_epochs"])
    valid_ratio = float(train_cfg.get("valid_ratio", 0.02))
    train_n = n_rows - max(1, round(n_rows * valid_ratio))
    batches_per_epoch = math.ceil(train_n / batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    total_steps = steps_per_epoch * max_epochs
    save_every = max(1, total_steps // TARGET_CKPTS)
    valid_every = max(1, total_steps // TARGET_VAL_POINTS)
    return save_every, valid_every, total_steps


def load_speaker_config(speaker: str) -> dict:
    path = Path(f"data/{speaker}/config.yaml")
    if not path.is_file():
        raise SystemExit(f"speaker config not found: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not cfg.get("sample_texts"):
        raise SystemExit(f"{path}: sample_texts is empty")
    return cfg


def build_prompts_block(sample_texts: list[str]) -> str:
    lines = []
    for i, text in enumerate(sample_texts):
        lines.append(f"    - name: sample_{i:02d}")
        lines.append(f"      text: {text}")
        lines.append("      no_ref: true")
        lines.append("      seconds: 30.0")
        lines.append("      seed: 42")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker", help="short speaker id (e.g. margo, leia)")
    ap.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest.jsonl for step derivation. Defaults to data/<speaker>/manifest.jsonl.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing config.")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"template not found: {TEMPLATE}")

    dst = Path(f"configs/train_500m_v2_{args.speaker}_lora.yaml")
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} already exists (use --force to overwrite)")

    manifest_path = Path(args.manifest) if args.manifest else Path(f"data/{args.speaker}/manifest.jsonl")
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    n_rows = count_rows(manifest_path)

    src_text = TEMPLATE.read_text(encoding="utf-8")
    template_cfg = yaml.safe_load(src_text) or {}
    save_every, valid_every, total_steps = compute_schedule(n_rows, template_cfg)

    speaker_cfg = load_speaker_config(args.speaker)
    sample_texts = speaker_cfg["sample_texts"]
    out_text = re.sub(r"^(\s*save_every:\s*)\d+", rf"\g<1>{save_every}", src_text, count=1, flags=re.MULTILINE)
    out_text = re.sub(r"^(\s*valid_every:\s*)\d+", rf"\g<1>{valid_every}", out_text, count=1, flags=re.MULTILINE)

    prompts_block = build_prompts_block(sample_texts)
    out_text, n_sub = re.subn(
        r"^(\s*)prompts:\s*\[\]\s*$",
        lambda m: f"{m.group(1)}prompts:\n{prompts_block}",
        out_text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_sub != 1:
        raise SystemExit(f"template {TEMPLATE} is missing a `prompts: []` placeholder")

    dst.write_text(out_text, encoding="utf-8")
    print(
        f"wrote {dst} (manifest={n_rows} total_steps~{total_steps} "
        f"save_every={save_every} valid_every={valid_every} "
        f"sample_texts={len(sample_texts)})"
    )


if __name__ == "__main__":
    main()
