#!/usr/bin/env python3
"""Re-upload existing per-checkpoint wavs to a fresh W&B run with stepped logging.

Reads wavs from ``<samples_dir>/<label>/<prompt>.wav`` (produced by
upload_post_samples.py) and logs them into a new run using the same key per
prompt, with the checkpoint's true training step as the log step. This gives
wandb's media panel a single audio widget per prompt with a step slider, the
way p1atdev's character run does it.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import soundfile as sf

LABEL_RE = re.compile(r"^(?:best_)?step_(\d+)(?:_loss_(\d+\.\d+))?$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir).resolve()
    entries: list[tuple[int, str, float | None, Path]] = []
    for child in sorted(samples_dir.iterdir()):
        if not child.is_dir():
            continue
        m = LABEL_RE.match(child.name)
        if not m:
            continue
        step = int(m.group(1))
        loss = float(m.group(2)) if m.group(2) else None
        entries.append((step, child.name, loss, child))
    entries.sort(key=lambda t: (t[0], 0 if t[2] is None else 1))
    if not entries:
        raise RuntimeError(f"No checkpoint subdirs found under {samples_dir}")

    print(f"Found {len(entries)} checkpoints:")
    for step, label, loss, _ in entries:
        tag = f"  best (val_loss={loss:.6f})" if loss is not None else ""
        print(f"  step={step:5d}  {label}{tag}")

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
    )
    print(f"Created W&B run: {run.name} ({run.id})")

    for step, label, loss, ckpt_dir in entries:
        payload: dict = {}
        for wav_path in sorted(ckpt_dir.glob("*.wav")):
            data, sr = sf.read(str(wav_path), dtype="float32")
            payload[f"samples/{wav_path.stem}"] = wandb.Audio(
                data,
                sample_rate=int(sr),
                caption=label,
            )
        if loss is not None:
            payload["samples/is_best"] = 1
            payload["samples/val_loss"] = float(loss)
        else:
            payload["samples/is_best"] = 0
        run.log(payload, step=step)
        print(f"  logged step={step} label={label}")

    run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
