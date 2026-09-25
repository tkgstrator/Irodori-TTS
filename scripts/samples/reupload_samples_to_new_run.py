#!/usr/bin/env python3
"""Re-upload existing per-checkpoint wavs to a fresh atmos job with stepped logging.

Reads wavs from ``<samples_dir>/<label>/<prompt>.wav`` (produced by
upload_post_samples.py) and logs them into a new job using the same key per
prompt, with the checkpoint's true training step as the log step. This gives
atmos's media panel a single audio widget per prompt with a step slider, the
way p1atdev's character run does it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

LABEL_RE = re.compile(r"^(?:best_)?step_(\d+)(?:_loss_(\d+\.\d+))?$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--atmos-project", required=True)
    parser.add_argument("--atmos-run-name", required=True)
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

    import atmos

    run = atmos.init(args.atmos_project, name=args.atmos_run_name)
    print(f"Created atmos job: {run.job_id}")

    for step, label, loss, ckpt_dir in entries:
        for wav_path in sorted(ckpt_dir.glob("*.wav")):
            # atmos.log_audio() takes a file path directly, so the wav
            # already on disk is uploaded as-is (no need to round-trip it
            # through soundfile first).
            run.log_audio(f"samples/{wav_path.stem}", wav_path, step)
        metrics: dict[str, float] = {"samples/is_best": 1.0 if loss is not None else 0.0}
        if loss is not None:
            metrics["samples/val_loss"] = float(loss)
        run.log(metrics, step)
        print(f"  logged step={step} label={label}")

    run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
