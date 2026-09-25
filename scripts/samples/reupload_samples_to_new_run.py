#!/usr/bin/env python3
"""Re-upload existing per-checkpoint wavs to a fresh metrics run with stepped logging.

Reads wavs from ``<samples_dir>/<label>/<prompt>.wav`` (produced by
upload_post_samples.py) and logs them into a new run using the same key per
prompt, with the checkpoint's true training step as the log step. This gives
the metrics backend's media panel a single audio widget per prompt with a
step slider, the way p1atdev's character run does it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from irodori_tts.metrics import create_metrics_logger

LABEL_RE = re.compile(r"^(?:best_)?step_(\d+)(?:_loss_(\d+\.\d+))?$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--metrics-backend", default="atmos", help="Metrics logging backend.")
    parser.add_argument("--metrics-project", required=True)
    parser.add_argument("--metrics-run-name", required=True)
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

    metrics_logger = create_metrics_logger(
        args.metrics_backend,
        project=args.metrics_project,
        run_name=args.metrics_run_name,
        run_id=None,
        enabled=True,
    )
    print(f"Created run: {metrics_logger.name}")

    for step, label, loss, ckpt_dir in entries:
        for wav_path in sorted(ckpt_dir.glob("*.wav")):
            # log_audio() takes a file path directly, so the wav already on
            # disk is uploaded as-is (no need to round-trip it through
            # soundfile first).
            metrics_logger.log_audio(f"samples/{wav_path.stem}", wav_path, step=step)
        metrics: dict[str, float] = {"samples/is_best": 1.0 if loss is not None else 0.0}
        if loss is not None:
            metrics["samples/val_loss"] = float(loss)
        metrics_logger.log(metrics, step=step)
        print(f"  logged step={step} label={label}")

    metrics_logger.finish()
    print("Done.")


if __name__ == "__main__":
    main()
