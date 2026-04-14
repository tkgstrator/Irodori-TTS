#!/usr/bin/env python3
"""Synthesize per-checkpoint audio samples after training and ship to W&B.

Used to back-fill samples for runs where sample_generation was not enabled
during training. Resumes the existing W&B run and logs audio under keys
``samples_post/<ckpt_name>/<prompt_name>`` without a step (so the post-hoc
samples appear next to the live training metrics in the same UI).

Usage:
    uv run python scripts/samples/upload_post_samples.py \\
        --output-dir outputs/ema_lora \\
        --base-checkpoint models/Irodori-TTS-500M-v2/model.safetensors \\
        --config configs/train_500m_v2_ema_lora.yaml \\
        --wandb-project irodori-tts-speaker-lora \\
        --wandb-run-id 63k8w8ee
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from irodori_tts.config import load_experiment_yaml, merge_sample_generation_overrides
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)


def discover_checkpoints(output_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[int, str, Path]] = []
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("checkpoint"):
            continue
        if child.name == "checkpoint_final":
            # checkpoint_final duplicates the last periodic save; skip.
            continue
        if child.name.startswith("checkpoint_best_val_loss_"):
            parts = child.name.split("_")
            step = int(parts[4])
            label = f"best_step_{step:07d}_loss_{parts[5]}"
            items.append((step, label, child))
        else:
            try:
                step = int(child.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            items.append((step, f"step_{step:07d}", child))
    items.sort(key=lambda t: (t[0], t[1]))
    return [(label, path) for _, label, path in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--config", required=True, help="Training YAML with sample_generation section")
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--codec-device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Skip W&B; only write local wavs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    base_ckpt = Path(args.base_checkpoint).resolve()
    if not base_ckpt.exists():
        raise FileNotFoundError(base_ckpt)

    exp_cfg = load_experiment_yaml(args.config)
    sample_cfg = merge_sample_generation_overrides(exp_cfg.get("sample_generation"))
    if not sample_cfg.prompts:
        raise ValueError(f"{args.config} has no sample_generation.prompts to synthesize.")

    ckpts = discover_checkpoints(output_dir)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {output_dir}")
    print(f"Found {len(ckpts)} checkpoints, {len(sample_cfg.prompts)} prompts")
    for label, path in ckpts:
        print(f"  {label} -> {path.name}")

    wandb_run = None
    if not args.dry_run:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_run_id,
            resume="must",
        )
        print(f"Resumed W&B run: {wandb_run.name} ({wandb_run.id})")

    samples_root = output_dir / "samples_post"
    samples_root.mkdir(parents=True, exist_ok=True)

    for label, ckpt_path in ckpts:
        print(f"\n=== {label} ===")
        runtime = InferenceRuntime.from_base_with_adapters(
            key=RuntimeKey(
                checkpoint=str(base_ckpt),
                model_device=args.device,
                codec_device=args.codec_device,
            ),
            adapters={"lora": str(ckpt_path)},
        )
        local_dir = samples_root / label
        local_dir.mkdir(parents=True, exist_ok=True)

        log_payload: dict = {}
        for prompt in sample_cfg.prompts:
            req = SamplingRequest(
                text=prompt.text,
                caption=prompt.caption,
                ref_wav=prompt.ref_wav,
                no_ref=bool(prompt.no_ref),
                num_candidates=1,
                seconds=float(prompt.seconds),
                num_steps=int(sample_cfg.num_steps),
                cfg_scale_text=float(sample_cfg.cfg_scale_text),
                cfg_scale_caption=float(sample_cfg.cfg_scale_caption),
                cfg_scale_speaker=float(sample_cfg.cfg_scale_speaker),
                cfg_guidance_mode=str(sample_cfg.cfg_guidance_mode),
                seed=prompt.seed,
            )
            result = runtime.synthesize(req)
            audio = result.audio.detach().to(torch.float32).cpu()
            sr = int(result.sample_rate)
            wav_path = local_dir / f"{prompt.name}.wav"
            save_wav(wav_path, audio, sample_rate=sr)
            print(f"  {prompt.name}: {wav_path}  ({result.total_to_decode:.2f}s)")
            if wandb_run is not None:
                import wandb

                log_payload[f"samples_post/{label}/{prompt.name}"] = wandb.Audio(
                    audio.squeeze(0).numpy(),
                    sample_rate=sr,
                    caption=label,
                )
        if wandb_run is not None and log_payload:
            wandb_run.log(log_payload)

        runtime.unload()
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if wandb_run is not None:
        wandb_run.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
