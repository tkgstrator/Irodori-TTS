#!/usr/bin/env python3
"""One-shot inference with a base checkpoint + a PEFT LoRA adapter directory.

Usage:
    uv run python scripts/lora/infer_with_adapter.py \
        --base models/Irodori-TTS-500M-v2/model.safetensors \
        --adapter outputs/ayaka_lora/checkpoint_best_val_loss_0001920_0.725963 \
        --text "..." \
        --output out.wav \
        --no-ref \
        --num-steps 40 \
        --cfg-scale-text 3.0 \
        --cfg-scale-speaker 5.0 \
        --seed 42

Mirrors the parameter defaults documented by the upstream Aratako/Irodori-TTS
repo. Provided as a separate entrypoint from `infer.py` because the CLI in
infer.py only takes a single merged checkpoint, not a base + adapter pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model.safetensors path.")
    ap.add_argument(
        "--adapter",
        required=True,
        help="PEFT adapter directory (containing adapter_model.safetensors).",
    )
    ap.add_argument("--adapter-name", default="lora")
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", default="out.wav")
    ap.add_argument("--model-device", default="cuda")
    ap.add_argument("--codec-device", default="cuda")
    ap.add_argument("--model-precision", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--codec-precision", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    ap.add_argument("--no-ref", action="store_true", default=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--num-steps", type=int, default=40)
    ap.add_argument("--cfg-scale-text", type=float, default=3.0)
    ap.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    ap.add_argument("--cfg-guidance-mode", default="independent")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    key = RuntimeKey(
        checkpoint=str(Path(args.base).resolve()),
        model_device=str(args.model_device),
        codec_repo=str(args.codec_repo),
        model_precision=str(args.model_precision),
        codec_device=str(args.codec_device),
        codec_precision=str(args.codec_precision),
        codec_deterministic_encode=True,
        codec_deterministic_decode=True,
        enable_watermark=False,
        compile_model=False,
        compile_dynamic=False,
    )

    runtime = InferenceRuntime.from_base_with_adapters(
        key,
        adapters={args.adapter_name: str(Path(args.adapter).resolve())},
        default_adapter=args.adapter_name,
    )

    req = SamplingRequest(
        text=str(args.text),
        no_ref=bool(args.no_ref),
        num_candidates=1,
        seconds=float(args.seconds),
        num_steps=int(args.num_steps),
        cfg_scale_text=float(args.cfg_scale_text),
        cfg_scale_caption=float(args.cfg_scale_text),
        cfg_scale_speaker=float(args.cfg_scale_speaker),
        cfg_guidance_mode=str(args.cfg_guidance_mode),
        seed=int(args.seed),
    )
    result = runtime.synthesize(req)
    out = save_wav(args.output, result.audio, result.sample_rate)
    print(f"Saved: {out} (sample_rate={result.sample_rate}, seed={result.used_seed})")


if __name__ == "__main__":
    main()
