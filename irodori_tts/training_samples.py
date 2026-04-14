"""Periodic audio sample generation during training, with optional W&B logging."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .codec import DACVAECodec
from .config import ModelConfig, SampleGenerationConfig, TrainConfig
from .inference_runtime import (
    InferenceRuntime,
    SamplingRequest,
    resolve_runtime_dtype,
    save_wav,
)

if TYPE_CHECKING:
    from .tokenizer import PretrainedTextTokenizer


def load_codec_for_sampling(
    sample_cfg: SampleGenerationConfig,
    *,
    expected_latent_dim: int,
) -> DACVAECodec:
    codec_device = torch.device(sample_cfg.codec_device)
    codec_dtype = resolve_runtime_dtype(
        precision=sample_cfg.codec_precision,
        device=codec_device,
    )
    codec = DACVAECodec.load(
        repo_id=sample_cfg.codec_repo,
        device=str(codec_device),
        dtype=codec_dtype,
        deterministic_encode=True,
        deterministic_decode=True,
        enable_watermark=False,
    )
    if codec.latent_dim != expected_latent_dim:
        raise ValueError(
            f"Codec latent_dim={codec.latent_dim} does not match model latent_dim={expected_latent_dim}."
        )
    return codec


def generate_training_samples(
    *,
    raw_model: torch.nn.Module,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    sample_cfg: SampleGenerationConfig,
    tokenizer: "PretrainedTextTokenizer",
    caption_tokenizer: "PretrainedTextTokenizer | None",
    codec: DACVAECodec,
    model_device: torch.device,
    step: int,
    output_dir: Path,
    wandb_run: Any | None,
    log_fn: Any | None = None,
) -> None:
    """Synthesize the configured prompts and ship to W&B / disk.

    The model is left in eval() during synthesis and restored to train() on exit.
    Caller is responsible for calling this only on rank0.
    """
    if not sample_cfg.enabled or not sample_cfg.prompts:
        return

    was_training = raw_model.training
    raw_model.eval()
    try:
        runtime = InferenceRuntime.from_components(
            model=raw_model,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=codec,
            model_device=str(model_device),
            codec_device=str(codec.device),
            max_text_len=train_cfg.max_text_len,
            max_caption_len=train_cfg.max_caption_len,
        )

        log_payload: dict[str, Any] = {}
        wandb_module = None
        if wandb_run is not None:
            import wandb as wandb_module  # type: ignore

        sample_dir = output_dir / "samples" / f"step_{step:07d}"
        if sample_cfg.save_local:
            sample_dir.mkdir(parents=True, exist_ok=True)

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
            try:
                result = runtime.synthesize(req, log_fn=log_fn)
            except Exception as exc:
                if log_fn is not None:
                    log_fn(f"[samples] prompt={prompt.name!r} failed: {exc}")
                continue

            audio = result.audio.detach().to(torch.float32).cpu()
            sr = int(result.sample_rate)

            if sample_cfg.save_local:
                save_wav(sample_dir / f"{prompt.name}.wav", audio, sample_rate=sr)

            if wandb_run is not None and wandb_module is not None:
                audio_np = audio.squeeze(0).numpy()
                log_payload[f"samples/{prompt.name}"] = wandb_module.Audio(
                    audio_np,
                    sample_rate=sr,
                    caption=f"step={step}",
                )

        if wandb_run is not None and log_payload:
            wandb_run.log(log_payload, step=step)
    finally:
        if was_training:
            raw_model.train()
