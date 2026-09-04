"""Command line interface definition for train.py."""

from __future__ import annotations

import argparse

import torch

from irodori_tts.lora import LORA_TARGET_PRESETS

WANDB_MODES = {"online", "offline", "disabled"}
TRAIN_MODES = {"rf", "duration_only"}


def cli_provided(argv: list[str], flag: str) -> bool:
    return any(x == flag or x.startswith(flag + "=") for x in argv)


def _lora_field_cli_explicit(field: str, args: argparse.Namespace, raw_argv: list[str]) -> bool:
    if field == "lora_enabled":
        return args.lora_enabled is not None
    flag = "--" + field.replace("_", "-")
    return cli_provided(raw_argv, flag)


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Train Irodori-TTS.")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config path containing model/train settings.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSONL manifest with text+latent_path (optional speaker_id for reference sampling).",
    )
    parser.add_argument("--output-dir", default="outputs/irodori_tts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="bf16",
        help=(
            "Compute precision for model forward pass. "
            "Model weights and optimizer states remain FP32."
        ),
    )
    parser.add_argument(
        "--tf32",
        dest="allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable TF32 matmul/cuDNN kernels on CUDA for speed.",
    )
    parser.add_argument(
        "--compile-model",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable torch.compile for the training model.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable activation checkpointing on diffusion blocks to reduce memory.",
    )
    parser.add_argument(
        "--train-mode",
        choices=sorted(TRAIN_MODES),
        default=None,
        help="Training objective: rf runs DiT/RF training; duration_only trains only the duration predictor.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume full training state from a training checkpoint (.pt or LoRA checkpoint dir).",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Initialize model weights from a checkpoint (.pt or .safetensors) and start a new run "
            "with fresh optimizer / scheduler state."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Stop after this many epochs. Overrides --max-steps by computing steps = ceil(epochs * batches_per_epoch / grad_accum).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Number of micro-batches to accumulate before optimizer.step(). "
            "1 disables accumulation."
        ),
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=256,
        help="Maximum token length for text conditioning (right-truncated).",
    )
    parser.add_argument(
        "--max-caption-len",
        type=int,
        default=None,
        help="Maximum token length for caption conditioning (defaults to max_text_len).",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained-text-encoder-learning-rate",
        type=float,
        default=1e-5,
        help=(
            "AdamW learning rate for a trainable pretrained text/caption backbone. "
            "The main scheduler multiplier is applied to this LR as well."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="muon")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "wsd"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--caption-warmup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "During the first caption_warmup_steps optimizer steps, update only caption-only parameters "
            "(caption encoder/norm and caption attention projections)."
        ),
    )
    parser.add_argument(
        "--caption-warmup-steps",
        type=int,
        default=0,
        help="Number of optimizer steps to run caption-only warmup for when caption_warmup is enabled.",
    )
    parser.add_argument(
        "--pretrained-projector-warmup-steps",
        type=int,
        default=0,
        help=(
            "Update only the text/caption projectors for this many initial optimizer "
            "steps, then update the rest of the trainable TTS model and backbone."
        ),
    )
    parser.add_argument("--stable-steps", type=int, default=0)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="If set, warmup_steps = round(max_steps * warmup_ratio). Computed after max_epochs resolves max_steps.",
    )
    parser.add_argument(
        "--decay-ratio",
        type=float,
        default=None,
        help="If set, decay length = round(max_steps * decay_ratio) for WSD scheduler.",
    )
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--latent-patch-size", type=int, default=1)
    parser.add_argument("--max-latent-steps", type=int, default=750)
    parser.add_argument(
        "--ref-min-seconds",
        type=float,
        default=1.0,
        help=(
            "Minimum reference-audio length (seconds) sampled per training step "
            "when concatenating same-speaker clips to build a long reference."
        ),
    )
    parser.add_argument(
        "--ref-max-seconds",
        type=float,
        default=120.0,
        help=(
            "Maximum reference-audio length (seconds). Concat is capped here, "
            "and the sampled target length is drawn from [min, max]."
        ),
    )
    parser.add_argument(
        "--fixed-target-latent-steps",
        type=int,
        default=None,
        help=(
            "If set, always train on this fixed target latent length "
            "(short samples are right-padded with zeros, long samples are truncated)."
        ),
    )
    parser.add_argument(
        "--fixed-target-full-mask",
        action="store_true",
        help="Use full target mask for fixed-length training (Echo-style includes padded tail in loss).",
    )
    parser.add_argument(
        "--rf-loss-mode",
        choices=["echo", "utterance_mean"],
        default=None,
        help="RF loss normalization mode.",
    )
    parser.add_argument("--duration-loss-weight", type=float, default=None)
    parser.add_argument(
        "--duration-backprop-to-condition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Allow joint RF+duration loss to update text/caption projectors and speaker "
            "conditioning through the duration predictor."
        ),
    )
    parser.add_argument("--duration-speaker-dropout", type=float, default=None)
    parser.add_argument("--duration-caption-dropout", type=float, default=None)
    parser.add_argument("--duration-huber-delta", type=float, default=None)
    parser.add_argument(
        "--text-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping text conditioning during training.",
    )
    parser.add_argument(
        "--caption-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping caption conditioning during training.",
    )
    parser.add_argument(
        "--speaker-condition-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping speaker/reference conditioning during training.",
    )
    speaker_inversion_group = parser.add_mutually_exclusive_group()
    speaker_inversion_group.add_argument(
        "--speaker-inversion",
        dest="speaker_inversion_enabled",
        action="store_true",
        help="Train only learned speaker inversion embedding tokens.",
    )
    speaker_inversion_group.add_argument(
        "--no-speaker-inversion",
        dest="speaker_inversion_enabled",
        action="store_false",
        help="Disable Speaker Inversion training.",
    )
    parser.set_defaults(speaker_inversion_enabled=None)
    parser.add_argument(
        "--speaker-inversion-tokens",
        type=int,
        default=None,
        help="Number of learned Speaker Inversion tokens.",
    )
    parser.add_argument(
        "--speaker-inversion-init-std",
        type=float,
        default=None,
        help="Stddev for random Speaker Inversion token initialization.",
    )
    parser.add_argument(
        "--speaker-inversion-init-embedding",
        default=None,
        help=("Optional existing Speaker Inversion .speaker.safetensors file to continue from."),
    )
    parser.add_argument(
        "--timestep-stratified",
        action="store_true",
        help="Use stratified logit-normal timestep sampling (Echo-style).",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-best-n",
        type=int,
        default=0,
        help=(
            "Keep up to N best validation-loss checkpoints in addition to latest. "
            "When validation is disabled, keeps latest N+1 periodic checkpoints. "
            "Set 0 to disable checkpoint-count limiting."
        ),
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.0,
        help=("Split ratio for validation set from the single manifest. 0 disables validation."),
    )
    parser.add_argument(
        "--valid-every",
        type=int,
        default=0,
        help=("Run validation every N training steps. Set <=0 to disable validation."),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable tqdm progress bar.",
    )
    parser.add_argument(
        "--progress-all",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show tqdm progress bars for all ranks in DDP mode (default: rank0 only).",
    )
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        dest="wandb_enabled",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    wandb_group.add_argument(
        "--no-wandb",
        dest="wandb_enabled",
        action="store_false",
        help="Disable Weights & Biases logging.",
    )
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases entity/team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=sorted(WANDB_MODES),
        default=None,
        help="Weights & Biases mode.",
    )
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--lora",
        dest="lora_enabled",
        action="store_true",
        help="Enable PEFT LoRA fine-tuning.",
    )
    lora_group.add_argument(
        "--no-lora",
        dest="lora_enabled",
        action="store_false",
        help="Disable PEFT LoRA fine-tuning.",
    )
    parser.set_defaults(lora_enabled=None)
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha scaling.")
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="LoRA dropout probability.",
    )
    parser.add_argument(
        "--lora-bias",
        choices=["none", "all", "lora_only"],
        default=None,
        help="Bias handling passed to PEFT LoRA.",
    )
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help=(
            "LoRA target preset, regex, or comma-separated module suffix list. "
            f"Presets: {', '.join(sorted(LORA_TARGET_PRESETS))}."
        ),
    )
    parser.add_argument(
        "--lora-modules-to-save",
        default=None,
        help=(
            "Comma-separated full modules to keep trainable and save inside the LoRA adapter. "
            "Use 'auto' to save duration_predictor for duration-enabled models, or 'none' to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    ddp_group = parser.add_mutually_exclusive_group()
    ddp_group.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_true",
        help=(
            "Enable DDP find_unused_parameters. Useful when conditional branches "
            "(e.g., speaker/text conditioning) may be fully masked in some steps."
        ),
    )
    ddp_group.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="Disable DDP find_unused_parameters.",
    )
    parser.set_defaults(ddp_find_unused_parameters=None)
    return parser
