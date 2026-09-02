#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
import uuid as _uuid
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from irodori_tts.config import (
    ModelConfig,
    TrainConfig,
    dump_configs,
    load_config_yaml,
    merge_dataclass_overrides,
    merge_sample_generation_overrides,
)
from irodori_tts.dataset import LatentTextDataset, TTSCollator, _ManifestIndex
from irodori_tts.duration import set_duration_has_speaker_feature
from irodori_tts.lora import (
    apply_lora,
    count_parameters,
    is_lora_adapter_dir,
    load_lora_adapter,
    train_config_uses_lora,
)
from irodori_tts.model import (
    DURATION_ARCHITECTURES,
    DURATION_CAPTION_FUSIONS,
    DURATION_CAPTION_POOLINGS,
    DURATION_SPEAKER_FUSIONS,
    TextToLatentRFDiT,
)
from irodori_tts.optim import (
    build_optimizer,
    build_scheduler,
    current_lr,
    current_pretrained_text_encoder_lr,
)
from irodori_tts.progress import TrainProgress
from irodori_tts.rf import (
    rf_interpolate,
    rf_velocity_target,
    sample_logit_normal_t,
    sample_stratified_logit_normal_t,
)
from irodori_tts.speaker_inversion import (
    SPEAKER_EMBEDDING_KEY,
    load_speaker_inversion_payload,
)
from irodori_tts.training.checkpointing import (
    RUNTIME_STATE_KEY,
    _collect_dataloader_state,
    _final_checkpoint_path,
    _load_checkpoint_payload,
    _periodic_checkpoint_path,
    _runtime_state_for_checkpoint,
    _select_dataloader_state_for_rank,
    enforce_periodic_checkpoint_limit,
    list_best_val_loss_checkpoints,
    maybe_save_best_val_loss_checkpoint,
    prune_best_val_loss_checkpoints,
    save_checkpoint,
)
from irodori_tts.training.cli_args import (
    TRAIN_MODES,
    WANDB_MODES,
    build_parser,
    cli_provided,
)
from irodori_tts.training.distributed import (
    cuda_prefetch_batches,
    reduce_mean,
    reduce_sum,
    setup_distributed,
)
from irodori_tts.training.duration_metrics import (
    DURATION_CONDITION_GROUP_TOTAL_SIZE,
    duration_condition_group_log_suffix,
    duration_condition_group_metrics,
    duration_condition_group_totals,
    duration_condition_group_wandb_metrics,
)
from irodori_tts.training.losses import compute_rf_loss
from irodori_tts.training.model_init import (
    _apply_base_initialization,
    _check_model_config_compatibility,
    _load_model_state_from_checkpoint,
    _normalize_checkpoint_path,
    _restore_resume_lora_config,
    build_caption_tokenizer,
    build_text_tokenizer,
    clear_non_caption_grads,
    clear_non_pretrained_projector_grads,
    freeze_for_duration_only,
    freeze_for_speaker_inversion,
    validate_caption_backbone_dim,
    validate_text_backbone_dim,
)
from irodori_tts.training.sampler import (
    VALID_MAX_COUNT,
    VALID_MIN_COUNT,
    LengthGroupedSampler,
    split_train_valid_indices,
)
from irodori_tts.training.speaker_prompts import (
    _autopick_prompts_from_manifest,
    _build_prompts_from_speaker_config,
    _resolve_speaker_id,
)
from irodori_tts.training.validation import run_validation

# DACVAE latent frame rate (Hz). Used for seconds<->frames conversion for
# reference audio concat length ranges.
_CODEC_FRAMES_PER_SECOND = 25


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_configs(  # noqa: C901, PLR0912, PLR0915
    *,
    args,
    device,
    distributed,
    is_main_process,
    rank,
) -> tuple:
    raw_argv = sys.argv[1:]
    exp_cfg = load_config_yaml(args.config) if args.config else {}
    unknown_root = sorted(set(exp_cfg) - {"model", "train", "sample_generation"})
    if unknown_root:
        raise ValueError(f"Unknown top-level config keys: {unknown_root}")
    if args.config and is_main_process:
        print(f"Loaded config: {args.config}")
    model_cfg = merge_dataclass_overrides(ModelConfig(), exp_cfg.get("model"), section="model")
    train_cfg = merge_dataclass_overrides(TrainConfig(), exp_cfg.get("train"), section="train")
    sample_cfg = merge_sample_generation_overrides(exp_cfg.get("sample_generation"))
    if not sample_cfg.prompts:
        speaker_prompts = _build_prompts_from_speaker_config(args.manifest)
        if speaker_prompts:
            sample_cfg = replace(sample_cfg, prompts=speaker_prompts)
            if is_main_process:
                print(
                    f"Loaded {len(speaker_prompts)} sample prompts from "
                    f"{Path(args.manifest).parent}/config.yaml:sample_texts"
                )
        else:
            picked = _autopick_prompts_from_manifest(args.manifest)
            if picked:
                sample_cfg = replace(sample_cfg, prompts=picked)
                if is_main_process:
                    print(
                        f"Auto-picked {len(picked)} sample prompts from "
                        f"{args.manifest} (no config.yaml:sample_texts found)"
                    )
    default_train_cfg = TrainConfig()

    train_cfg = replace(train_cfg, manifest_path=args.manifest)
    if train_cfg.output_dir == default_train_cfg.output_dir and not cli_provided(
        raw_argv, "--output-dir"
    ):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)

    if cli_provided(raw_argv, "--output-dir"):
        train_cfg = replace(train_cfg, output_dir=args.output_dir)
    if cli_provided(raw_argv, "--precision"):
        train_cfg = replace(train_cfg, precision=args.precision)
    if args.allow_tf32 is not None:
        train_cfg = replace(train_cfg, allow_tf32=args.allow_tf32)
    if args.compile_model is not None:
        train_cfg = replace(train_cfg, compile_model=args.compile_model)
    if args.gradient_checkpointing is not None:
        train_cfg = replace(train_cfg, gradient_checkpointing=args.gradient_checkpointing)
    if cli_provided(raw_argv, "--train-mode"):
        train_cfg = replace(train_cfg, train_mode=args.train_mode)
    if cli_provided(raw_argv, "--batch-size"):
        train_cfg = replace(train_cfg, batch_size=args.batch_size)
    if cli_provided(raw_argv, "--gradient-accumulation-steps"):
        train_cfg = replace(
            train_cfg,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    if cli_provided(raw_argv, "--max-text-len"):
        train_cfg = replace(train_cfg, max_text_len=args.max_text_len)
    if cli_provided(raw_argv, "--max-caption-len"):
        train_cfg = replace(train_cfg, max_caption_len=args.max_caption_len)
    if cli_provided(raw_argv, "--num-workers"):
        train_cfg = replace(train_cfg, num_workers=args.num_workers)
    if cli_provided(raw_argv, "--lr"):
        train_cfg = replace(train_cfg, learning_rate=args.lr)
    if cli_provided(raw_argv, "--pretrained-text-encoder-learning-rate"):
        train_cfg = replace(
            train_cfg,
            pretrained_text_encoder_learning_rate=(args.pretrained_text_encoder_learning_rate),
        )
    if cli_provided(raw_argv, "--weight-decay"):
        train_cfg = replace(train_cfg, weight_decay=args.weight_decay)
    if cli_provided(raw_argv, "--optimizer"):
        train_cfg = replace(train_cfg, optimizer=args.optimizer)
    if cli_provided(raw_argv, "--adam-beta1"):
        train_cfg = replace(train_cfg, adam_beta1=args.adam_beta1)
    if cli_provided(raw_argv, "--adam-beta2"):
        train_cfg = replace(train_cfg, adam_beta2=args.adam_beta2)
    if cli_provided(raw_argv, "--adam-eps"):
        train_cfg = replace(train_cfg, adam_eps=args.adam_eps)
    if cli_provided(raw_argv, "--muon-momentum"):
        train_cfg = replace(train_cfg, muon_momentum=args.muon_momentum)
    if cli_provided(raw_argv, "--lr-scheduler"):
        train_cfg = replace(train_cfg, lr_scheduler=args.lr_scheduler)
    if cli_provided(raw_argv, "--warmup-steps"):
        train_cfg = replace(train_cfg, warmup_steps=args.warmup_steps)
    if args.caption_warmup is not None:
        train_cfg = replace(train_cfg, caption_warmup=bool(args.caption_warmup))
    if cli_provided(raw_argv, "--caption-warmup-steps"):
        train_cfg = replace(train_cfg, caption_warmup_steps=args.caption_warmup_steps)
    if cli_provided(raw_argv, "--pretrained-projector-warmup-steps"):
        train_cfg = replace(
            train_cfg,
            pretrained_projector_warmup_steps=args.pretrained_projector_warmup_steps,
        )
    if cli_provided(raw_argv, "--stable-steps"):
        train_cfg = replace(train_cfg, stable_steps=args.stable_steps)
    if cli_provided(raw_argv, "--warmup-ratio"):
        train_cfg = replace(train_cfg, warmup_ratio=args.warmup_ratio)
    if cli_provided(raw_argv, "--decay-ratio"):
        train_cfg = replace(train_cfg, decay_ratio=args.decay_ratio)
    if cli_provided(raw_argv, "--min-lr-scale"):
        train_cfg = replace(train_cfg, min_lr_scale=args.min_lr_scale)
    if cli_provided(raw_argv, "--max-steps"):
        train_cfg = replace(train_cfg, max_steps=args.max_steps)
    if cli_provided(raw_argv, "--max-epochs"):
        train_cfg = replace(train_cfg, max_epochs=args.max_epochs)
    if cli_provided(raw_argv, "--text-condition-dropout"):
        train_cfg = replace(train_cfg, text_condition_dropout=args.text_condition_dropout)
    if cli_provided(raw_argv, "--caption-condition-dropout"):
        train_cfg = replace(train_cfg, caption_condition_dropout=args.caption_condition_dropout)
    if cli_provided(raw_argv, "--speaker-condition-dropout"):
        train_cfg = replace(train_cfg, speaker_condition_dropout=args.speaker_condition_dropout)
    if args.speaker_inversion_enabled is not None:
        train_cfg = replace(
            train_cfg,
            speaker_inversion_enabled=bool(args.speaker_inversion_enabled),
        )
    if cli_provided(raw_argv, "--speaker-inversion-tokens"):
        train_cfg = replace(train_cfg, speaker_inversion_tokens=args.speaker_inversion_tokens)
    if cli_provided(raw_argv, "--speaker-inversion-init-std"):
        train_cfg = replace(train_cfg, speaker_inversion_init_std=args.speaker_inversion_init_std)
    if cli_provided(raw_argv, "--speaker-inversion-init-embedding"):
        train_cfg = replace(
            train_cfg,
            speaker_inversion_init_embedding=args.speaker_inversion_init_embedding,
        )
    if cli_provided(raw_argv, "--timestep-stratified"):
        train_cfg = replace(train_cfg, timestep_stratified=True)
    if cli_provided(raw_argv, "--max-latent-steps"):
        train_cfg = replace(train_cfg, max_latent_steps=args.max_latent_steps)
    if cli_provided(raw_argv, "--ref-min-seconds"):
        train_cfg = replace(train_cfg, ref_min_seconds=args.ref_min_seconds)
    if cli_provided(raw_argv, "--ref-max-seconds"):
        train_cfg = replace(train_cfg, ref_max_seconds=args.ref_max_seconds)
    if cli_provided(raw_argv, "--fixed-target-latent-steps"):
        train_cfg = replace(train_cfg, fixed_target_latent_steps=args.fixed_target_latent_steps)
    if cli_provided(raw_argv, "--fixed-target-full-mask"):
        train_cfg = replace(train_cfg, fixed_target_full_mask=True)
    if cli_provided(raw_argv, "--rf-loss-mode"):
        train_cfg = replace(train_cfg, rf_loss_mode=args.rf_loss_mode)
    if cli_provided(raw_argv, "--duration-loss-weight"):
        train_cfg = replace(train_cfg, duration_loss_weight=args.duration_loss_weight)
    if args.duration_backprop_to_condition is not None:
        train_cfg = replace(
            train_cfg,
            duration_backprop_to_condition=bool(args.duration_backprop_to_condition),
        )
    if cli_provided(raw_argv, "--duration-speaker-dropout"):
        train_cfg = replace(train_cfg, duration_speaker_dropout=args.duration_speaker_dropout)
    if cli_provided(raw_argv, "--duration-caption-dropout"):
        train_cfg = replace(train_cfg, duration_caption_dropout=args.duration_caption_dropout)
    if cli_provided(raw_argv, "--duration-huber-delta"):
        train_cfg = replace(train_cfg, duration_huber_delta=args.duration_huber_delta)
    if cli_provided(raw_argv, "--log-every"):
        train_cfg = replace(train_cfg, log_every=args.log_every)
    if cli_provided(raw_argv, "--save-every"):
        train_cfg = replace(train_cfg, save_every=args.save_every)
    if cli_provided(raw_argv, "--checkpoint-best-n"):
        train_cfg = replace(train_cfg, checkpoint_best_n=args.checkpoint_best_n)
    if cli_provided(raw_argv, "--valid-ratio"):
        train_cfg = replace(train_cfg, valid_ratio=args.valid_ratio)
    if cli_provided(raw_argv, "--valid-every"):
        train_cfg = replace(train_cfg, valid_every=args.valid_every)
    if args.progress is not None:
        train_cfg = replace(train_cfg, progress=args.progress)
    if args.progress_all is not None:
        train_cfg = replace(train_cfg, progress_all_ranks=args.progress_all)
    if args.wandb_enabled is not None:
        train_cfg = replace(train_cfg, wandb_enabled=args.wandb_enabled)
    if cli_provided(raw_argv, "--wandb-project"):
        train_cfg = replace(train_cfg, wandb_project=args.wandb_project)
    if cli_provided(raw_argv, "--wandb-entity"):
        train_cfg = replace(train_cfg, wandb_entity=args.wandb_entity)
    if cli_provided(raw_argv, "--wandb-run-name"):
        train_cfg = replace(train_cfg, wandb_run_name=args.wandb_run_name)
    if cli_provided(raw_argv, "--wandb-mode"):
        train_cfg = replace(train_cfg, wandb_mode=args.wandb_mode)
    if args.lora_enabled is not None:
        train_cfg = replace(train_cfg, lora_enabled=args.lora_enabled)
    if cli_provided(raw_argv, "--lora-r"):
        train_cfg = replace(train_cfg, lora_r=args.lora_r)
    if cli_provided(raw_argv, "--lora-alpha"):
        train_cfg = replace(train_cfg, lora_alpha=args.lora_alpha)
    if cli_provided(raw_argv, "--lora-dropout"):
        train_cfg = replace(train_cfg, lora_dropout=args.lora_dropout)
    if cli_provided(raw_argv, "--lora-bias"):
        train_cfg = replace(train_cfg, lora_bias=args.lora_bias)
    if cli_provided(raw_argv, "--lora-target-modules"):
        train_cfg = replace(train_cfg, lora_target_modules=args.lora_target_modules)
    if cli_provided(raw_argv, "--lora-modules-to-save"):
        train_cfg = replace(train_cfg, lora_modules_to_save=args.lora_modules_to_save)
    if args.ddp_find_unused_parameters is not None:
        train_cfg = replace(
            train_cfg,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        )
    if cli_provided(raw_argv, "--seed"):
        train_cfg = replace(train_cfg, seed=args.seed)

    resume_path = Path(args.resume).expanduser() if args.resume is not None else None
    resume_model_cfg = None
    resume_train_cfg = None
    resume_base_init = None
    resume_text_encoder_config = None
    if args.resume is not None:
        resume_meta = _load_checkpoint_payload(resume_path, map_location="cpu")
        raw_resume_model_cfg = resume_meta.get("model_config")
        if raw_resume_model_cfg is not None and not isinstance(raw_resume_model_cfg, dict):
            raise ValueError("Resume checkpoint model_config must be a dictionary when present.")
        resume_model_cfg = raw_resume_model_cfg
        raw_resume_train_cfg = resume_meta.get("train_config")
        if raw_resume_train_cfg is not None and not isinstance(raw_resume_train_cfg, dict):
            raise ValueError("Resume checkpoint train_config must be a dictionary when present.")
        resume_train_cfg = raw_resume_train_cfg
        raw_resume_base_init = resume_meta.get("base_init")
        if raw_resume_base_init is not None and not isinstance(raw_resume_base_init, dict):
            raise ValueError("Resume checkpoint base_init must be a dictionary when present.")
        resume_base_init = raw_resume_base_init
        raw_resume_text_encoder_config = resume_meta.get("text_encoder_config")
        if raw_resume_text_encoder_config is not None and not isinstance(
            raw_resume_text_encoder_config, dict
        ):
            raise ValueError(
                "Resume checkpoint text_encoder_config must be a dictionary when present."
            )
        resume_text_encoder_config = raw_resume_text_encoder_config
        train_cfg = _restore_resume_lora_config(
            train_cfg,
            resume_train_cfg=resume_train_cfg,
            args=args,
            raw_argv=raw_argv,
            exp_cfg=exp_cfg,
        )
        del resume_meta

    if cli_provided(raw_argv, "--latent-dim"):
        model_cfg = replace(model_cfg, latent_dim=args.latent_dim)
    if cli_provided(raw_argv, "--latent-patch-size"):
        model_cfg = replace(model_cfg, latent_patch_size=args.latent_patch_size)

    set_seed(train_cfg.seed + rank)
    text_encoder_type = str(model_cfg.text_encoder_type).strip().lower()
    if text_encoder_type not in {"scratch", "pretrained"}:
        raise ValueError(
            "model.text_encoder_type must be 'scratch' or 'pretrained', "
            f"got {model_cfg.text_encoder_type!r}."
        )
    model_cfg = replace(model_cfg, text_encoder_type=text_encoder_type)
    pretrained_projector_type = str(model_cfg.pretrained_projector_type).strip().lower()
    if pretrained_projector_type not in {"linear", "residual_mlp"}:
        raise ValueError(
            "model.pretrained_projector_type must be 'linear' or 'residual_mlp', "
            f"got {model_cfg.pretrained_projector_type!r}."
        )
    if model_cfg.pretrained_projector_hidden_ratio <= 0:
        raise ValueError(
            "model.pretrained_projector_hidden_ratio must be > 0, got "
            f"{model_cfg.pretrained_projector_hidden_ratio}."
        )
    if not 0.0 <= model_cfg.pretrained_projector_dropout <= 1.0:
        raise ValueError(
            "model.pretrained_projector_dropout must be in [0, 1], got "
            f"{model_cfg.pretrained_projector_dropout}."
        )
    model_cfg = replace(
        model_cfg,
        pretrained_projector_type=pretrained_projector_type,
    )
    if train_cfg.pretrained_text_encoder_learning_rate <= 0:
        raise ValueError(
            "pretrained_text_encoder_learning_rate must be > 0, got "
            f"{train_cfg.pretrained_text_encoder_learning_rate}."
        )
    if (
        model_cfg.use_pretrained_text_encoder
        and model_cfg.use_caption_condition
        and model_cfg.caption_tokenizer_repo_resolved != model_cfg.text_tokenizer_repo
    ):
        raise ValueError(
            "Pretrained text/caption encoder sharing requires caption_tokenizer_repo "
            "to be unset or equal to text_tokenizer_repo."
        )
    if args.resume is not None:
        if model_cfg.use_pretrained_text_encoder and resume_model_cfg is None:
            raise ValueError(
                "Pretrained text encoder resume requires checkpoint model_config metadata "
                "to verify the backbone architecture and configuration."
            )
        if resume_path is None:
            raise RuntimeError("Resume path is unexpectedly missing.")
        _check_model_config_compatibility(
            resume_path,
            resume_model_cfg,
            model_cfg,
            require_caption_match=True,
        )
    if not (0.0 <= train_cfg.text_condition_dropout <= 1.0):
        raise ValueError(
            f"text_condition_dropout must be in [0, 1], got {train_cfg.text_condition_dropout}"
        )
    if train_cfg.max_text_len <= 0:
        raise ValueError(f"max_text_len must be > 0, got {train_cfg.max_text_len}")
    if str(train_cfg.train_mode).strip().lower() not in TRAIN_MODES:
        raise ValueError(
            f"train_mode must be one of {sorted(TRAIN_MODES)}, got {train_cfg.train_mode!r}"
        )
    train_cfg = replace(train_cfg, train_mode=str(train_cfg.train_mode).strip().lower())
    if train_cfg.max_caption_len is not None and train_cfg.max_caption_len <= 0:
        raise ValueError(f"max_caption_len must be > 0, got {train_cfg.max_caption_len}")
    if train_cfg.gradient_accumulation_steps <= 0:
        raise ValueError(
            f"gradient_accumulation_steps must be > 0, got {train_cfg.gradient_accumulation_steps}"
        )
    if not (0.0 <= train_cfg.speaker_condition_dropout <= 1.0):
        raise ValueError(
            "speaker_condition_dropout must be in [0, 1], "
            f"got {train_cfg.speaker_condition_dropout}"
        )
    if train_cfg.speaker_inversion_enabled:
        if not model_cfg.use_speaker_condition_resolved:
            raise ValueError(
                "speaker_inversion_enabled=True requires a speaker-conditioned model config."
            )
        if args.init_checkpoint is None:
            raise ValueError(
                "speaker_inversion_enabled=True requires --init-checkpoint so the frozen "
                "base TTS model is initialized from trained weights."
            )
        if args.resume is not None:
            raise ValueError(
                "speaker_inversion_enabled=True saves embedding-only checkpoints; "
                "--resume full trainer state is not supported. Use "
                "speaker_inversion_init_embedding to continue from a saved embedding."
            )
        if train_config_uses_lora(train_cfg):
            raise ValueError("speaker_inversion_enabled=True does not support LoRA training.")
        if train_cfg.train_mode != "rf":
            raise ValueError("speaker_inversion_enabled=True supports train_mode='rf' only.")
        if train_cfg.caption_warmup:
            raise ValueError("speaker_inversion_enabled=True does not support caption_warmup.")
        if train_cfg.speaker_inversion_tokens <= 0:
            raise ValueError(
                f"speaker_inversion_tokens must be > 0, got {train_cfg.speaker_inversion_tokens}"
            )
        if train_cfg.speaker_inversion_init_std < 0:
            raise ValueError(
                "speaker_inversion_init_std must be >= 0, "
                f"got {train_cfg.speaker_inversion_init_std}"
            )
        optimizer_explicit = cli_provided(raw_argv, "--optimizer") or (
            isinstance(exp_cfg.get("train"), dict) and "optimizer" in exp_cfg.get("train", {})
        )
        if str(train_cfg.optimizer).strip().lower() == "muon":
            if optimizer_explicit:
                raise ValueError(
                    "speaker_inversion_enabled=True supports optimizer='adamw'. "
                    "Muon has no compatible matrix parameter when only speaker tokens are trainable."
                )
            train_cfg = replace(train_cfg, optimizer="adamw")
    if not (0.0 <= train_cfg.caption_condition_dropout <= 1.0):
        raise ValueError(
            "caption_condition_dropout must be in [0, 1], "
            f"got {train_cfg.caption_condition_dropout}"
        )
    if train_cfg.fixed_target_latent_steps is not None and train_cfg.fixed_target_latent_steps <= 0:
        raise ValueError(
            "fixed_target_latent_steps must be > 0 when provided, "
            f"got {train_cfg.fixed_target_latent_steps}"
        )
    if train_cfg.fixed_target_full_mask and train_cfg.fixed_target_latent_steps is None:
        raise ValueError(
            "fixed_target_full_mask=True requires fixed_target_latent_steps to be set."
        )
    if str(train_cfg.rf_loss_mode).strip().lower() not in {"echo", "utterance_mean"}:
        raise ValueError(
            "rf_loss_mode must be one of ['echo', 'utterance_mean'], "
            f"got {train_cfg.rf_loss_mode!r}"
        )
    if train_cfg.duration_loss_weight < 0:
        raise ValueError(f"duration_loss_weight must be >= 0, got {train_cfg.duration_loss_weight}")
    if train_cfg.duration_backprop_to_condition:
        if not model_cfg.use_duration_predictor:
            raise ValueError(
                "duration_backprop_to_condition=True requires model.use_duration_predictor=True."
            )
        if train_cfg.train_mode != "rf":
            raise ValueError(
                "duration_backprop_to_condition=True is only supported for joint "
                "train_mode='rf' training."
            )
    if not (0.0 <= train_cfg.duration_speaker_dropout <= 1.0):
        raise ValueError(
            f"duration_speaker_dropout must be in [0, 1], got {train_cfg.duration_speaker_dropout}"
        )
    if not (0.0 <= train_cfg.duration_caption_dropout <= 1.0):
        raise ValueError(
            f"duration_caption_dropout must be in [0, 1], got {train_cfg.duration_caption_dropout}"
        )
    if train_cfg.duration_huber_delta <= 0:
        raise ValueError(f"duration_huber_delta must be > 0, got {train_cfg.duration_huber_delta}")
    if train_cfg.train_mode == "duration_only" and not model_cfg.use_duration_predictor:
        raise ValueError("train_mode='duration_only' requires model.use_duration_predictor=True.")
    if train_cfg.train_mode == "duration_only" and train_config_uses_lora(train_cfg):
        raise ValueError("train_mode='duration_only' does not support LoRA training.")
    if train_cfg.train_mode == "duration_only" and train_cfg.caption_warmup:
        raise ValueError("train_mode='duration_only' does not support caption_warmup.")
    if (
        train_cfg.train_mode == "duration_only"
        and args.init_checkpoint is None
        and args.resume is None
    ):
        raise ValueError(
            "train_mode='duration_only' requires --init-checkpoint or --resume "
            "so the frozen text/speaker encoders are initialized from trained weights."
        )
    if model_cfg.use_duration_predictor:
        if model_cfg.duration_aux_dim <= 0:
            raise ValueError(f"duration_aux_dim must be > 0, got {model_cfg.duration_aux_dim}")
        if model_cfg.duration_hidden_dim <= 0:
            raise ValueError(
                f"duration_hidden_dim must be > 0, got {model_cfg.duration_hidden_dim}"
            )
        if model_cfg.duration_layers <= 0:
            raise ValueError(f"duration_layers must be > 0, got {model_cfg.duration_layers}")
        if not (0.0 <= model_cfg.duration_dropout <= 1.0):
            raise ValueError(
                f"duration_dropout must be in [0, 1], got {model_cfg.duration_dropout}"
            )
        if model_cfg.duration_attention_heads <= 0:
            raise ValueError(
                f"duration_attention_heads must be > 0, got {model_cfg.duration_attention_heads}"
            )
        if model_cfg.text_dim % model_cfg.duration_attention_heads != 0:
            raise ValueError(
                "text_dim must be divisible by duration_attention_heads: "
                f"text_dim={model_cfg.text_dim}, "
                f"duration_attention_heads={model_cfg.duration_attention_heads}"
            )
        duration_architecture = str(model_cfg.duration_architecture).strip().lower()
        if duration_architecture not in DURATION_ARCHITECTURES:
            raise ValueError(
                "duration_architecture must be one of "
                f"{sorted(DURATION_ARCHITECTURES)}, got {model_cfg.duration_architecture!r}"
            )
        if model_cfg.duration_token_init_frames <= 0:
            raise ValueError(
                "duration_token_init_frames must be > 0, "
                f"got {model_cfg.duration_token_init_frames}"
            )
        duration_speaker_fusion = str(model_cfg.duration_speaker_fusion).strip().lower()
        if duration_speaker_fusion not in DURATION_SPEAKER_FUSIONS:
            raise ValueError(
                "duration_speaker_fusion must be one of "
                f"{sorted(DURATION_SPEAKER_FUSIONS)}, got {model_cfg.duration_speaker_fusion!r}"
            )
        duration_caption_fusion = str(model_cfg.duration_caption_fusion).strip().lower()
        if duration_caption_fusion not in DURATION_CAPTION_FUSIONS:
            raise ValueError(
                "duration_caption_fusion must be one of "
                f"{sorted(DURATION_CAPTION_FUSIONS)}, got {model_cfg.duration_caption_fusion!r}"
            )
        duration_caption_pooling = str(model_cfg.duration_caption_pooling).strip().lower()
        if duration_caption_pooling not in DURATION_CAPTION_POOLINGS:
            raise ValueError(
                "duration_caption_pooling must be one of "
                f"{sorted(DURATION_CAPTION_POOLINGS)}, got {model_cfg.duration_caption_pooling!r}"
            )
        if (
            duration_architecture == "token_sum_adarn_zero_no_aux"
            and duration_speaker_fusion != "adarn_zero"
        ):
            raise ValueError(
                "duration_architecture='token_sum_adarn_zero_no_aux' requires "
                "duration_speaker_fusion='adarn_zero'."
            )
        if duration_architecture == "token_sum_dual_adarn_zero_no_aux":
            if duration_speaker_fusion != "adarn_zero":
                raise ValueError(
                    "duration_architecture='token_sum_dual_adarn_zero_no_aux' requires "
                    "duration_speaker_fusion='adarn_zero'."
                )
            if duration_caption_fusion != "adarn_zero":
                raise ValueError(
                    "duration_architecture='token_sum_dual_adarn_zero_no_aux' requires "
                    "duration_caption_fusion='adarn_zero'."
                )
            if not model_cfg.use_speaker_condition_resolved or not model_cfg.use_caption_condition:
                raise ValueError(
                    "duration_architecture='token_sum_dual_adarn_zero_no_aux' requires "
                    "both speaker and caption conditioning."
                )
        model_cfg = replace(
            model_cfg,
            duration_architecture=duration_architecture,
            duration_speaker_fusion=duration_speaker_fusion,
            duration_caption_fusion=duration_caption_fusion,
            duration_caption_pooling=duration_caption_pooling,
        )
    if train_cfg.caption_warmup_steps < 0:
        raise ValueError(f"caption_warmup_steps must be >= 0, got {train_cfg.caption_warmup_steps}")
    if train_cfg.pretrained_projector_warmup_steps < 0:
        raise ValueError(
            "pretrained_projector_warmup_steps must be >= 0, got "
            f"{train_cfg.pretrained_projector_warmup_steps}"
        )
    if train_cfg.pretrained_projector_warmup_steps > 0:
        if not model_cfg.use_pretrained_text_encoder:
            raise ValueError(
                "pretrained_projector_warmup_steps requires model.text_encoder_type='pretrained'."
            )
        if args.init_checkpoint is None and args.resume is None:
            raise ValueError(
                "pretrained projector warmup requires --init-checkpoint or --resume; "
                "it is intended for replacing an encoder in a trained TTS model."
            )
        if train_cfg.caption_warmup:
            raise ValueError(
                "pretrained projector warmup and caption_warmup cannot be enabled together."
            )
        if train_cfg.train_mode == "duration_only":
            raise ValueError(
                "pretrained projector warmup requires train_mode='rf' so projector and TTS "
                "parameters remain in the optimizer."
            )
        if train_config_uses_lora(train_cfg):
            raise ValueError("pretrained projector warmup does not support LoRA training.")
        if train_cfg.speaker_inversion_enabled:
            raise ValueError(
                "pretrained projector warmup does not support Speaker Inversion training."
            )
    if train_cfg.dataloader_prefetch_factor <= 0:
        raise ValueError(
            f"dataloader_prefetch_factor must be > 0, got {train_cfg.dataloader_prefetch_factor}"
        )
    if train_cfg.length_bucket_window_batches <= 0:
        raise ValueError(
            "length_bucket_window_batches must be > 0, "
            f"got {train_cfg.length_bucket_window_batches}"
        )
    if train_cfg.latent_length_bucket_size < 0:
        raise ValueError(
            f"latent_length_bucket_size must be >= 0, got {train_cfg.latent_length_bucket_size}"
        )
    if not (0.0 <= train_cfg.valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be in [0, 1), got {train_cfg.valid_ratio}")
    if train_cfg.valid_every < 0:
        raise ValueError(f"valid_every must be >= 0, got {train_cfg.valid_every}")
    if train_cfg.valid_ratio > 0.0 and train_cfg.valid_every <= 0:
        raise ValueError("valid_every must be > 0 when valid_ratio > 0.")
    if train_cfg.valid_ratio == 0.0 and train_cfg.valid_every > 0 and is_main_process:
        print("warning: valid_every is set but valid_ratio=0. Validation is disabled.")
    if train_cfg.checkpoint_best_n < 0:
        raise ValueError(f"checkpoint_best_n must be >= 0, got {train_cfg.checkpoint_best_n}")
    if train_cfg.wandb_mode not in WANDB_MODES:
        raise ValueError(
            f"wandb_mode must be one of {sorted(WANDB_MODES)}, got {train_cfg.wandb_mode!r}"
        )
    precision = str(train_cfg.precision).lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"precision must be one of ['fp32', 'bf16'], got {train_cfg.precision!r}")
    if precision == "bf16":
        if device.type != "cuda":
            if is_main_process:
                print("warning: precision=bf16 requested on non-CUDA device. Falling back to fp32.")
            train_cfg = replace(train_cfg, precision="fp32")
        elif not torch.cuda.is_bf16_supported():
            if is_main_process:
                print("warning: CUDA bf16 is not supported on this GPU. Falling back to fp32.")
            train_cfg = replace(train_cfg, precision="fp32")
    use_bf16 = train_cfg.precision == "bf16"
    if device.type == "cuda":
        tf32_enabled = bool(train_cfg.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        torch.set_float32_matmul_precision("high" if tf32_enabled else "highest")
        if is_main_process:
            print(f"TF32 matmul/cuDNN: {'enabled' if tf32_enabled else 'disabled'}")
    elif train_cfg.allow_tf32 and is_main_process:
        print("warning: allow_tf32=True requested on non-CUDA device; ignoring.")

    output_dir = Path(train_cfg.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_configs(output_dir / "config.json", model_cfg, train_cfg)
        print(f"Compute precision={train_cfg.precision} (weights/optimizer states kept in fp32).")
    if distributed:
        dist.barrier()
    return (
        exp_cfg,
        model_cfg,
        output_dir,
        resume_base_init,
        resume_model_cfg,
        resume_path,
        resume_text_encoder_config,
        sample_cfg,
        train_cfg,
        use_bf16,
    )


def _setup_wandb_and_tokenizers(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    args,
    distributed,
    is_main_process,
    model_cfg,
    output_dir,
    train_cfg,
) -> tuple:
    from irodori_tts.wandb_client import WandbClient
    from irodori_tts.wandb_client import from_env as _wandb_cfg_from_env

    # Resolve the persistent run UUID before wandb.init so the same wandb run
    # is reused when training resumes from a checkpoint. The uuid is stored on
    # the prior adapter_model.safetensors metadata (see _build_lora_safetensors_metadata).
    run_uuid: str | None = None
    if args.resume is not None:
        try:
            from safetensors import safe_open as _safe_open_for_uuid

            _adapter_path = Path(args.resume) / "adapter_model.safetensors"
            if _adapter_path.is_file():
                with _safe_open_for_uuid(str(_adapter_path), framework="pt", device="cpu") as _f:
                    run_uuid = (_f.metadata() or {}).get("uuid")
        except Exception:
            run_uuid = None
    if not run_uuid:
        run_uuid = str(_uuid.uuid4())

    wandb_client = WandbClient(
        _wandb_cfg_from_env(
            enabled=train_cfg.wandb_enabled and is_main_process,
            project=train_cfg.wandb_project,
            entity=train_cfg.wandb_entity,
            run_name=train_cfg.wandb_run_name,
            mode=train_cfg.wandb_mode or "online",
            run_id=run_uuid,
            resume="allow",
        ),
        config={
            "model": asdict(model_cfg),
            "train": asdict(train_cfg),
            "script": "train.py",
        },
        output_dir=output_dir,
    )
    if wandb_client.enabled:
        print(
            f"W&B enabled: project={train_cfg.wandb_project} mode={train_cfg.wandb_mode} "
            f"run={wandb_client.name} base_url={wandb_client.base_url or '<default>'}"
        )

    # The distributed path assigns these through two complementary branches, so
    # bind them up front to keep the definite-assignment check simple.
    tokenizer = None
    caption_tokenizer = None
    if distributed:
        local_files_only = not is_main_process
        if is_main_process:
            tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)
            text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=False)
            caption_tokenizer = None
            caption_hidden_size = None
            if model_cfg.use_caption_condition:
                caption_tokenizer = build_caption_tokenizer(model_cfg, local_files_only=False)
                caption_hidden_size = validate_caption_backbone_dim(
                    model_cfg,
                    local_files_only=False,
                )
        dist.barrier()
        if not is_main_process:
            tokenizer = build_text_tokenizer(model_cfg, local_files_only=local_files_only)
            text_hidden_size = validate_text_backbone_dim(
                model_cfg,
                local_files_only=local_files_only,
            )
            caption_tokenizer = None
            caption_hidden_size = None
            if model_cfg.use_caption_condition:
                caption_tokenizer = build_caption_tokenizer(
                    model_cfg,
                    local_files_only=local_files_only,
                )
                caption_hidden_size = validate_caption_backbone_dim(
                    model_cfg,
                    local_files_only=local_files_only,
                )
        dist.barrier()
    else:
        tokenizer = build_text_tokenizer(model_cfg, local_files_only=False)
        text_hidden_size = validate_text_backbone_dim(model_cfg, local_files_only=False)
        caption_tokenizer = None
        caption_hidden_size = None
        if model_cfg.use_caption_condition:
            caption_tokenizer = build_caption_tokenizer(model_cfg, local_files_only=False)
            caption_hidden_size = validate_caption_backbone_dim(
                model_cfg,
                local_files_only=False,
            )
    if is_main_process:
        print(
            f"Text tokenizer={model_cfg.text_tokenizer_repo} vocab={tokenizer.vocab_size} add_bos={model_cfg.text_add_bos} padding_side=right "
            f"(pretrained hidden_size={text_hidden_size})."
        )
        if model_cfg.use_caption_condition and caption_tokenizer is not None:
            print(
                f"Caption tokenizer={model_cfg.caption_tokenizer_repo_resolved} vocab={caption_tokenizer.vocab_size} add_bos={model_cfg.caption_add_bos_resolved} padding_side=right "
                f"(pretrained hidden_size={caption_hidden_size})."
            )
    return (
        caption_tokenizer,
        run_uuid,
        tokenizer,
        wandb_client,
    )


def _build_data(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    caption_tokenizer,
    device,
    distributed,
    is_main_process,
    model_cfg,
    output_dir,
    rank,
    run_uuid,
    tokenizer,
    train_cfg,
    wandb_client,
    world_size,
) -> tuple:
    manifest_index = _ManifestIndex.build(
        manifest_path=Path(train_cfg.manifest_path),
        show_progress=bool(train_cfg.progress and is_main_process),
        progress_desc="Index Manifest",
    )
    manifest_size = len(manifest_index.offsets)
    ref_min_frames_cfg: int | None = None
    ref_max_frames_cfg: int | None = None
    if model_cfg.use_speaker_condition_resolved and train_cfg.ref_max_seconds > 0.0:
        ref_min_frames_cfg = max(
            1, round(float(train_cfg.ref_min_seconds) * _CODEC_FRAMES_PER_SECOND)
        )
        ref_max_frames_cfg = max(
            ref_min_frames_cfg,
            round(float(train_cfg.ref_max_seconds) * _CODEC_FRAMES_PER_SECOND),
        )
        if is_main_process:
            print(
                "Reference concat enabled: "
                f"ref_min_seconds={train_cfg.ref_min_seconds} "
                f"ref_max_seconds={train_cfg.ref_max_seconds} "
                f"(frames {ref_min_frames_cfg}..{ref_max_frames_cfg} at "
                f"{_CODEC_FRAMES_PER_SECOND} Hz)."
            )
    valid_dataset = None
    if train_cfg.valid_ratio > 0.0:
        train_indices, valid_indices = split_train_valid_indices(
            num_samples=len(manifest_index.offsets),
            valid_ratio=train_cfg.valid_ratio,
            seed=train_cfg.seed,
        )
        train_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=train_indices,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
            manifest_index=manifest_index,
            ref_min_frames=ref_min_frames_cfg,
            ref_max_frames=ref_max_frames_cfg,
        )
        valid_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=valid_indices,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
            manifest_index=manifest_index,
            ref_min_frames=ref_min_frames_cfg,
            ref_max_frames=ref_max_frames_cfg,
        )
        if is_main_process:
            print(
                f"Validation split enabled: train={len(train_dataset)} valid={len(valid_dataset)} "
                f"(ratio={train_cfg.valid_ratio:.4f}, clamp=[{VALID_MIN_COUNT},{VALID_MAX_COUNT}], valid_every={train_cfg.valid_every} steps)."
            )
    else:
        train_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition_resolved,
            manifest_index=manifest_index,
            ref_min_frames=ref_min_frames_cfg,
            ref_max_frames=ref_max_frames_cfg,
        )
    drop_last = len(train_dataset) >= train_cfg.batch_size
    if not drop_last and is_main_process:
        print(
            f"warning: dataset size ({len(train_dataset)}) is smaller than batch_size ({train_cfg.batch_size}). "
            "Using drop_last=False to avoid empty dataloader."
        )
    collator = TTSCollator(
        tokenizer=tokenizer,
        caption_tokenizer=caption_tokenizer,
        latent_dim=model_cfg.latent_dim,
        latent_patch_size=model_cfg.latent_patch_size,
        fixed_target_latent_steps=train_cfg.fixed_target_latent_steps,
        fixed_target_full_mask=train_cfg.fixed_target_full_mask,
        latent_length_bucket_size=train_cfg.latent_length_bucket_size,
        max_text_len=train_cfg.max_text_len,
        max_caption_len=(
            train_cfg.max_text_len
            if train_cfg.max_caption_len is None
            else train_cfg.max_caption_len
        ),
    )
    if train_cfg.fixed_target_latent_steps is not None and is_main_process:
        print(
            f"Fixed target latent length enabled: steps={train_cfg.fixed_target_latent_steps} full_mask={train_cfg.fixed_target_full_mask}"
        )
    elif train_cfg.latent_length_bucket_size > 0 and is_main_process:
        print(
            "Fixed latent length buckets enabled: "
            f"bucket_size={train_cfg.latent_length_bucket_size}."
        )
    if not model_cfg.use_speaker_condition_resolved and is_main_process:
        print("Speaker conditioning disabled for this model config.")
    if train_cfg.caption_warmup and is_main_process:
        if not model_cfg.use_caption_condition:
            print(
                "warning: caption_warmup=True requested, but caption conditioning is disabled. Ignoring."
            )
        elif train_cfg.caption_warmup_steps <= 0:
            print(
                "warning: caption_warmup=True requested, but caption_warmup_steps <= 0. Ignoring."
            )
        else:
            print(
                "Caption warmup enabled: only caption-only parameters will update for the first "
                f"{train_cfg.caption_warmup_steps} optimizer steps."
            )
    if train_cfg.pretrained_projector_warmup_steps > 0 and is_main_process:
        print(
            "Pretrained projector warmup enabled: only text/caption projectors will update "
            f"for the first {train_cfg.pretrained_projector_warmup_steps} optimizer steps; "
            "all model parameters update after warmup."
        )
    if train_cfg.timestep_stratified and is_main_process:
        print("Using stratified logit-normal timestep sampling.")
    train_sampler = None
    train_loader_generator = None
    if train_cfg.length_bucket_enabled:
        length_bucket_values = train_dataset.length_bucket_values()
        positive_mask = length_bucket_values > 0
        if not bool(positive_mask.any()):
            if is_main_process:
                print(
                    "warning: length_bucket_enabled=True but manifest has no positive num_frames; "
                    "falling back to normal random sampling."
                )
        else:
            if not bool(positive_mask.all()):
                fallback_length = int(length_bucket_values[positive_mask].median().item())
                length_bucket_values[~positive_mask] = fallback_length
                if is_main_process:
                    missing_count = int((~positive_mask).sum().item())
                    print(
                        "warning: length bucket found samples without num_frames; "
                        f"using median length={fallback_length} for {missing_count} samples."
                    )
            train_sampler = LengthGroupedSampler(
                length_bucket_values,
                batch_size=train_cfg.batch_size,
                window_batches=train_cfg.length_bucket_window_batches,
                num_replicas=world_size if distributed else 1,
                rank=rank if distributed else 0,
                seed=train_cfg.seed,
                drop_last=drop_last,
            )
            if is_main_process:
                window_samples = (
                    train_cfg.batch_size
                    * max(1, world_size if distributed else 1)
                    * train_cfg.length_bucket_window_batches
                )
                print(
                    "Length bucket sampling enabled: "
                    f"window_batches={train_cfg.length_bucket_window_batches} "
                    f"window_samples={window_samples} "
                    "with shuffled batch order."
                )
    if train_sampler is None and distributed:
        train_sampler = StatefulDistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=train_cfg.seed,
            drop_last=drop_last,
        )
    elif train_sampler is None:
        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(int(train_cfg.seed))
    dataloader_common_kwargs = {
        "batch_size": train_cfg.batch_size,
        "num_workers": train_cfg.num_workers,
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collator,
    }
    if train_cfg.num_workers > 0:
        dataloader_common_kwargs["persistent_workers"] = bool(
            train_cfg.dataloader_persistent_workers
        )
        dataloader_common_kwargs["prefetch_factor"] = int(train_cfg.dataloader_prefetch_factor)
    elif train_cfg.dataloader_persistent_workers and is_main_process:
        print("warning: dataloader_persistent_workers=True is ignored because num_workers=0.")
    if train_cfg.dataloader_cuda_prefetch and device.type != "cuda" and is_main_process:
        print("warning: dataloader_cuda_prefetch=True is ignored because device is not CUDA.")
    loader = StatefulDataLoader(
        dataset=train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=drop_last,
        generator=train_loader_generator,
        snapshot_every_n_steps=1,
        **dataloader_common_kwargs,
    )
    if len(loader) == 0:
        raise ValueError("Dataloader yielded zero batches. Check manifest and batch_size settings.")
    import math

    optim_steps_per_epoch = max(
        1,
        math.ceil(len(loader) / max(1, train_cfg.gradient_accumulation_steps)),
    )

    speaker_name = _resolve_speaker_id(train_cfg.manifest_path)
    run_name = wandb_client.name or train_cfg.wandb_run_name or output_dir.name
    if is_main_process:
        print(f"[run identity] uuid={run_uuid} name={run_name} speaker={speaker_name}")
    if train_cfg.max_epochs is not None:
        grad_accum = max(1, train_cfg.gradient_accumulation_steps)
        derived_max_steps = train_cfg.max_epochs * optim_steps_per_epoch
        if rank == 0:
            print(
                f"[max_epochs={train_cfg.max_epochs}] batches_per_epoch={len(loader)}, "
                f"grad_accum={grad_accum}, optim_steps_per_epoch={optim_steps_per_epoch}, "
                f"derived max_steps={derived_max_steps} (was {train_cfg.max_steps})"
            )
        train_cfg = replace(train_cfg, max_steps=derived_max_steps)
    if train_cfg.warmup_ratio is not None or train_cfg.decay_ratio is not None:
        ms = int(train_cfg.max_steps)
        warmup = (
            round(ms * float(train_cfg.warmup_ratio))
            if train_cfg.warmup_ratio is not None
            else int(train_cfg.warmup_steps)
        )
        decay = (
            round(ms * float(train_cfg.decay_ratio))
            if train_cfg.decay_ratio is not None
            else max(0, ms - warmup - int(train_cfg.stable_steps))
        )
        stable = max(0, ms - warmup - decay)
        if rank == 0:
            print(
                f"[lr_schedule_ratio] max_steps={ms} warmup={warmup} "
                f"stable={stable} decay={decay} "
                f"(warmup_ratio={train_cfg.warmup_ratio}, decay_ratio={train_cfg.decay_ratio})"
            )
        train_cfg = replace(train_cfg, warmup_steps=warmup, stable_steps=stable)
    valid_loader = None
    valid_sampler = None
    if valid_dataset is not None:
        if distributed:
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        valid_loader = TorchDataLoader(
            dataset=valid_dataset,
            shuffle=False,
            sampler=valid_sampler,
            drop_last=False,
            **dataloader_common_kwargs,
        )
        if len(valid_loader) == 0:
            raise ValueError(
                "Validation dataloader yielded zero batches. Decrease batch_size or valid_ratio."
            )

    has_validation = valid_loader is not None and train_cfg.valid_every > 0
    checkpoint_retention_enabled = train_cfg.checkpoint_best_n > 0
    periodic_checkpoint_keep = 0
    if checkpoint_retention_enabled:
        periodic_checkpoint_keep = (
            10_000 if has_validation else int(train_cfg.checkpoint_best_n) + 1
        )
    best_val_checkpoints: list[tuple[float, int, Path]] = []
    if is_main_process:
        if checkpoint_retention_enabled and has_validation:
            best_val_checkpoints = list_best_val_loss_checkpoints(output_dir)
            best_val_checkpoints = prune_best_val_loss_checkpoints(
                best_val_checkpoints,
                train_cfg.checkpoint_best_n,
            )
        if checkpoint_retention_enabled and has_validation:
            print(
                f"Checkpoint retention: periodic_keep={periodic_checkpoint_keep} + best_val_loss={train_cfg.checkpoint_best_n}."
            )
        elif checkpoint_retention_enabled:
            print(
                f"Checkpoint retention: validation disabled, keep latest {periodic_checkpoint_keep} periodic checkpoints."
            )
    return (
        best_val_checkpoints,
        checkpoint_retention_enabled,
        has_validation,
        loader,
        manifest_size,
        optim_steps_per_epoch,
        periodic_checkpoint_keep,
        run_name,
        speaker_name,
        train_cfg,
        train_dataset,
        train_sampler,
        valid_loader,
    )


def _build_model(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    args,
    device,
    distributed,
    exp_cfg,
    is_main_process,
    loader,
    local_rank,
    model_cfg,
    rank,
    resume_base_init,
    resume_model_cfg,
    resume_path,
    resume_text_encoder_config,
    train_cfg,
    train_dataset,
    train_sampler,
    world_size,
) -> tuple:
    if not (0.0 <= train_cfg.lora_dropout <= 1.0):
        raise ValueError(f"lora_dropout must be in [0, 1], got {train_cfg.lora_dropout}")
    if train_cfg.lora_r <= 0:
        raise ValueError(f"lora_r must be > 0, got {train_cfg.lora_r}")
    if train_cfg.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be > 0, got {train_cfg.lora_alpha}")

    if args.resume is not None:
        if train_config_uses_lora(train_cfg):
            if resume_path is None or not is_lora_adapter_dir(resume_path):
                raise ValueError("LoRA resume expects an adapter checkpoint directory.")
        elif resume_path is not None and resume_path.is_dir():
            raise ValueError(
                "Non-LoRA resume expects a .pt training checkpoint, not a checkpoint directory."
            )
        if args.init_checkpoint is not None and not train_config_uses_lora(train_cfg):
            raise ValueError(
                "--resume and --init-checkpoint can only be combined for LoRA adapter resumes."
            )

    if train_config_uses_lora(train_cfg) and args.resume is None and args.init_checkpoint is None:
        raise ValueError(
            "LoRA fine-tuning requires --init-checkpoint for the base model, "
            "or --resume from a LoRA adapter checkpoint directory."
        )

    preloaded_init_checkpoint = None
    pretrained_backbone_config = None
    load_pretrained_backbone_weights = True
    if model_cfg.use_pretrained_text_encoder:
        if args.resume is not None and not train_config_uses_lora(train_cfg):
            if isinstance(resume_model_cfg, dict):
                checkpoint_cfg = merge_dataclass_overrides(
                    ModelConfig(),
                    resume_model_cfg,
                    section="resume checkpoint model_config",
                )
                if checkpoint_cfg.use_pretrained_text_encoder:
                    load_pretrained_backbone_weights = False
                    pretrained_backbone_config = resume_text_encoder_config
        else:
            pretrained_base_path = None
            if args.init_checkpoint is not None:
                pretrained_base_path = _normalize_checkpoint_path(args.init_checkpoint)
            elif train_config_uses_lora(train_cfg) and isinstance(resume_base_init, dict):
                checkpoint_path = resume_base_init.get("checkpoint_path")
                if resume_base_init.get("mode") == "checkpoint" and isinstance(
                    checkpoint_path, str
                ):
                    pretrained_base_path = _normalize_checkpoint_path(checkpoint_path)
            if pretrained_base_path is not None:
                init_checkpoint_path = pretrained_base_path
                preloaded_init_checkpoint = _load_model_state_from_checkpoint(init_checkpoint_path)
                init_state = preloaded_init_checkpoint.model_state
                init_model_cfg = preloaded_init_checkpoint.model_config
                init_text_encoder_config = preloaded_init_checkpoint.text_encoder_config
                checkpoint_uses_pretrained = any(
                    key.startswith("pretrained_text_backbone.") for key in init_state
                )
                if isinstance(init_model_cfg, dict):
                    checkpoint_cfg = merge_dataclass_overrides(
                        ModelConfig(),
                        init_model_cfg,
                        section="init checkpoint model_config",
                    )
                    checkpoint_uses_pretrained = checkpoint_cfg.use_pretrained_text_encoder
                if checkpoint_uses_pretrained:
                    load_pretrained_backbone_weights = False
                    pretrained_backbone_config = init_text_encoder_config

    raw_model: torch.nn.Module = TextToLatentRFDiT(
        model_cfg,
        pretrained_backbone_config=pretrained_backbone_config,
        load_pretrained_backbone_weights=load_pretrained_backbone_weights,
    ).to(device)
    lora_wrapped = False
    base_init: dict | None = None
    if args.resume is not None and train_config_uses_lora(train_cfg):
        base_init = resume_base_init
        if args.init_checkpoint is not None:
            override_init_path = _normalize_checkpoint_path(args.init_checkpoint)
            base_init = {"mode": "checkpoint", "checkpoint_path": str(override_init_path)}
        _apply_base_initialization(
            raw_model,
            model_cfg=model_cfg,
            base_init=base_init,
            distributed=distributed,
            is_main_process=is_main_process,
            preloaded_checkpoint=preloaded_init_checkpoint,
        )
        if resume_path is None or not is_lora_adapter_dir(resume_path):
            raise ValueError("LoRA resume expects an adapter checkpoint directory.")
        raw_model = load_lora_adapter(raw_model, resume_path, is_trainable=True)
        lora_wrapped = True
    elif args.resume is None and args.init_checkpoint is None:
        _apply_base_initialization(
            raw_model,
            model_cfg=model_cfg,
            base_init=None,
            distributed=distributed,
            is_main_process=is_main_process,
        )
        if train_config_uses_lora(train_cfg):
            raw_model = apply_lora(raw_model, train_cfg)
            lora_wrapped = True
    elif args.init_checkpoint is not None:
        init_checkpoint_path = _normalize_checkpoint_path(args.init_checkpoint)
        base_init = {"mode": "checkpoint", "checkpoint_path": str(init_checkpoint_path)}
        _apply_base_initialization(
            raw_model,
            model_cfg=model_cfg,
            base_init=base_init,
            distributed=distributed,
            is_main_process=is_main_process,
            preloaded_checkpoint=preloaded_init_checkpoint,
        )
        if train_config_uses_lora(train_cfg) and not lora_wrapped:
            raw_model = apply_lora(raw_model, train_cfg)
            lora_wrapped = True

    if train_config_uses_lora(train_cfg) and is_main_process:
        trainable_params, total_params = count_parameters(raw_model)
        print(
            "LoRA enabled: "
            f"r={train_cfg.lora_r} alpha={train_cfg.lora_alpha} "
            f"dropout={train_cfg.lora_dropout:.3f} "
            f"target_modules={train_cfg.lora_target_modules!r} "
            f"modules_to_save={train_cfg.lora_modules_to_save!r} "
            f"trainable={trainable_params:,}/{total_params:,}"
        )
    if train_cfg.speaker_inversion_enabled:
        init_embedding = None
        if train_cfg.speaker_inversion_init_embedding is not None:
            init_payload = load_speaker_inversion_payload(
                train_cfg.speaker_inversion_init_embedding,
            )
            init_embedding = init_payload[SPEAKER_EMBEDDING_KEY]
            if is_main_process:
                print(
                    "Loaded Speaker Inversion init embedding: "
                    f"{train_cfg.speaker_inversion_init_embedding}"
                )
        speaker_inversion = raw_model.enable_speaker_inversion(
            num_tokens=train_cfg.speaker_inversion_tokens,
            init_std=train_cfg.speaker_inversion_init_std,
            init_embedding=init_embedding,
        )
        speaker_inversion.to(device)
        if is_main_process:
            print(
                "Speaker Inversion parameters initialized: "
                f"embedding={tuple(speaker_inversion.embedding.shape)}."
            )
    if train_cfg.train_mode == "duration_only":
        trainable_duration_params, frozen_params = freeze_for_duration_only(raw_model)
        if trainable_duration_params == 0:
            raise RuntimeError(
                "No duration predictor parameters were found for duration_only mode."
            )
        if is_main_process:
            print(
                "Duration-only training enabled: "
                f"trainable={trainable_duration_params:,} frozen={frozen_params:,}."
            )
    if train_cfg.speaker_inversion_enabled:
        trainable_speaker_params, frozen_params = freeze_for_speaker_inversion(raw_model)
        if trainable_speaker_params == 0:
            raise RuntimeError("No Speaker Inversion parameters were found.")
        if is_main_process:
            print(
                "Speaker Inversion freeze applied: "
                f"trainable={trainable_speaker_params:,} frozen={frozen_params:,}."
            )
    if train_cfg.gradient_checkpointing:
        raw_model.set_gradient_checkpointing(True)
        if is_main_process:
            scope = "diffusion blocks"
            if model_cfg.use_pretrained_text_encoder:
                scope += " and the pretrained text encoder (when supported)"
            print(f"Gradient checkpointing enabled on {scope}.")
    train_model = raw_model
    if train_cfg.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_model=True requires torch.compile (PyTorch 2+).")
        if is_main_process:
            print("torch.compile enabled (dynamic=True).")
        train_model = torch.compile(raw_model, dynamic=True)
    ddp_find_unused_parameters = bool(train_cfg.ddp_find_unused_parameters)
    ddp_find_unused_parameters_explicit = args.ddp_find_unused_parameters is not None or (
        isinstance(exp_cfg.get("train"), dict)
        and "ddp_find_unused_parameters" in exp_cfg.get("train", {})
    )
    if distributed:
        # Auto-enable for common configs where conditional branches can be fully
        # masked in a step. Without this, DDP can hang after step 1 due to
        # unreduced gradients in ranks where a branch is entirely unused.
        if not ddp_find_unused_parameters and not ddp_find_unused_parameters_explicit:
            speaker_labeled_count = train_dataset.speaker_labeled_count
            has_partial_or_no_speaker_labels = speaker_labeled_count < len(train_dataset)
            caption_labeled_count = train_dataset.caption_labeled_count
            has_partial_or_no_caption_labels = (
                model_cfg.use_caption_condition and caption_labeled_count < len(train_dataset)
            )
            has_stochastic_cond_drop = (
                train_cfg.text_condition_dropout > 0.0
                or train_cfg.speaker_condition_dropout > 0.0
                or (model_cfg.use_caption_condition and train_cfg.caption_condition_dropout > 0.0)
            )
            if (
                has_partial_or_no_speaker_labels
                or has_partial_or_no_caption_labels
                or has_stochastic_cond_drop
            ):
                ddp_find_unused_parameters = True
                if is_main_process:
                    print(
                        "DDP find_unused_parameters auto-enabled "
                        "(conditional branches may be fully masked in some steps)."
                    )
        model = DDP(
            train_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused_parameters,
            broadcast_buffers=False,
        )
    else:
        model = train_model
    optimizer = build_optimizer(raw_model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    if is_main_process:
        print(
            f"Optimizer={train_cfg.optimizer} Scheduler={train_cfg.lr_scheduler} lr={current_lr(optimizer):.3e}"
        )
        pretrained_lr = current_pretrained_text_encoder_lr(optimizer)
        if pretrained_lr is not None:
            print(f"Pretrained text encoder optimizer=adamw lr={pretrained_lr:.3e}.")
        if train_cfg.gradient_accumulation_steps > 1:
            print(
                f"Gradient accumulation enabled: steps={train_cfg.gradient_accumulation_steps} (effective global batch={train_cfg.batch_size * world_size * train_cfg.gradient_accumulation_steps})."
            )

    step = 0
    resume_epoch = 0
    resume_loader_state_loaded = False
    progress: TrainProgress | None = None
    resumed_es_best_val: float | None = None
    resumed_es_no_improve: int | None = None
    # Only the resume path fills these in, but they are returned either way.
    ckpt: dict | None = None
    dataloader_state: dict | None = None
    runtime_state: dict | None = None
    if args.resume is not None:
        ckpt = _load_checkpoint_payload(resume_path, map_location="cpu")
        if not train_config_uses_lora(train_cfg):
            raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt["step"])
        if scheduler is not None:
            scheduler_state = ckpt.get("scheduler")
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            elif step > 0:
                scheduler.last_step = step
        raw_best = ckpt.get("es_best_val")
        raw_no_improve = ckpt.get("es_no_improve")
        if raw_best is not None:
            resumed_es_best_val = float(raw_best)
        if raw_no_improve is not None:
            resumed_es_no_improve = int(raw_no_improve)
        runtime_state = ckpt.get(RUNTIME_STATE_KEY)
        if isinstance(runtime_state, dict):
            resume_epoch = int(runtime_state.get("sampler_epoch", 0))
        dataloader_state = _select_dataloader_state_for_rank(
            ckpt,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        if dataloader_state is not None:
            if train_sampler is not None:
                train_sampler.set_epoch(resume_epoch)
            loader.load_state_dict(dataloader_state)
            resume_loader_state_loaded = True
        if is_main_process:
            print(
                f"Resumed from step={step}"
                + (
                    f" es_best_val={resumed_es_best_val:.6f} es_no_improve={resumed_es_no_improve}"
                    if resumed_es_best_val is not None and resumed_es_no_improve is not None
                    else ""
                )
            )
            if dataloader_state is None:
                print(
                    "warning: resume checkpoint has no dataloader_state; "
                    "data iteration will restart at the beginning of an epoch."
                )
            else:
                print("Restored dataloader state for mid-epoch resume.")
    return (
        base_init,
        ckpt,
        dataloader_state,
        model,
        optimizer,
        progress,
        raw_model,
        resume_epoch,
        resume_loader_state_loaded,
        resumed_es_best_val,
        resumed_es_no_improve,
        runtime_state,
        scheduler,
        step,
    )


def _run_training_loop(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    _maybe_emit_samples,
    args,
    base_init,
    best_val_checkpoints,
    checkpoint_retention_enabled,
    ckpt,
    dataloader_state,
    device,
    distributed,
    has_validation,
    is_main_process,
    loader,
    manifest_size,
    model,
    model_cfg,
    optim_steps_per_epoch,
    optimizer,
    output_dir,
    periodic_checkpoint_keep,
    progress,
    rank,
    raw_model,
    resume_epoch,
    resume_loader_state_loaded,
    resumed_es_best_val,
    resumed_es_no_improve,
    run_name,
    run_uuid,
    runtime_state,
    sample_cfg,
    sampling_codec,
    scheduler,
    speaker_name,
    step,
    train_cfg,
    train_sampler,
    use_bf16,
    valid_loader,
    wandb_client,
    world_size,
) -> None:
    accum_steps = int(train_cfg.gradient_accumulation_steps)
    global_batch_size = train_cfg.batch_size * world_size * accum_steps
    duration_only = train_cfg.train_mode == "duration_only"
    caption_warmup_active = bool(
        train_cfg.caption_warmup
        and model_cfg.use_caption_condition
        and train_cfg.caption_warmup_steps > 0
        and step < train_cfg.caption_warmup_steps
    )
    pretrained_projector_warmup_active = bool(
        model_cfg.use_pretrained_text_encoder
        and train_cfg.pretrained_projector_warmup_steps > 0
        and step < train_cfg.pretrained_projector_warmup_steps
    )
    if caption_warmup_active and is_main_process:
        print(
            "Caption warmup active: non-caption gradients will be cleared for the first "
            f"{train_cfg.caption_warmup_steps} optimizer steps."
        )
    if pretrained_projector_warmup_active and is_main_process:
        print(
            "Pretrained projector warmup active: non-projector gradients will be cleared "
            f"through optimizer step {train_cfg.pretrained_projector_warmup_steps}."
        )

    es_enabled = bool(train_cfg.early_stop_enabled) and has_validation
    es_best_val: float = float("inf")
    es_no_improve: int = 0
    stop_early: bool = False
    if resumed_es_best_val is not None:
        es_best_val = resumed_es_best_val
    if resumed_es_no_improve is not None:
        es_no_improve = resumed_es_no_improve
    if es_enabled and is_main_process:
        print(
            "Early stopping enabled: "
            f"min_step={train_cfg.early_stop_min_step} "
            f"patience={train_cfg.early_stop_patience} "
            f"min_delta={train_cfg.early_stop_min_delta} "
            f"regression_ratio={train_cfg.early_stop_regression_ratio}"
        )
    if args.resume is not None and step >= train_cfg.max_steps:
        stop_early = True
        if is_main_process:
            print(
                f"resume: step={step} already >= max_steps={train_cfg.max_steps}; exiting without further training."
            )
    elif (
        args.resume is not None
        and es_enabled
        and step >= train_cfg.early_stop_min_step
        and es_no_improve >= train_cfg.early_stop_patience
    ):
        stop_early = True
        if is_main_process:
            print(
                f"resume: early-stop condition already met "
                f"(es_no_improve={es_no_improve} >= patience={train_cfg.early_stop_patience}); "
                f"exiting without further training."
            )

    try:
        model.train()
        if scheduler is not None and step == 0:
            # Ensure the very first optimizer step uses warmup-scaled LR.
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        accum_micro_steps = 0
        accum_loss = torch.zeros((), device=device, dtype=torch.float32)
        accum_rf_loss = torch.zeros((), device=device, dtype=torch.float32)
        accum_duration_loss = torch.zeros((), device=device, dtype=torch.float32)
        accum_duration_mae_frames = torch.zeros((), device=device, dtype=torch.float32)
        accum_duration_group_totals = torch.zeros(
            DURATION_CONDITION_GROUP_TOTAL_SIZE,
            device=device,
            dtype=torch.float64,
        )
        epoch = resume_epoch
        epoch_step_offset = int(
            ckpt.get(RUNTIME_STATE_KEY, {}).get("epoch_step", 0)
            if args.resume is not None
            and resume_loader_state_loaded
            and isinstance(ckpt.get(RUNTIME_STATE_KEY), dict)
            else 0
        )
        last_epoch_step = epoch_step_offset
        while step < train_cfg.max_steps and not stop_early:
            if train_sampler is not None and not resume_loader_state_loaded:
                train_sampler.set_epoch(epoch)
            epoch += 1
            current_epoch_step_offset = epoch_step_offset if resume_loader_state_loaded else 0
            epoch_step_offset = 0
            train_batches = cuda_prefetch_batches(
                loader,
                device=device,
                enabled=bool(train_cfg.dataloader_cuda_prefetch),
            )
            for raw_epoch_step, batch in enumerate(train_batches, start=1):
                epoch_step = raw_epoch_step + current_epoch_step_offset
                last_epoch_step = epoch_step
                resume_loader_state_loaded = False
                accum_micro_steps += 1
                text_ids = batch["text_ids"].to(device, non_blocking=True)
                text_mask = batch["text_mask"].to(device, non_blocking=True)
                caption_ids = None
                caption_mask = None
                has_caption = None
                if raw_model.cfg.use_caption_condition:
                    caption_ids = batch["caption_ids"].to(device, non_blocking=True)
                    caption_mask = batch["caption_mask"].to(device, non_blocking=True)
                    has_caption = batch["has_caption"].to(device, non_blocking=True)
                num_frames = batch["num_frames"].to(device, non_blocking=True)
                duration_features = batch["duration_features"].to(device, non_blocking=True)
                ref_latent = None
                ref_mask = None
                if raw_model.cfg.use_speaker_condition_resolved:
                    ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
                    ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
                    has_speaker = batch["has_speaker"].to(device, non_blocking=True)
                else:
                    has_speaker = None

                bsz = text_ids.shape[0]
                x_mask = None
                x_mask_valid = None
                x_t = None
                t = None
                v_target = None
                if not duration_only:
                    x0 = batch["latent_patched"].to(device, non_blocking=True)
                    x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
                    x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
                    if train_cfg.timestep_stratified:
                        t = sample_stratified_logit_normal_t(
                            batch_size=bsz,
                            device=device,
                            mean=train_cfg.timestep_logit_mean,
                            std=train_cfg.timestep_logit_std,
                            t_min=train_cfg.timestep_min,
                            t_max=train_cfg.timestep_max,
                        )
                    else:
                        t = sample_logit_normal_t(
                            batch_size=bsz,
                            device=device,
                            mean=train_cfg.timestep_logit_mean,
                            std=train_cfg.timestep_logit_std,
                            t_min=train_cfg.timestep_min,
                            t_max=train_cfg.timestep_max,
                        )
                    noise = torch.randn_like(x0)
                    x_t = rf_interpolate(x0, noise, t)
                    v_target = rf_velocity_target(x0, noise)

                text_cond_drop = torch.rand(bsz, device=device) < train_cfg.text_condition_dropout
                if text_cond_drop.any() and not raw_model.cfg.use_duration_predictor:
                    text_mask = text_mask.clone()
                    text_mask[text_cond_drop] = False
                caption_cond_drop = None
                caption_drop_for_model = None
                duration_has_caption = None
                if raw_model.cfg.use_caption_condition:
                    if has_caption is None or caption_mask is None:
                        raise RuntimeError(
                            "Caption conditioning is enabled but caption batch tensors are missing."
                        )
                    caption_cond_drop = (
                        torch.rand(bsz, device=device) < train_cfg.caption_condition_dropout
                    )
                    use_caption = has_caption & (~caption_cond_drop)
                    caption_drop_for_model = ~use_caption
                    duration_caption_drop = (
                        torch.rand(bsz, device=device) < train_cfg.duration_caption_dropout
                    )
                    duration_has_caption = has_caption & (~duration_caption_drop)
                    if not raw_model.cfg.use_duration_predictor:
                        caption_mask = caption_mask & use_caption[:, None]

                speaker_drop_for_model = None
                duration_has_speaker = None
                if raw_model.cfg.use_speaker_condition_resolved:
                    speaker_cond_drop = (
                        torch.rand(bsz, device=device) < train_cfg.speaker_condition_dropout
                    )
                    if train_cfg.speaker_inversion_enabled:
                        # Speaker Inversion learns one embedding for this run, so all samples are
                        # speaker-conditioned even when the manifest has no speaker_id.
                        use_speaker = ~speaker_cond_drop
                        speaker_drop_for_model = speaker_cond_drop
                    else:
                        use_speaker = has_speaker & (~speaker_cond_drop)
                        speaker_drop_for_model = ~use_speaker
                    duration_speaker_drop = (
                        torch.rand(bsz, device=device) < train_cfg.duration_speaker_dropout
                    )
                    if train_cfg.speaker_inversion_enabled:
                        duration_has_speaker = ~duration_speaker_drop
                    else:
                        duration_has_speaker = has_speaker & (~duration_speaker_drop)
                    duration_features = set_duration_has_speaker_feature(
                        duration_features,
                        duration_has_speaker,
                    )
                    if (
                        not raw_model.cfg.use_duration_predictor
                        and not train_cfg.speaker_inversion_enabled
                    ):
                        ref_mask = ref_mask & use_speaker[:, None]
                        ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)

                should_step = (accum_micro_steps % accum_steps) == 0
                sync_context = model.no_sync() if distributed and not should_step else nullcontext()
                with sync_context:
                    with (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if use_bf16
                        else nullcontext()
                    ):
                        if duration_only:
                            duration_pred = model(
                                x_t=None,
                                t=None,
                                text_input_ids=text_ids,
                                text_mask=text_mask,
                                ref_latent=ref_latent,
                                ref_mask=ref_mask,
                                caption_input_ids=caption_ids,
                                caption_mask=caption_mask,
                                latent_mask=None,
                                duration_features=duration_features,
                                duration_has_speaker=duration_has_speaker,
                                duration_has_caption=duration_has_caption,
                                duration_only=True,
                            )
                            v_pred = None
                        elif raw_model.cfg.use_duration_predictor:
                            v_pred, duration_pred = model(
                                x_t=x_t,
                                t=t,
                                text_input_ids=text_ids,
                                text_mask=text_mask,
                                ref_latent=ref_latent,
                                ref_mask=ref_mask,
                                caption_input_ids=caption_ids,
                                caption_mask=caption_mask,
                                latent_mask=x_mask,
                                text_condition_dropout=text_cond_drop,
                                speaker_condition_dropout=speaker_drop_for_model,
                                caption_condition_dropout=caption_drop_for_model,
                                duration_features=duration_features,
                                duration_has_speaker=duration_has_speaker,
                                duration_has_caption=duration_has_caption,
                                duration_backprop_to_condition=(
                                    train_cfg.duration_backprop_to_condition
                                ),
                            )
                        else:
                            v_pred = model(
                                x_t=x_t,
                                t=t,
                                text_input_ids=text_ids,
                                text_mask=text_mask,
                                ref_latent=ref_latent,
                                ref_mask=ref_mask,
                                caption_input_ids=caption_ids,
                                caption_mask=caption_mask,
                                latent_mask=x_mask,
                                text_condition_dropout=None,
                                speaker_condition_dropout=speaker_drop_for_model
                                if train_cfg.speaker_inversion_enabled
                                else None,
                                caption_condition_dropout=None,
                            )
                            duration_pred = None

                    rf_loss = torch.zeros((), device=device, dtype=torch.float32)
                    if not duration_only:
                        if (
                            v_pred is None
                            or v_target is None
                            or x_mask is None
                            or x_mask_valid is None
                        ):
                            raise RuntimeError("RF training tensors are missing.")
                        v_pred = v_pred.float()
                        rf_loss = compute_rf_loss(
                            pred=v_pred,
                            target=v_target.float(),
                            loss_mask=x_mask,
                            valid_mask=x_mask_valid,
                            mode=train_cfg.rf_loss_mode,
                        )
                    duration_loss = torch.zeros((), device=device, dtype=torch.float32)
                    duration_mae_frames = torch.zeros((), device=device, dtype=torch.float32)
                    duration_group_totals = torch.zeros(
                        DURATION_CONDITION_GROUP_TOTAL_SIZE,
                        device=device,
                        dtype=torch.float64,
                    )
                    if raw_model.cfg.use_duration_predictor:
                        if duration_pred is None:
                            raise RuntimeError(
                                "Duration predictor is enabled but duration_pred is missing."
                            )
                        duration_target = torch.log1p(num_frames.float())
                        duration_loss_per_sample = F.huber_loss(
                            duration_pred.float(),
                            duration_target,
                            delta=float(train_cfg.duration_huber_delta),
                            reduction="none",
                        )
                        duration_loss = duration_loss_per_sample.mean()
                        pred_frames = torch.expm1(duration_pred.float()).clamp_min(0.0)
                        duration_mae_frames = (pred_frames - num_frames.float()).abs().mean()
                        if duration_only:
                            duration_group_totals = duration_condition_group_totals(
                                duration_loss_per_sample=duration_loss_per_sample,
                                pred_frames=pred_frames,
                                target_frames=num_frames.float(),
                                has_speaker=has_speaker,
                                has_caption=has_caption,
                            )
                    if duration_only:
                        loss = duration_loss
                    else:
                        loss = rf_loss + (float(train_cfg.duration_loss_weight) * duration_loss)
                    (loss / float(accum_steps)).backward()
                    if pretrained_projector_warmup_active:
                        clear_non_pretrained_projector_grads(raw_model)
                    elif caption_warmup_active:
                        clear_non_caption_grads(raw_model)

                accum_loss += loss.detach()
                accum_rf_loss += rf_loss.detach()
                accum_duration_loss += duration_loss.detach()
                accum_duration_mae_frames += duration_mae_frames.detach()
                accum_duration_group_totals += duration_group_totals
                if not should_step:
                    continue

                step_loss = accum_loss / float(accum_steps)
                step_rf_loss = accum_rf_loss / float(accum_steps)
                step_duration_loss = accum_duration_loss / float(accum_steps)
                step_duration_mae_frames = accum_duration_mae_frames / float(accum_steps)
                step_duration_group_totals = accum_duration_group_totals.clone()
                accum_loss.zero_()
                accum_rf_loss.zero_()
                accum_duration_loss.zero_()
                accum_duration_mae_frames.zero_()
                accum_duration_group_totals.zero_()

                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                step += 1
                progress.update(step)
                if caption_warmup_active and step >= train_cfg.caption_warmup_steps:
                    caption_warmup_active = False
                    if is_main_process:
                        progress.write("caption warmup complete; all parameters are now updating.")
                if (
                    pretrained_projector_warmup_active
                    and step >= train_cfg.pretrained_projector_warmup_steps
                ):
                    pretrained_projector_warmup_active = False
                    if is_main_process:
                        progress.write(
                            "pretrained projector warmup complete; DiT, speaker, duration, and "
                            "the pretrained backbone are now updating."
                        )

                if step % train_cfg.log_every == 0:
                    loss_value = reduce_mean(step_loss, world_size, distributed).item()
                    rf_loss_value = reduce_mean(step_rf_loss, world_size, distributed).item()
                    duration_loss_value = reduce_mean(
                        step_duration_loss, world_size, distributed
                    ).item()
                    duration_mae_frames_value = reduce_mean(
                        step_duration_mae_frames, world_size, distributed
                    ).item()
                    duration_group_metrics: dict[str, float] = {}
                    if duration_only:
                        duration_group_totals = reduce_sum(
                            step_duration_group_totals,
                            distributed,
                        )
                        duration_group_metrics = duration_condition_group_metrics(
                            duration_group_totals
                        )
                    lr_value = current_lr(optimizer)
                    pretrained_lr_value = current_pretrained_text_encoder_lr(optimizer)
                    progress_metrics: dict[str, float] = {
                        "loss": loss_value,
                        "rf": rf_loss_value,
                        "lr": lr_value,
                    }
                    if pretrained_lr_value is not None:
                        progress_metrics["text_lr"] = pretrained_lr_value
                    if raw_model.cfg.use_duration_predictor:
                        progress_metrics["dur"] = duration_loss_value
                        progress_metrics["dur_mae"] = duration_mae_frames_value
                        if duration_only:
                            progress_metrics["dur_sp"] = duration_group_metrics[
                                "duration_loss_speaker"
                            ]
                            progress_metrics["dur_no_sp"] = duration_group_metrics[
                                "duration_loss_no_speaker"
                            ]
                            progress_metrics["dur_cap"] = duration_group_metrics[
                                "duration_loss_caption"
                            ]
                            progress_metrics["dur_no_cap"] = duration_group_metrics[
                                "duration_loss_no_caption"
                            ]
                    progress.log(
                        step=step,
                        epoch=epoch,
                        epoch_step=epoch_step,
                        epoch_total=len(loader),
                        metrics=progress_metrics,
                        global_batch_size=global_batch_size,
                    )
                    if is_main_process:
                        if raw_model.cfg.use_duration_predictor:
                            message = (
                                f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f} "
                                f"dur={duration_loss_value:.6f} "
                                f"dur_mae={duration_mae_frames_value:.2f}"
                            )
                            if duration_only:
                                group_suffix = duration_condition_group_log_suffix(
                                    duration_group_metrics
                                )
                                if group_suffix:
                                    message += f" {group_suffix}"
                            progress.write(f"{message} lr={lr_value:.3e}")
                        else:
                            progress.write(
                                f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f} "
                                f"lr={lr_value:.3e}"
                            )
                        metrics = {
                            "train/loss": loss_value,
                            "train/rf_loss": rf_loss_value,
                            "train/lr": lr_value,
                        }
                        if pretrained_lr_value is not None:
                            metrics["train/pretrained_text_encoder_lr"] = pretrained_lr_value
                        if raw_model.cfg.use_duration_predictor:
                            metrics["train/duration_loss"] = duration_loss_value
                            metrics["train/duration_mae_frames"] = duration_mae_frames_value
                            if duration_only:
                                metrics.update(
                                    duration_condition_group_wandb_metrics(
                                        "train",
                                        duration_group_metrics,
                                    )
                                )
                        wandb_client.log(metrics, step=step)

                if step % train_cfg.save_every == 0:
                    dataloader_state = _collect_dataloader_state(
                        loader,
                        distributed=distributed,
                        rank=rank,
                        world_size=world_size,
                    )
                    runtime_state = _runtime_state_for_checkpoint(
                        epoch=epoch,
                        epoch_step=epoch_step,
                    )
                    if is_main_process:
                        save_checkpoint(
                            _periodic_checkpoint_path(output_dir, step, train_cfg),
                            raw_model,
                            optimizer,
                            scheduler,
                            step,
                            model_cfg,
                            train_cfg,
                            base_init=base_init,
                            es_best_val=es_best_val,
                            es_no_improve=es_no_improve,
                            manifest_size=manifest_size,
                            run_uuid=run_uuid,
                            run_name=run_name,
                            speaker_name=speaker_name,
                            optim_steps_per_epoch=optim_steps_per_epoch,
                            dataloader_state=dataloader_state,
                            runtime_state=runtime_state,
                        )
                        enforce_periodic_checkpoint_limit(
                            output_dir=output_dir,
                            keep_count=periodic_checkpoint_keep,
                        )
                        if (
                            sample_cfg.enabled
                            and sample_cfg.every > 0
                            and step % sample_cfg.every == 0
                        ):
                            _maybe_emit_samples(step)

                if (
                    valid_loader is not None
                    and train_cfg.valid_every > 0
                    and step % train_cfg.valid_every == 0
                ):
                    valid_metrics = run_validation(
                        model=model,
                        loader=valid_loader,
                        train_cfg=train_cfg,
                        device=device,
                        use_bf16=use_bf16,
                        distributed=distributed,
                    )
                    best_dataloader_state = None
                    best_runtime_state = None
                    if checkpoint_retention_enabled:
                        best_dataloader_state = _collect_dataloader_state(
                            loader,
                            distributed=distributed,
                            rank=rank,
                            world_size=world_size,
                        )
                        best_runtime_state = _runtime_state_for_checkpoint(
                            epoch=epoch,
                            epoch_step=epoch_step,
                        )
                    if is_main_process:
                        if raw_model.cfg.use_duration_predictor:
                            message = (
                                "valid step={} loss={:.6f} rf={:.6f} dur={:.6f} dur_mae={:.2f}"
                            ).format(
                                step,
                                valid_metrics["loss"],
                                valid_metrics["rf_loss"],
                                valid_metrics["duration_loss"],
                                valid_metrics["duration_mae_frames"],
                            )
                            if duration_only:
                                group_suffix = duration_condition_group_log_suffix(valid_metrics)
                                if group_suffix:
                                    message += f" {group_suffix}"
                            progress.write(
                                "{} (samples={:.0f})".format(
                                    message,
                                    valid_metrics["num_samples"],
                                )
                            )
                        else:
                            progress.write(
                                ("valid step={} loss={:.6f} rf={:.6f} (samples={:.0f})").format(
                                    step,
                                    valid_metrics["loss"],
                                    valid_metrics["rf_loss"],
                                    valid_metrics["num_samples"],
                                )
                            )
                        metrics = {
                            "valid/loss": valid_metrics["loss"],
                            "valid/rf_loss": valid_metrics["rf_loss"],
                        }
                        if raw_model.cfg.use_duration_predictor:
                            metrics["valid/duration_loss"] = valid_metrics["duration_loss"]
                            metrics["valid/duration_mae_frames"] = valid_metrics[
                                "duration_mae_frames"
                            ]
                            if duration_only:
                                metrics.update(
                                    duration_condition_group_wandb_metrics(
                                        "valid",
                                        valid_metrics,
                                    )
                                )
                        wandb_client.log(metrics, step=step)
                        if es_enabled:
                            cur_val = float(valid_metrics["loss"])
                            if cur_val < es_best_val - train_cfg.early_stop_min_delta:
                                es_best_val = cur_val
                                es_no_improve = 0
                            else:
                                es_no_improve += 1
                            wandb_client.log(
                                {
                                    "es/no_improve": es_no_improve,
                                    "es/best_val": es_best_val,
                                },
                                step=step,
                            )
                            if step >= train_cfg.early_stop_min_step:
                                if es_no_improve >= train_cfg.early_stop_patience:
                                    progress.write(
                                        f"early stop: patience ({es_no_improve} "
                                        f">= {train_cfg.early_stop_patience}) at step={step} "
                                        f"best_val={es_best_val:.6f}"
                                    )
                                    stop_early = True
                                elif es_best_val < float("inf") and cur_val > es_best_val * (
                                    1.0 + train_cfg.early_stop_regression_ratio
                                ):
                                    progress.write(
                                        f"early stop: regression "
                                        f"({cur_val:.6f} > best {es_best_val:.6f} * "
                                        f"{1.0 + train_cfg.early_stop_regression_ratio:.2f}) "
                                        f"at step={step}"
                                    )
                                    stop_early = True

                        best_val_checkpoints, best_path = maybe_save_best_val_loss_checkpoint(
                            output_dir=output_dir,
                            checkpoints=best_val_checkpoints,
                            keep_best_n=train_cfg.checkpoint_best_n,
                            val_loss=float(valid_metrics["loss"]),
                            step=step,
                            model=raw_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            model_cfg=model_cfg,
                            train_cfg=train_cfg,
                            base_init=base_init,
                            es_best_val=es_best_val,
                            es_no_improve=es_no_improve,
                            manifest_size=manifest_size,
                            run_uuid=run_uuid,
                            run_name=run_name,
                            speaker_name=speaker_name,
                            optim_steps_per_epoch=optim_steps_per_epoch,
                            dataloader_state=best_dataloader_state,
                            runtime_state=best_runtime_state,
                        )
                        if best_path is not None:
                            progress.write(
                                "saved best val checkpoint: {} (loss={:.6f})".format(
                                    best_path.name,
                                    float(valid_metrics["loss"]),
                                )
                            )
                            if sample_cfg.enabled and sample_cfg.on_best_val:
                                _maybe_emit_samples(step)

                if step >= train_cfg.max_steps or stop_early:
                    break

        if (
            valid_loader is not None
            and train_cfg.valid_every > 0
            and step % train_cfg.valid_every != 0
        ):
            valid_metrics = run_validation(
                model=model,
                loader=valid_loader,
                train_cfg=train_cfg,
                device=device,
                use_bf16=use_bf16,
                distributed=distributed,
            )
            final_best_dataloader_state = None
            final_best_runtime_state = None
            if checkpoint_retention_enabled:
                final_best_dataloader_state = _collect_dataloader_state(
                    loader,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                )
                final_best_runtime_state = _runtime_state_for_checkpoint(
                    epoch=epoch,
                    epoch_step=last_epoch_step,
                )
            if is_main_process:
                if raw_model.cfg.use_duration_predictor:
                    message = (
                        "valid final step={} loss={:.6f} rf={:.6f} dur={:.6f} dur_mae={:.2f}"
                    ).format(
                        step,
                        valid_metrics["loss"],
                        valid_metrics["rf_loss"],
                        valid_metrics["duration_loss"],
                        valid_metrics["duration_mae_frames"],
                    )
                    if duration_only:
                        group_suffix = duration_condition_group_log_suffix(valid_metrics)
                        if group_suffix:
                            message += f" {group_suffix}"
                    progress.write(
                        "{} (samples={:.0f})".format(
                            message,
                            valid_metrics["num_samples"],
                        )
                    )
                else:
                    progress.write(
                        ("valid final step={} loss={:.6f} rf={:.6f} (samples={:.0f})").format(
                            step,
                            valid_metrics["loss"],
                            valid_metrics["rf_loss"],
                            valid_metrics["num_samples"],
                        )
                    )
                metrics = {
                    "valid/loss": valid_metrics["loss"],
                    "valid/rf_loss": valid_metrics["rf_loss"],
                }
                if raw_model.cfg.use_duration_predictor:
                    metrics["valid/duration_loss"] = valid_metrics["duration_loss"]
                    metrics["valid/duration_mae_frames"] = valid_metrics["duration_mae_frames"]
                    if duration_only:
                        metrics.update(
                            duration_condition_group_wandb_metrics(
                                "valid",
                                valid_metrics,
                            )
                        )
                wandb_client.log(metrics, step=step)
                best_val_checkpoints, best_path = maybe_save_best_val_loss_checkpoint(
                    output_dir=output_dir,
                    checkpoints=best_val_checkpoints,
                    keep_best_n=train_cfg.checkpoint_best_n,
                    val_loss=float(valid_metrics["loss"]),
                    step=step,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    base_init=base_init,
                    es_best_val=es_best_val,
                    es_no_improve=es_no_improve,
                    manifest_size=manifest_size,
                    run_uuid=run_uuid,
                    run_name=run_name,
                    speaker_name=speaker_name,
                    optim_steps_per_epoch=optim_steps_per_epoch,
                    dataloader_state=final_best_dataloader_state,
                    runtime_state=final_best_runtime_state,
                )
                if best_path is not None:
                    progress.write(
                        "saved best val checkpoint: {} (loss={:.6f})".format(
                            best_path.name,
                            float(valid_metrics["loss"]),
                        )
                    )
                    if sample_cfg.enabled and sample_cfg.on_best_val:
                        _maybe_emit_samples(step)

        final_dataloader_state = _collect_dataloader_state(
            loader,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        final_runtime_state = _runtime_state_for_checkpoint(
            epoch=epoch,
            epoch_step=last_epoch_step,
        )
        if is_main_process:
            save_checkpoint(
                _final_checkpoint_path(output_dir, train_cfg),
                raw_model,
                optimizer,
                scheduler,
                step,
                model_cfg,
                train_cfg,
                base_init=base_init,
                es_best_val=es_best_val,
                es_no_improve=es_no_improve,
                manifest_size=manifest_size,
                run_uuid=run_uuid,
                run_name=run_name,
                speaker_name=speaker_name,
                optim_steps_per_epoch=optim_steps_per_epoch,
                dataloader_state=final_dataloader_state,
                runtime_state=final_runtime_state,
            )
            if sample_cfg.enabled:
                _maybe_emit_samples(step)
            wandb_client.set_summary("train/final_step", step)
            progress.write(f"Training finished at step={step}.")
    finally:
        if progress is not None:
            progress.close()
        if sampling_codec is not None:
            del sampling_codec
        wandb_client.finish()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.resume is not None and Path(args.resume).suffix.lower() == ".safetensors":
        raise ValueError(
            "--resume expects a training checkpoint (.pt or LoRA checkpoint dir). "
            "Use --init-checkpoint for inference-only .safetensors weights."
        )

    rank, world_size, local_rank, distributed, device = setup_distributed(args.device)
    is_main_process = rank == 0

    (
        exp_cfg,
        model_cfg,
        output_dir,
        resume_base_init,
        resume_model_cfg,
        resume_path,
        resume_text_encoder_config,
        sample_cfg,
        train_cfg,
        use_bf16,
    ) = _resolve_configs(
        args=args,
        device=device,
        distributed=distributed,
        is_main_process=is_main_process,
        rank=rank,
    )
    if is_main_process and distributed:
        print(f"DDP enabled: world_size={world_size} (local_rank={local_rank})")
    (
        caption_tokenizer,
        run_uuid,
        tokenizer,
        wandb_client,
    ) = _setup_wandb_and_tokenizers(
        args=args,
        distributed=distributed,
        is_main_process=is_main_process,
        model_cfg=model_cfg,
        output_dir=output_dir,
        train_cfg=train_cfg,
    )
    (
        best_val_checkpoints,
        checkpoint_retention_enabled,
        has_validation,
        loader,
        manifest_size,
        optim_steps_per_epoch,
        periodic_checkpoint_keep,
        run_name,
        speaker_name,
        train_cfg,
        train_dataset,
        train_sampler,
        valid_loader,
    ) = _build_data(
        caption_tokenizer=caption_tokenizer,
        device=device,
        distributed=distributed,
        is_main_process=is_main_process,
        model_cfg=model_cfg,
        output_dir=output_dir,
        rank=rank,
        run_uuid=run_uuid,
        tokenizer=tokenizer,
        train_cfg=train_cfg,
        wandb_client=wandb_client,
        world_size=world_size,
    )

    (
        base_init,
        ckpt,
        dataloader_state,
        model,
        optimizer,
        progress,
        raw_model,
        resume_epoch,
        resume_loader_state_loaded,
        resumed_es_best_val,
        resumed_es_no_improve,
        runtime_state,
        scheduler,
        step,
    ) = _build_model(
        args=args,
        device=device,
        distributed=distributed,
        exp_cfg=exp_cfg,
        is_main_process=is_main_process,
        loader=loader,
        local_rank=local_rank,
        model_cfg=model_cfg,
        rank=rank,
        resume_base_init=resume_base_init,
        resume_model_cfg=resume_model_cfg,
        resume_path=resume_path,
        resume_text_encoder_config=resume_text_encoder_config,
        train_cfg=train_cfg,
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        world_size=world_size,
    )

    sampling_codec = None
    if sample_cfg.enabled and is_main_process and sample_cfg.prompts:
        from irodori_tts.training_samples import load_codec_for_sampling

        sampling_codec = load_codec_for_sampling(
            sample_cfg,
            expected_latent_dim=model_cfg.latent_dim,
        )
        print(
            f"Sample generation enabled: every={sample_cfg.every} prompts={len(sample_cfg.prompts)} "
            f"codec_device={sample_cfg.codec_device}"
        )
    elif sample_cfg.enabled and is_main_process and not sample_cfg.prompts:
        print("warning: sample_generation.enabled=true but prompts list is empty; disabling.")

    last_sampled_step: list[int] = [-1]

    def _maybe_emit_samples(current_step: int) -> None:
        if sampling_codec is None:
            return
        if current_step == last_sampled_step[0]:
            return
        last_sampled_step[0] = current_step
        from irodori_tts.training_samples import generate_training_samples

        generate_training_samples(
            raw_model=raw_model,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            sample_cfg=sample_cfg,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=sampling_codec,
            model_device=device,
            step=current_step,
            output_dir=output_dir,
            wandb_client=wandb_client,
            log_fn=lambda msg: progress.write(msg) if progress is not None else None,
        )

    progress = TrainProgress(
        max_steps=train_cfg.max_steps,
        start_step=step,
        rank=rank,
        world_size=world_size,
        enabled=train_cfg.progress,
        show_all_ranks=train_cfg.progress_all_ranks,
        description="Train Duration" if train_cfg.train_mode == "duration_only" else "Train RF",
    )
    _run_training_loop(
        _maybe_emit_samples=_maybe_emit_samples,
        args=args,
        base_init=base_init,
        best_val_checkpoints=best_val_checkpoints,
        checkpoint_retention_enabled=checkpoint_retention_enabled,
        ckpt=ckpt,
        dataloader_state=dataloader_state,
        device=device,
        distributed=distributed,
        has_validation=has_validation,
        is_main_process=is_main_process,
        loader=loader,
        manifest_size=manifest_size,
        model=model,
        model_cfg=model_cfg,
        optim_steps_per_epoch=optim_steps_per_epoch,
        optimizer=optimizer,
        output_dir=output_dir,
        periodic_checkpoint_keep=periodic_checkpoint_keep,
        progress=progress,
        rank=rank,
        raw_model=raw_model,
        resume_epoch=resume_epoch,
        resume_loader_state_loaded=resume_loader_state_loaded,
        resumed_es_best_val=resumed_es_best_val,
        resumed_es_no_improve=resumed_es_no_improve,
        run_name=run_name,
        run_uuid=run_uuid,
        runtime_state=runtime_state,
        sample_cfg=sample_cfg,
        sampling_codec=sampling_codec,
        scheduler=scheduler,
        speaker_name=speaker_name,
        step=step,
        train_cfg=train_cfg,
        train_sampler=train_sampler,
        use_bf16=use_bf16,
        valid_loader=valid_loader,
        wandb_client=wandb_client,
        world_size=world_size,
    )


if __name__ == "__main__":
    main()
