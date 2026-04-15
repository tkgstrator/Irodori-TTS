#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import uuid as _uuid
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from irodori_tts.config import (
    ModelConfig,
    TrainConfig,
    dump_configs,
    load_experiment_yaml,
    merge_dataclass_overrides,
    merge_sample_generation_overrides,
)
from irodori_tts.dataset import LatentTextDataset, TTSCollator
from irodori_tts.lora import (
    LORA_METADATA_NAME,
    LORA_TARGET_PRESETS,
    LORA_TRAIN_CONFIG_FIELDS,
    LORA_TRAINER_STATE_NAME,
    apply_lora,
    count_parameters,
    is_lora_adapter_dir,
    load_lora_adapter,
    train_config_uses_lora,
)
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.optim import build_optimizer, build_scheduler, current_lr
from irodori_tts.progress import TrainProgress
from irodori_tts.rf import (
    rf_interpolate,
    rf_velocity_target,
    sample_logit_normal_t,
    sample_stratified_logit_normal_t,
)
from irodori_tts.tokenizer import PretrainedTextTokenizer

WANDB_MODES = {"online", "offline", "disabled"}
CHECKPOINT_STEP_RE = re.compile(r"^checkpoint_(\d+)(?:\.pt)?$")
CHECKPOINT_BEST_VAL_LOSS_RE = re.compile(
    r"^checkpoint_best_val_loss_(\d+)_(-?\d+(?:\.\d+)?)(?:\.pt)?$"
)
SAFETENSORS_CONFIG_META_KEY = "config_json"
SAFETENSORS_INFERENCE_CONFIG_KEYS = {"max_text_len", "max_caption_len", "fixed_target_latent_steps"}

VALID_MIN_COUNT = 50
VALID_MAX_COUNT = 100


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def echo_style_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Echo/JAX-style diffusion loss:
    - take mean squared error over loss_masked tokens
    - divide by mean valid-token ratio (short samples get up-weighted)

    If loss_mask == valid_mask, this reduces to standard masked MSE.
    """
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1)  # (B, S)
    loss_weight = loss_mask.float()
    valid_weight = valid_mask.float()

    # Keep normalization stable for degenerate samples with no valid target tokens.
    has_valid = (valid_weight.sum(dim=-1) > 0).float()[:, None]
    denom = (loss_weight * valid_weight * has_valid).mean().clamp_min(1e-6)
    return (diff * loss_weight).mean() / denom


def _resolve_speaker_name(manifest_path: str | Path | None) -> str | None:
    if manifest_path is None:
        return None
    p = Path(manifest_path)
    speaker_dir = p.parent
    cfg_path = speaker_dir / "config.yaml"
    speaker_id = speaker_dir.name or None
    if not cfg_path.is_file():
        return speaker_id
    try:
        import yaml
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        speaker = (data.get("speaker") or {}) if isinstance(data, dict) else {}
        return speaker.get("label") or speaker.get("name") or speaker.get("id") or speaker_id
    except Exception:
        return speaker_id


def _build_lora_safetensors_metadata(
    *,
    run_uuid: str | None,
    run_name: str | None,
    speaker_name: str | None,
    base_model: str | None,
    step: int,
    optim_steps_per_epoch: int | None,
    train_cfg: TrainConfig,
    val_loss: float | None,
) -> dict[str, str]:
    meta: dict[str, str] = {}
    if run_uuid:
        meta["uuid"] = str(run_uuid)
    if run_name:
        meta["model_name"] = str(run_name)
    if speaker_name:
        meta["speaker"] = str(speaker_name)
    if base_model:
        meta["base_model"] = str(base_model)
    meta["step"] = str(int(step))
    if optim_steps_per_epoch and optim_steps_per_epoch > 0:
        meta["epoch"] = str(int(step) // int(optim_steps_per_epoch))
    if val_loss is not None:
        meta["val_loss"] = f"{float(val_loss):.6f}"
    meta["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["lora_r"] = str(int(train_cfg.lora_r))
    meta["lora_alpha"] = str(int(train_cfg.lora_alpha))
    meta["lora_dropout"] = f"{float(train_cfg.lora_dropout):.6f}"
    meta["lora_target_modules"] = str(train_cfg.lora_target_modules)
    return meta


def _inject_safetensors_metadata(adapter_path: Path, extra_metadata: dict[str, str]) -> None:
    """Re-save adapter_model.safetensors with merged __metadata__."""
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        return
    if not adapter_path.is_file():
        return
    tensors: dict[str, torch.Tensor] = {}
    existing_meta: dict[str, str] = {}
    with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
        existing_meta = dict(f.metadata() or {})
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    merged = {**existing_meta, **{k: v for k, v in extra_metadata.items() if v is not None}}
    save_file(tensors, str(adapter_path), metadata=merged)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    base_init: dict | None = None,
    es_best_val: float | None = None,
    es_no_improve: int | None = None,
    manifest_size: int | None = None,
    run_uuid: str | None = None,
    run_name: str | None = None,
    speaker_name: str | None = None,
    optim_steps_per_epoch: int | None = None,
    val_loss: float | None = None,
) -> None:
    path = Path(path)
    es_state = {
        "es_best_val": float(es_best_val) if es_best_val is not None else None,
        "es_no_improve": int(es_no_improve) if es_no_improve is not None else None,
    }
    manifest_meta = {"manifest_size": int(manifest_size) if manifest_size is not None else None}
    if train_config_uses_lora(train_cfg):
        if path.exists():
            _safe_unlink(path)
        path.mkdir(parents=True, exist_ok=True)
        if not hasattr(model, "save_pretrained"):
            raise RuntimeError(
                "LoRA checkpoint saving requires a PEFT model with save_pretrained()."
            )
        model.save_pretrained(path)
        adapter_safetensors = path / "adapter_model.safetensors"
        if adapter_safetensors.is_file():
            base_model_str: str | None = None
            if base_init is not None:
                base_model_str = base_init.get("checkpoint_path")
            extra_meta = _build_lora_safetensors_metadata(
                run_uuid=run_uuid,
                run_name=run_name,
                speaker_name=speaker_name,
                base_model=base_model_str,
                step=step,
                optim_steps_per_epoch=optim_steps_per_epoch,
                train_cfg=train_cfg,
                val_loss=val_loss,
            )
            _inject_safetensors_metadata(adapter_safetensors, extra_meta)
        dump_configs(path / "config.json", model_cfg, train_cfg)
        (path / LORA_METADATA_NAME).write_text(
            json.dumps({"base_init": base_init}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if manifest_size is not None:
            (path / "manifest_size.txt").write_text(f"{int(manifest_size)}\n", encoding="utf-8")
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "model_config": asdict(model_cfg),
                "train_config": asdict(train_cfg),
                "base_init": base_init,
                **es_state,
                **manifest_meta,
            },
            path / LORA_TRAINER_STATE_NAME,
        )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            **es_state,
            **manifest_meta,
        },
        path,
    )


def _safe_unlink(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        return


def list_periodic_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint_*"):
        match = CHECKPOINT_STEP_RE.match(path.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    return checkpoints


def enforce_periodic_checkpoint_limit(output_dir: Path, keep_count: int) -> None:
    if keep_count <= 0:
        return
    checkpoints = list_periodic_checkpoints(output_dir)
    for _, stale_path in checkpoints[keep_count:]:
        _safe_unlink(stale_path)


def list_best_val_loss_checkpoints(output_dir: Path) -> list[tuple[float, int, Path]]:
    checkpoints: list[tuple[float, int, Path]] = []
    for path in output_dir.glob("checkpoint_best_val_loss_*"):
        match = CHECKPOINT_BEST_VAL_LOSS_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        score = float(match.group(2))
        checkpoints.append((score, step, path))
    checkpoints.sort(key=lambda item: (item[0], item[1]))
    return checkpoints


def prune_best_val_loss_checkpoints(
    checkpoints: list[tuple[float, int, Path]],
    keep_best_n: int,
) -> list[tuple[float, int, Path]]:
    if keep_best_n <= 0:
        return checkpoints
    checkpoints = sorted(checkpoints, key=lambda item: (item[0], item[1]))
    while len(checkpoints) > keep_best_n:
        _, _, stale_path = checkpoints.pop()
        _safe_unlink(stale_path)
    return checkpoints


def _upload_best_checkpoint_artifact(
    *,
    wandb_run,
    path: Path,
    step: int,
    val_loss: float,
) -> None:
    if wandb_run is None:
        return
    try:
        import wandb
    except ImportError:
        return
    try:
        artifact_name = f"lora-best-{wandb_run.name}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata={"step": int(step), "val_loss": float(val_loss)},
        )
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        wandb_run.log_artifact(artifact, aliases=["latest", "best"])
    except Exception as exc:  # pragma: no cover - best-effort upload
        print(f"warning: failed to upload best checkpoint artifact: {exc}")


def maybe_save_best_val_loss_checkpoint(
    *,
    output_dir: Path,
    checkpoints: list[tuple[float, int, Path]],
    keep_best_n: int,
    val_loss: float,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    base_init: dict | None,
    es_best_val: float | None = None,
    es_no_improve: int | None = None,
    manifest_size: int | None = None,
    run_uuid: str | None = None,
    run_name: str | None = None,
    speaker_name: str | None = None,
    optim_steps_per_epoch: int | None = None,
) -> tuple[list[tuple[float, int, Path]], Path | None]:
    if keep_best_n <= 0:
        return checkpoints, None

    checkpoints = sorted(checkpoints, key=lambda item: (item[0], item[1]))
    if len(checkpoints) >= keep_best_n:
        worst_score = checkpoints[-1][0]
        if val_loss >= worst_score:
            return checkpoints, None

    kept: list[tuple[float, int, Path]] = []
    for score, saved_step, path in checkpoints:
        if saved_step == step:
            _safe_unlink(path)
            continue
        kept.append((score, saved_step, path))
    checkpoints = kept

    path = _best_checkpoint_path(output_dir, step=step, val_loss=val_loss, train_cfg=train_cfg)
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
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
        val_loss=float(val_loss),
    )
    checkpoints.append((float(val_loss), int(step), path))
    checkpoints = prune_best_val_loss_checkpoints(checkpoints, keep_best_n)
    return checkpoints, path


def cli_provided(argv: list[str], flag: str) -> bool:
    return any(x == flag or x.startswith(flag + "=") for x in argv)


def _periodic_checkpoint_path(output_dir: Path, step: int, train_cfg: TrainConfig) -> Path:
    if train_config_uses_lora(train_cfg):
        return output_dir / f"checkpoint_{step:07d}"
    return output_dir / f"checkpoint_{step:07d}.pt"


def _best_checkpoint_path(
    output_dir: Path, *, step: int, val_loss: float, train_cfg: TrainConfig
) -> Path:
    if train_config_uses_lora(train_cfg):
        return output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}"
    return output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}.pt"


def _final_checkpoint_path(output_dir: Path, train_cfg: TrainConfig) -> Path:
    if train_config_uses_lora(train_cfg):
        return output_dir / "checkpoint_final"
    return output_dir / "checkpoint_final.pt"


def build_condition_tokenizer(
    *,
    repo_id: str,
    add_bos: bool,
    vocab_size: int,
    local_files_only: bool = False,
) -> PretrainedTextTokenizer:
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=repo_id,
        add_bos=bool(add_bos),
        local_files_only=local_files_only,
    )
    if tokenizer.vocab_size != vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size mismatch: expected {vocab_size} but tokenizer "
            f"({repo_id}) vocab_size={tokenizer.vocab_size}."
        )
    return tokenizer


def build_text_tokenizer(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> PretrainedTextTokenizer:
    return build_condition_tokenizer(
        repo_id=model_cfg.text_tokenizer_repo,
        add_bos=bool(model_cfg.text_add_bos),
        vocab_size=int(model_cfg.text_vocab_size),
        local_files_only=local_files_only,
    )


def build_caption_tokenizer(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> PretrainedTextTokenizer:
    return build_condition_tokenizer(
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        add_bos=model_cfg.caption_add_bos_resolved,
        vocab_size=model_cfg.caption_vocab_size_resolved,
        local_files_only=local_files_only,
    )


def validate_pretrained_backbone_dim(
    *,
    repo_id: str,
    expected_dim: int,
    local_files_only: bool = False,
) -> int:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for pretrained text embedding initialization. "
            "Install with `pip install transformers sentencepiece`."
        ) from exc

    text_cfg = AutoConfig.from_pretrained(
        repo_id,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    hidden_size = getattr(text_cfg, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Could not read hidden_size from pretrained config: {repo_id}")
    hidden_size = int(hidden_size)
    if hidden_size != expected_dim:
        raise ValueError(
            f"Condition encoder dim mismatch: expected {expected_dim} but pretrained hidden_size={hidden_size} "
            f"for repo {repo_id}."
        )
    return hidden_size


def validate_text_backbone_dim(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> int:
    return validate_pretrained_backbone_dim(
        repo_id=model_cfg.text_tokenizer_repo,
        expected_dim=int(model_cfg.text_dim),
        local_files_only=local_files_only,
    )


def validate_caption_backbone_dim(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> int:
    return validate_pretrained_backbone_dim(
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        expected_dim=model_cfg.caption_dim_resolved,
        local_files_only=local_files_only,
    )


def initialize_embedding_from_pretrained(
    embedding: torch.nn.Embedding,
    *,
    repo_id: str,
    local_files_only: bool = False,
) -> None:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for pretrained text embedding initialization. "
            "Install with `pip install transformers sentencepiece`."
        ) from exc

    text_backbone = AutoModel.from_pretrained(
        repo_id,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    pretrained_embedding = text_backbone.get_input_embeddings()
    if pretrained_embedding is None:
        raise ValueError(f"Pretrained model has no input embeddings: {repo_id}")
    src_weight = pretrained_embedding.weight.detach().to(device="cpu", dtype=torch.float32)
    tgt_weight = embedding.weight
    src_vocab, src_dim = tuple(src_weight.shape)
    tgt_vocab, tgt_dim = tuple(tgt_weight.shape)
    if src_dim != tgt_dim:
        raise ValueError(
            f"Embedding hidden size mismatch: pretrained={src_dim} model={tgt_dim} for repo={repo_id}."
        )

    copy_rows = min(src_vocab, tgt_vocab)
    with torch.no_grad():
        tgt_weight[:copy_rows].copy_(
            src_weight[:copy_rows].to(device=tgt_weight.device, dtype=tgt_weight.dtype)
        )

    del text_backbone


def initialize_text_embedding_from_pretrained(
    model: TextToLatentRFDiT,
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> None:
    initialize_embedding_from_pretrained(
        model.text_encoder.text_embedding,
        repo_id=model_cfg.text_tokenizer_repo,
        local_files_only=local_files_only,
    )


def initialize_caption_embedding_from_pretrained(
    model: TextToLatentRFDiT,
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> None:
    if model.caption_encoder is None:
        raise RuntimeError(
            "Caption embedding initialization requested but caption encoder is absent."
        )
    initialize_embedding_from_pretrained(
        model.caption_encoder.text_embedding,
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        local_files_only=local_files_only,
    )


def _load_model_state_from_checkpoint(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict | None, dict | None]:
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file as load_safetensors_file

        checkpoint_model_cfg = None
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        config_json = metadata.get(SAFETENSORS_CONFIG_META_KEY)
        if config_json:
            parsed = json.loads(config_json)
            if isinstance(parsed, dict):
                checkpoint_model_cfg = {
                    key: value
                    for key, value in parsed.items()
                    if key not in SAFETENSORS_INFERENCE_CONFIG_KEYS
                }
        return load_safetensors_file(str(path), device="cpu"), checkpoint_model_cfg, None

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")

    raw_model = payload.get("model")
    if raw_model is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
        raw_model = payload
    if not isinstance(raw_model, dict):
        raise ValueError(f"Checkpoint does not contain a model state dictionary: {path}")

    checkpoint_model_cfg = payload.get("model_config")
    if checkpoint_model_cfg is not None and not isinstance(checkpoint_model_cfg, dict):
        raise ValueError(f"Checkpoint model_config must be a dictionary when present: {path}")
    checkpoint_train_cfg = payload.get("train_config")
    if checkpoint_train_cfg is not None and not isinstance(checkpoint_train_cfg, dict):
        raise ValueError(f"Checkpoint train_config must be a dictionary when present: {path}")
    return raw_model, checkpoint_model_cfg, checkpoint_train_cfg


def _check_model_config_compatibility(
    checkpoint_path: Path,
    checkpoint_model_cfg: dict | None,
    current_model_cfg: ModelConfig,
    *,
    require_caption_match: bool,
) -> None:
    if checkpoint_model_cfg is None:
        return

    checkpoint_cfg = merge_dataclass_overrides(
        ModelConfig(),
        checkpoint_model_cfg,
        section="checkpoint model_config",
    )

    comparisons: list[tuple[str, object, object]] = [
        ("latent_dim", checkpoint_cfg.latent_dim, current_model_cfg.latent_dim),
        (
            "latent_patch_size",
            checkpoint_cfg.latent_patch_size,
            current_model_cfg.latent_patch_size,
        ),
        ("model_dim", checkpoint_cfg.model_dim, current_model_cfg.model_dim),
        ("num_layers", checkpoint_cfg.num_layers, current_model_cfg.num_layers),
        ("num_heads", checkpoint_cfg.num_heads, current_model_cfg.num_heads),
        ("mlp_ratio", checkpoint_cfg.mlp_ratio, current_model_cfg.mlp_ratio),
        ("text_vocab_size", checkpoint_cfg.text_vocab_size, current_model_cfg.text_vocab_size),
        ("text_dim", checkpoint_cfg.text_dim, current_model_cfg.text_dim),
        ("text_layers", checkpoint_cfg.text_layers, current_model_cfg.text_layers),
        ("text_heads", checkpoint_cfg.text_heads, current_model_cfg.text_heads),
        (
            "text_mlp_ratio",
            checkpoint_cfg.text_mlp_ratio_resolved,
            current_model_cfg.text_mlp_ratio_resolved,
        ),
        ("adaln_rank", checkpoint_cfg.adaln_rank, current_model_cfg.adaln_rank),
    ]
    if checkpoint_cfg.use_speaker_condition and current_model_cfg.use_speaker_condition:
        comparisons.extend(
            [
                ("speaker_dim", checkpoint_cfg.speaker_dim, current_model_cfg.speaker_dim),
                ("speaker_layers", checkpoint_cfg.speaker_layers, current_model_cfg.speaker_layers),
                ("speaker_heads", checkpoint_cfg.speaker_heads, current_model_cfg.speaker_heads),
                (
                    "speaker_mlp_ratio",
                    checkpoint_cfg.speaker_mlp_ratio_resolved,
                    current_model_cfg.speaker_mlp_ratio_resolved,
                ),
                (
                    "speaker_patch_size",
                    checkpoint_cfg.speaker_patch_size,
                    current_model_cfg.speaker_patch_size,
                ),
            ]
        )
    if require_caption_match:
        comparisons.extend(
            [
                (
                    "use_caption_condition",
                    checkpoint_cfg.use_caption_condition,
                    current_model_cfg.use_caption_condition,
                ),
                (
                    "use_speaker_condition",
                    checkpoint_cfg.use_speaker_condition,
                    current_model_cfg.use_speaker_condition,
                ),
                (
                    "caption_vocab_size",
                    checkpoint_cfg.caption_vocab_size_resolved,
                    current_model_cfg.caption_vocab_size_resolved,
                ),
                (
                    "caption_tokenizer_repo",
                    checkpoint_cfg.caption_tokenizer_repo_resolved,
                    current_model_cfg.caption_tokenizer_repo_resolved,
                ),
                (
                    "caption_add_bos",
                    checkpoint_cfg.caption_add_bos_resolved,
                    current_model_cfg.caption_add_bos_resolved,
                ),
                (
                    "caption_dim",
                    checkpoint_cfg.caption_dim_resolved,
                    current_model_cfg.caption_dim_resolved,
                ),
                (
                    "caption_layers",
                    checkpoint_cfg.caption_layers_resolved,
                    current_model_cfg.caption_layers_resolved,
                ),
                (
                    "caption_heads",
                    checkpoint_cfg.caption_heads_resolved,
                    current_model_cfg.caption_heads_resolved,
                ),
                (
                    "caption_mlp_ratio",
                    checkpoint_cfg.caption_mlp_ratio_resolved,
                    current_model_cfg.caption_mlp_ratio_resolved,
                ),
            ]
        )

    for key, checkpoint_value, current_value in comparisons:
        if checkpoint_value != current_value:
            raise ValueError(
                f"Checkpoint/config mismatch for '{key}': checkpoint={checkpoint_value} "
                f"current={current_value} ({checkpoint_path})"
            )


def checkpoint_uses_caption_condition(
    checkpoint_model_cfg: dict | None,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    if checkpoint_model_cfg is not None:
        checkpoint_cfg = merge_dataclass_overrides(
            ModelConfig(),
            checkpoint_model_cfg,
            section="checkpoint model_config",
        )
        if checkpoint_cfg.use_caption_condition:
            return True
    return any(
        key.startswith("caption_encoder.")
        or key.startswith("caption_norm.")
        or ".wk_caption." in key
        or ".wv_caption." in key
        for key in state_dict
    )


def load_model_state_partially(
    model: TextToLatentRFDiT,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    skipped_extra: list[str] = []

    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            skipped_extra.append(key)
            continue
        if tuple(target.shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        filtered_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    if unexpected_keys:
        skipped_extra.extend(unexpected_keys)
    return missing_keys, skipped_shape, skipped_extra


def _canonical_parameter_key(key: str) -> str:
    prefix = "base_model.model."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def is_caption_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return (
        key.startswith("caption_encoder.")
        or key.startswith("caption_norm.")
        or ".wk_caption." in key
        or ".wv_caption." in key
    )


def is_speaker_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return (
        key.startswith("speaker_encoder.")
        or key.startswith("speaker_norm.")
        or ".wk_speaker." in key
        or ".wv_speaker." in key
    )


def clear_non_caption_grads(model: TextToLatentRFDiT) -> tuple[int, int]:
    caption_grad_params = 0
    cleared_grad_params = 0
    for key, param in model.named_parameters():
        if is_caption_only_parameter(key):
            if param.grad is not None:
                caption_grad_params += 1
            continue
        if param.grad is not None:
            cleared_grad_params += 1
        param.grad = None
    return caption_grad_params, cleared_grad_params


def validate_caption_upgrade_partial_load(
    checkpoint_path: Path,
    missing_keys: list[str],
    skipped_shape: list[str],
    skipped_extra: list[str],
) -> None:
    if skipped_shape:
        raise ValueError(
            "Checkpoint/config shape mismatch while upgrading caption conditioning: "
            f"{checkpoint_path} skipped_shape={skipped_shape[:8]}"
        )
    non_speaker_extra = [key for key in skipped_extra if not is_speaker_only_parameter(key)]
    if non_speaker_extra:
        raise ValueError(
            "Unexpected checkpoint keys while upgrading caption conditioning: "
            f"{checkpoint_path} skipped_extra={non_speaker_extra[:8]}"
        )
    non_caption_missing = [key for key in missing_keys if not is_caption_only_parameter(key)]
    if non_caption_missing:
        raise ValueError(
            "Partial init from caption-free checkpoint left non-caption parameters missing: "
            f"{checkpoint_path} missing={non_caption_missing[:8]}"
        )


def _load_checkpoint_payload(path: str | Path, *, map_location) -> dict:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        state_path = checkpoint_path / LORA_TRAINER_STATE_NAME
        payload = torch.load(state_path, map_location=map_location, weights_only=True)
    else:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")
    return payload


def _normalize_checkpoint_path(path: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))


def _lora_field_cli_explicit(field: str, args: argparse.Namespace, raw_argv: list[str]) -> bool:
    if field == "lora_enabled":
        return args.lora_enabled is not None
    flag = "--" + field.replace("_", "-")
    return cli_provided(raw_argv, flag)


def _restore_resume_lora_config(
    train_cfg: TrainConfig,
    *,
    resume_train_cfg: dict | None,
    args: argparse.Namespace,
    raw_argv: list[str],
    exp_cfg: dict,
) -> TrainConfig:
    if not isinstance(resume_train_cfg, dict):
        return train_cfg

    train_overrides = exp_cfg.get("train", {})
    if not isinstance(train_overrides, dict):
        train_overrides = {}

    updates: dict[str, object] = {}
    for field in LORA_TRAIN_CONFIG_FIELDS:
        if field not in resume_train_cfg:
            continue
        explicit = _lora_field_cli_explicit(field, args, raw_argv) or field in train_overrides
        current_value = getattr(train_cfg, field)
        resume_value = resume_train_cfg[field]
        if explicit:
            if current_value != resume_value:
                raise ValueError(
                    f"Resume checkpoint expects train.{field}={resume_value!r}, "
                    f"but current config requests {current_value!r}."
                )
            continue
        updates[field] = resume_value

    if updates:
        train_cfg = replace(train_cfg, **updates)
    return train_cfg


def _initialize_base_model_from_pretrained_embeddings(
    raw_model: torch.nn.Module,
    *,
    model_cfg: ModelConfig,
    distributed: bool,
    is_main_process: bool,
) -> None:
    if distributed:
        if is_main_process:
            print(
                f"Initializing text embedding from pretrained model: {model_cfg.text_tokenizer_repo}"
            )
            initialize_text_embedding_from_pretrained(
                raw_model,
                model_cfg,
                local_files_only=False,
            )
            if model_cfg.use_caption_condition:
                print(
                    "Initializing caption embedding from pretrained model: "
                    f"{model_cfg.caption_tokenizer_repo_resolved}"
                )
                initialize_caption_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=False,
                )
        dist.barrier()
        if not is_main_process:
            initialize_text_embedding_from_pretrained(
                raw_model,
                model_cfg,
                local_files_only=True,
            )
            if model_cfg.use_caption_condition:
                initialize_caption_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=True,
                )
        dist.barrier()
        return

    if is_main_process:
        print(f"Initializing text embedding from pretrained model: {model_cfg.text_tokenizer_repo}")
    initialize_text_embedding_from_pretrained(
        raw_model,
        model_cfg,
        local_files_only=False,
    )
    if model_cfg.use_caption_condition:
        if is_main_process:
            print(
                "Initializing caption embedding from pretrained model: "
                f"{model_cfg.caption_tokenizer_repo_resolved}"
            )
        initialize_caption_embedding_from_pretrained(
            raw_model,
            model_cfg,
            local_files_only=False,
        )


def _apply_base_initialization(
    raw_model: torch.nn.Module,
    *,
    model_cfg: ModelConfig,
    base_init: dict | None,
    distributed: bool,
    is_main_process: bool,
) -> None:
    mode = None if base_init is None else base_init.get("mode")
    if mode is None:
        _initialize_base_model_from_pretrained_embeddings(
            raw_model,
            model_cfg=model_cfg,
            distributed=distributed,
            is_main_process=is_main_process,
        )
        return

    if mode == "checkpoint":
        checkpoint_path = base_init.get("checkpoint_path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("LoRA checkpoint metadata is missing base_init.checkpoint_path.")
        init_path = _normalize_checkpoint_path(checkpoint_path)
        init_state, init_model_cfg, _ = _load_model_state_from_checkpoint(init_path)
        checkpoint_has_caption = checkpoint_uses_caption_condition(init_model_cfg, init_state)
        current_has_caption = bool(model_cfg.use_caption_condition)
        if checkpoint_has_caption and not current_has_caption:
            raise ValueError(
                "Caption-conditioned checkpoint cannot initialize a caption-free config. "
                "Use a caption-enabled config for this checkpoint."
            )

        require_caption_match = checkpoint_has_caption and current_has_caption
        _check_model_config_compatibility(
            init_path,
            init_model_cfg,
            model_cfg,
            require_caption_match=require_caption_match,
        )

        missing_keys: list[str] = []
        initialized_caption_embedding = False
        if current_has_caption and not checkpoint_has_caption:
            missing_keys, skipped_shape, skipped_extra = load_model_state_partially(
                raw_model,
                init_state,
            )
            validate_caption_upgrade_partial_load(
                init_path,
                missing_keys,
                skipped_shape,
                skipped_extra,
            )
            if distributed:
                if is_main_process:
                    print(
                        "Initializing caption embedding from pretrained model after caption-free checkpoint load: "
                        f"{model_cfg.caption_tokenizer_repo_resolved}"
                    )
                    initialize_caption_embedding_from_pretrained(
                        raw_model,
                        model_cfg,
                        local_files_only=False,
                    )
                dist.barrier()
                if not is_main_process:
                    initialize_caption_embedding_from_pretrained(
                        raw_model,
                        model_cfg,
                        local_files_only=True,
                    )
                dist.barrier()
            else:
                if is_main_process:
                    print(
                        "Initializing caption embedding from pretrained model after caption-free checkpoint load: "
                        f"{model_cfg.caption_tokenizer_repo_resolved}"
                    )
                initialize_caption_embedding_from_pretrained(
                    raw_model,
                    model_cfg,
                    local_files_only=False,
                )
            initialized_caption_embedding = True
        else:
            raw_model.load_state_dict(init_state, strict=True)

        if is_main_process:
            print(f"Initialized model weights from: {init_path}")
            if missing_keys:
                print(f"Partial load missing keys: {len(missing_keys)}")
            if initialized_caption_embedding:
                print("Caption embedding was initialized from its pretrained tokenizer backbone.")
        return

    raise ValueError(f"Unsupported base_init mode: {mode!r}")


def resolve_dist_env() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def setup_distributed(device_arg: str) -> tuple[int, int, int, bool, torch.device]:
    rank, world_size, local_rank = resolve_dist_env()
    distributed = world_size > 1
    if distributed:
        if not str(device_arg).startswith("cuda"):
            raise ValueError(
                f"WORLD_SIZE={world_size} detected, but --device={device_arg!r}. "
                "DDP multi-GPU training requires --device cuda."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("WORLD_SIZE>1 detected, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_arg)
    return rank, world_size, local_rank, distributed, device


def reduce_mean(value: torch.Tensor, world_size: int, distributed: bool) -> torch.Tensor:
    reduced = value.detach().clone()
    if not distributed:
        return reduced
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= float(world_size)
    return reduced


def split_train_valid_indices(
    *,
    num_samples: int,
    valid_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if valid_ratio <= 0.0:
        return list(range(num_samples)), []
    if num_samples < 2:
        raise ValueError(
            f"Validation split requires at least 2 samples in manifest, got {num_samples}."
        )

    valid_count = int(num_samples * valid_ratio)
    valid_count = max(VALID_MIN_COUNT, min(VALID_MAX_COUNT, valid_count))
    if valid_count >= num_samples:
        valid_count = num_samples - 1
    valid_count = max(1, valid_count)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator).tolist()
    valid_indices = sorted(perm[:valid_count])
    train_indices = sorted(perm[valid_count:])
    if not train_indices or not valid_indices:
        raise ValueError(
            "Failed to create non-empty train/valid split. "
            f"num_samples={num_samples} valid_ratio={valid_ratio}"
        )
    return train_indices, valid_indices


def run_validation(
    *,
    model,
    loader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
    use_bf16: bool,
    distributed: bool,
) -> dict[str, float]:
    was_training = model.training
    model_cfg = model.module.cfg if isinstance(model, DDP) else model.cfg
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)

    with torch.no_grad():
        for batch in loader:
            text_ids = batch["text_ids"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)
            caption_ids = None
            caption_mask = None
            if model_cfg.use_caption_condition:
                caption_ids = batch["caption_ids"].to(device, non_blocking=True)
                caption_mask = batch["caption_mask"].to(device, non_blocking=True)
            x0 = batch["latent_patched"].to(device, non_blocking=True)
            x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
            x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
            ref_latent = None
            ref_mask = None
            if model_cfg.use_speaker_condition:
                ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
                ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
                has_speaker = batch["has_speaker"].to(device, non_blocking=True)
            else:
                has_speaker = None

            bsz = x0.shape[0]
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

            if model_cfg.use_speaker_condition:
                use_speaker = has_speaker
                ref_mask = ref_mask & use_speaker[:, None]
                ref_latent = ref_latent * use_speaker[:, None, None].to(ref_latent.dtype)

            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            ):
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
                )

            v_pred = v_pred.float()
            rf_loss = echo_style_masked_mse(
                v_pred,
                v_target.float(),
                loss_mask=x_mask,
                valid_mask=x_mask_valid,
            )
            loss = rf_loss

            weight = float(bsz)
            totals[0] += loss.detach().double() * weight
            totals[1] += rf_loss.detach().double() * weight
            totals[2] += weight

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    denom = max(float(totals[2].item()), 1.0)
    metrics = {
        "loss": float(totals[0].item() / denom),
        "rf_loss": float(totals[1].item() / denom),
        "num_samples": float(totals[2].item()),
    }
    if was_training:
        model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Irodori-TTS.")
    parser.add_argument("--config", default=None, help="YAML config path (model/train overrides)")
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
    args = parser.parse_args()
    if args.resume is not None and Path(args.resume).suffix.lower() == ".safetensors":
        raise ValueError(
            "--resume expects a training checkpoint (.pt or LoRA checkpoint dir). "
            "Use --init-checkpoint for inference-only .safetensors weights."
        )

    rank, world_size, local_rank, distributed, device = setup_distributed(args.device)
    is_main_process = rank == 0

    raw_argv = sys.argv[1:]
    exp_cfg = load_experiment_yaml(args.config) if args.config else {}
    unknown_root = sorted(set(exp_cfg) - {"model", "train", "sample_generation"})
    if unknown_root:
        raise ValueError(f"Unknown top-level config keys: {unknown_root}")
    if args.config and is_main_process:
        print(f"Loaded config: {args.config}")
    model_cfg = merge_dataclass_overrides(ModelConfig(), exp_cfg.get("model"), section="model")
    train_cfg = merge_dataclass_overrides(TrainConfig(), exp_cfg.get("train"), section="train")
    sample_cfg = merge_sample_generation_overrides(exp_cfg.get("sample_generation"))
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
    if cli_provided(raw_argv, "--timestep-stratified"):
        train_cfg = replace(train_cfg, timestep_stratified=True)
    if cli_provided(raw_argv, "--max-latent-steps"):
        train_cfg = replace(train_cfg, max_latent_steps=args.max_latent_steps)
    if cli_provided(raw_argv, "--fixed-target-latent-steps"):
        train_cfg = replace(train_cfg, fixed_target_latent_steps=args.fixed_target_latent_steps)
    if cli_provided(raw_argv, "--fixed-target-full-mask"):
        train_cfg = replace(train_cfg, fixed_target_full_mask=True)
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
    if args.ddp_find_unused_parameters is not None:
        train_cfg = replace(
            train_cfg,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        )
    if cli_provided(raw_argv, "--seed"):
        train_cfg = replace(train_cfg, seed=args.seed)

    resume_path = Path(args.resume).expanduser() if args.resume is not None else None
    resume_train_cfg = None
    resume_base_init = None
    if args.resume is not None:
        resume_meta = _load_checkpoint_payload(resume_path, map_location="cpu")
        raw_resume_train_cfg = resume_meta.get("train_config")
        if raw_resume_train_cfg is not None and not isinstance(raw_resume_train_cfg, dict):
            raise ValueError("Resume checkpoint train_config must be a dictionary when present.")
        resume_train_cfg = raw_resume_train_cfg
        raw_resume_base_init = resume_meta.get("base_init")
        if raw_resume_base_init is not None and not isinstance(raw_resume_base_init, dict):
            raise ValueError("Resume checkpoint base_init must be a dictionary when present.")
        resume_base_init = raw_resume_base_init
        train_cfg = _restore_resume_lora_config(
            train_cfg,
            resume_train_cfg=resume_train_cfg,
            args=args,
            raw_argv=raw_argv,
            exp_cfg=exp_cfg,
        )

    if cli_provided(raw_argv, "--latent-dim"):
        model_cfg = replace(model_cfg, latent_dim=args.latent_dim)
    if cli_provided(raw_argv, "--latent-patch-size"):
        model_cfg = replace(model_cfg, latent_patch_size=args.latent_patch_size)

    set_seed(train_cfg.seed + rank)
    if not (0.0 <= train_cfg.text_condition_dropout <= 1.0):
        raise ValueError(
            f"text_condition_dropout must be in [0, 1], got {train_cfg.text_condition_dropout}"
        )
    if train_cfg.max_text_len <= 0:
        raise ValueError(f"max_text_len must be > 0, got {train_cfg.max_text_len}")
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
    if train_cfg.caption_warmup_steps < 0:
        raise ValueError(f"caption_warmup_steps must be >= 0, got {train_cfg.caption_warmup_steps}")
    if train_cfg.dataloader_prefetch_factor <= 0:
        raise ValueError(
            f"dataloader_prefetch_factor must be > 0, got {train_cfg.dataloader_prefetch_factor}"
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
    if is_main_process and distributed:
        print(f"DDP enabled: world_size={world_size} (local_rank={local_rank})")
    wandb_run = None
    if train_cfg.wandb_enabled and is_main_process:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but `wandb` is not installed. "
                "Install it with `pip install wandb`."
            ) from exc
        wandb_settings = None
        cf_id = os.environ.get("CF_ACCESS_CLIENT_ID")
        cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")
        if cf_id and cf_secret:
            wandb_settings = wandb.Settings(
                x_extra_http_headers={
                    "CF-Access-Client-Id": cf_id,
                    "CF-Access-Client-Secret": cf_secret,
                }
            )
        wandb_run = wandb.init(
            project=train_cfg.wandb_project or None,
            entity=train_cfg.wandb_entity or None,
            name=train_cfg.wandb_run_name or None,
            mode=train_cfg.wandb_mode or "online",
            dir=str(output_dir),
            config={
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "script": "train.py",
            },
            settings=wandb_settings,
        )
        print(
            f"W&B enabled: project={train_cfg.wandb_project} mode={train_cfg.wandb_mode} run={wandb_run.name if wandb_run is not None else train_cfg.wandb_run_name}"
        )

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
    full_dataset = LatentTextDataset(
        manifest_path=train_cfg.manifest_path,
        latent_dim=model_cfg.latent_dim,
        max_latent_steps=train_cfg.max_latent_steps,
        enable_caption_condition=model_cfg.use_caption_condition,
        enable_speaker_condition=model_cfg.use_speaker_condition,
        show_manifest_progress=bool(train_cfg.progress and is_main_process),
        manifest_progress_desc="Index Manifest",
    )
    manifest_size = len(full_dataset)
    train_dataset = full_dataset
    valid_dataset = None
    if train_cfg.valid_ratio > 0.0:
        train_indices, valid_indices = split_train_valid_indices(
            num_samples=len(full_dataset),
            valid_ratio=train_cfg.valid_ratio,
            seed=train_cfg.seed,
        )
        train_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=train_indices,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition,
            manifest_index=full_dataset.manifest_index,
        )
        valid_dataset = LatentTextDataset(
            manifest_path=train_cfg.manifest_path,
            latent_dim=model_cfg.latent_dim,
            max_latent_steps=train_cfg.max_latent_steps,
            subset_indices=valid_indices,
            enable_caption_condition=model_cfg.use_caption_condition,
            enable_speaker_condition=model_cfg.use_speaker_condition,
            manifest_index=full_dataset.manifest_index,
        )
        if is_main_process:
            print(
                f"Validation split enabled: train={len(train_dataset)} valid={len(valid_dataset)} "
                f"(ratio={train_cfg.valid_ratio:.4f}, clamp=[{VALID_MIN_COUNT},{VALID_MAX_COUNT}], valid_every={train_cfg.valid_every} steps)."
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
    if not model_cfg.use_speaker_condition and is_main_process:
        print("Speaker conditioning disabled for caption-conditioned voice-design model.")
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
    if train_cfg.timestep_stratified and is_main_process:
        print("Using stratified logit-normal timestep sampling.")
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=drop_last,
        )
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
    loader = DataLoader(
        dataset=train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=drop_last,
        **dataloader_common_kwargs,
    )
    if len(loader) == 0:
        raise ValueError("Dataloader yielded zero batches. Check manifest and batch_size settings.")
    import math
    optim_steps_per_epoch = max(
        1,
        math.ceil(len(loader) / max(1, train_cfg.gradient_accumulation_steps)),
    )

    speaker_name = _resolve_speaker_name(train_cfg.manifest_path)
    run_name = (wandb_run.name if wandb_run is not None else None) or train_cfg.wandb_run_name or output_dir.name
    run_uuid: str | None = None
    if args.resume is not None:
        try:
            from safetensors import safe_open
            existing_adapter = Path(args.resume) / "adapter_model.safetensors"
            if existing_adapter.is_file():
                with safe_open(str(existing_adapter), framework="pt", device="cpu") as _f:
                    run_uuid = (_f.metadata() or {}).get("uuid")
        except Exception:
            run_uuid = None
    if not run_uuid:
        run_uuid = str(_uuid.uuid4())
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
            int(round(ms * float(train_cfg.warmup_ratio)))
            if train_cfg.warmup_ratio is not None
            else int(train_cfg.warmup_steps)
        )
        decay = (
            int(round(ms * float(train_cfg.decay_ratio)))
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
        valid_loader = DataLoader(
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
        periodic_checkpoint_keep = 10_000 if has_validation else int(train_cfg.checkpoint_best_n) + 1
    best_val_checkpoints: list[tuple[float, int, Path]] = []
    if is_main_process:
        if checkpoint_retention_enabled and has_validation:
            best_val_checkpoints = list_best_val_loss_checkpoints(output_dir)
            best_val_checkpoints = prune_best_val_loss_checkpoints(
                best_val_checkpoints,
                train_cfg.checkpoint_best_n,
            )
        if checkpoint_retention_enabled and has_validation:
            print(f"Checkpoint retention: periodic_keep={periodic_checkpoint_keep} + best_val_loss={train_cfg.checkpoint_best_n}.")
        elif checkpoint_retention_enabled:
            print(
                f"Checkpoint retention: validation disabled, keep latest {periodic_checkpoint_keep} periodic checkpoints."
            )

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

    raw_model: torch.nn.Module = TextToLatentRFDiT(model_cfg).to(device)
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
            f"trainable={trainable_params:,}/{total_params:,}"
        )
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
        if train_cfg.gradient_accumulation_steps > 1:
            print(
                f"Gradient accumulation enabled: steps={train_cfg.gradient_accumulation_steps} (effective global batch={train_cfg.batch_size * world_size * train_cfg.gradient_accumulation_steps})."
            )

    step = 0
    progress: TrainProgress | None = None
    resumed_es_best_val: float | None = None
    resumed_es_no_improve: int | None = None
    if args.resume is not None:
        ckpt = _load_checkpoint_payload(resume_path, map_location=device)
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
        if is_main_process:
            print(
                f"Resumed from step={step}"
                + (
                    f" es_best_val={resumed_es_best_val:.6f} es_no_improve={resumed_es_no_improve}"
                    if resumed_es_best_val is not None and resumed_es_no_improve is not None
                    else ""
                )
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
            wandb_run=wandb_run,
            log_fn=lambda msg: progress.write(msg) if progress is not None else None,
        )

    progress = TrainProgress(
        max_steps=train_cfg.max_steps,
        start_step=step,
        rank=rank,
        world_size=world_size,
        enabled=train_cfg.progress,
        show_all_ranks=train_cfg.progress_all_ranks,
        description="Train RF",
    )
    accum_steps = int(train_cfg.gradient_accumulation_steps)
    global_batch_size = train_cfg.batch_size * world_size * accum_steps
    caption_warmup_active = bool(
        train_cfg.caption_warmup
        and model_cfg.use_caption_condition
        and train_cfg.caption_warmup_steps > 0
        and step < train_cfg.caption_warmup_steps
    )
    if caption_warmup_active and is_main_process:
        print(
            "Caption warmup active: non-caption gradients will be cleared for the first "
            f"{train_cfg.caption_warmup_steps} optimizer steps."
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
        epoch = 0
        while step < train_cfg.max_steps and not stop_early:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch += 1
            for epoch_step, batch in enumerate(loader, start=1):
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
                x0 = batch["latent_patched"].to(device, non_blocking=True)
                x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
                x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
                ref_latent = None
                ref_mask = None
                if raw_model.cfg.use_speaker_condition:
                    ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
                    ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
                    has_speaker = batch["has_speaker"].to(device, non_blocking=True)
                else:
                    has_speaker = None

                bsz = x0.shape[0]
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
                if text_cond_drop.any():
                    text_mask = text_mask.clone()
                    text_mask[text_cond_drop] = False
                caption_cond_drop = None
                if raw_model.cfg.use_caption_condition:
                    if has_caption is None or caption_mask is None:
                        raise RuntimeError(
                            "Caption conditioning is enabled but caption batch tensors are missing."
                        )
                    caption_cond_drop = (
                        torch.rand(bsz, device=device) < train_cfg.caption_condition_dropout
                    )
                    use_caption = has_caption & (~caption_cond_drop)
                    caption_mask = caption_mask & use_caption[:, None]

                if raw_model.cfg.use_speaker_condition:
                    speaker_cond_drop = (
                        torch.rand(bsz, device=device) < train_cfg.speaker_condition_dropout
                    )
                    use_speaker = has_speaker & (~speaker_cond_drop)
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
                            speaker_condition_dropout=None,
                            caption_condition_dropout=None,
                        )

                    v_pred = v_pred.float()
                    rf_loss = echo_style_masked_mse(
                        v_pred,
                        v_target.float(),
                        loss_mask=x_mask,
                        valid_mask=x_mask_valid,
                    )
                    loss = rf_loss
                    (loss / float(accum_steps)).backward()
                    if caption_warmup_active:
                        clear_non_caption_grads(raw_model)

                accum_loss += loss.detach()
                accum_rf_loss += rf_loss.detach()
                if not should_step:
                    continue

                step_loss = accum_loss / float(accum_steps)
                step_rf_loss = accum_rf_loss / float(accum_steps)
                accum_loss.zero_()
                accum_rf_loss.zero_()

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

                if step % train_cfg.log_every == 0:
                    loss_value = reduce_mean(step_loss, world_size, distributed).item()
                    rf_loss_value = reduce_mean(step_rf_loss, world_size, distributed).item()
                    lr_value = current_lr(optimizer)
                    progress_metrics: dict[str, float] = {
                        "loss": loss_value,
                        "rf": rf_loss_value,
                        "lr": lr_value,
                    }
                    progress.log(
                        step=step,
                        epoch=epoch,
                        epoch_step=epoch_step,
                        epoch_total=len(loader),
                        metrics=progress_metrics,
                        global_batch_size=global_batch_size,
                    )
                    if is_main_process:
                        progress.write(
                            f"step={step} loss={loss_value:.6f} rf={rf_loss_value:.6f} lr={lr_value:.3e}"
                        )
                        if wandb_run is not None:
                            metrics = {
                                "train/loss": loss_value,
                                "train/rf_loss": rf_loss_value,
                                "train/lr": lr_value,
                            }
                            wandb_run.log(metrics, step=step)

                if step % train_cfg.save_every == 0 and is_main_process:
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
                    )
                    enforce_periodic_checkpoint_limit(
                        output_dir=output_dir,
                        keep_count=periodic_checkpoint_keep,
                    )
                    if sample_cfg.enabled and sample_cfg.every > 0 and step % sample_cfg.every == 0:
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
                    if is_main_process:
                        progress.write(
                            ("valid step={} loss={:.6f} rf={:.6f} (samples={:.0f})").format(
                                step,
                                valid_metrics["loss"],
                                valid_metrics["rf_loss"],
                                valid_metrics["num_samples"],
                            )
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "valid/loss": valid_metrics["loss"],
                                    "valid/rf_loss": valid_metrics["rf_loss"],
                                },
                                step=step,
                            )
                        if es_enabled:
                            cur_val = float(valid_metrics["loss"])
                            if cur_val < es_best_val - train_cfg.early_stop_min_delta:
                                es_best_val = cur_val
                                es_no_improve = 0
                            else:
                                es_no_improve += 1
                            if wandb_run is not None:
                                wandb_run.log(
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
                                elif (
                                    es_best_val < float("inf")
                                    and cur_val > es_best_val * (1.0 + train_cfg.early_stop_regression_ratio)
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
                        )
                        if best_path is not None:
                            progress.write(
                                "saved best val checkpoint: {} (loss={:.6f})".format(
                                    best_path.name,
                                    float(valid_metrics["loss"]),
                                )
                            )
                            _upload_best_checkpoint_artifact(
                                wandb_run=wandb_run,
                                path=best_path,
                                step=step,
                                val_loss=float(valid_metrics["loss"]),
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
            if is_main_process:
                progress.write(
                    ("valid final step={} loss={:.6f} rf={:.6f} (samples={:.0f})").format(
                        step,
                        valid_metrics["loss"],
                        valid_metrics["rf_loss"],
                        valid_metrics["num_samples"],
                    )
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "valid/loss": valid_metrics["loss"],
                            "valid/rf_loss": valid_metrics["rf_loss"],
                        },
                        step=step,
                    )
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
                )
                if best_path is not None:
                    progress.write(
                        "saved best val checkpoint: {} (loss={:.6f})".format(
                            best_path.name,
                            float(valid_metrics["loss"]),
                        )
                    )
                    _upload_best_checkpoint_artifact(
                        wandb_run=wandb_run,
                        path=best_path,
                        step=step,
                        val_loss=float(valid_metrics["loss"]),
                    )
                    if sample_cfg.enabled and sample_cfg.on_best_val:
                        _maybe_emit_samples(step)

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
            )
            if sample_cfg.enabled:
                _maybe_emit_samples(step)
            if wandb_run is not None:
                wandb_run.summary["train/final_step"] = step
            progress.write(f"Training finished at step={step}.")
    finally:
        if progress is not None:
            progress.close()
        if sampling_codec is not None:
            del sampling_codec
        if wandb_run is not None:
            wandb_run.finish()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
