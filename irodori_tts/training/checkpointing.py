"""Checkpoint writing, path conventions and retention for the training loop."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from irodori_tts.config import ModelConfig, TrainConfig, dump_configs
from irodori_tts.lora import (
    LORA_METADATA_NAME,
    LORA_TRAINER_STATE_NAME,
    train_config_uses_lora,
)
from irodori_tts.speaker_inversion import (
    SPEAKER_INVERSION_SAFETENSORS_SUFFIX,
    save_speaker_inversion_checkpoint,
)
from irodori_tts.training.speaker_prompts import (
    _build_lora_safetensors_metadata,
    _inject_safetensors_metadata,
)

CHECKPOINT_STEP_RE = re.compile(
    rf"^checkpoint_(\d+)(?:\.pt|{re.escape(SPEAKER_INVERSION_SAFETENSORS_SUFFIX)})?$"
)
CHECKPOINT_BEST_VAL_LOSS_RE = re.compile(
    rf"^checkpoint_best_val_loss_(\d+)_(-?\d+(?:\.\d+)?)"
    rf"(?:\.pt|{re.escape(SPEAKER_INVERSION_SAFETENSORS_SUFFIX)})?$"
)
DATALOADER_STATE_KEY = "dataloader_state"
RUNTIME_STATE_KEY = "runtime_state"


def save_checkpoint(  # noqa: PLR0913, PLR0917
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
    dataloader_state: dict | None = None,
    runtime_state: dict | None = None,
) -> None:
    path = Path(path)
    if train_cfg.speaker_inversion_enabled:
        save_speaker_inversion_checkpoint(path, model=model)
        return

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
                speaker_id=speaker_name,
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
                DATALOADER_STATE_KEY: dataloader_state,
                RUNTIME_STATE_KEY: runtime_state,
            },
            path / LORA_TRAINER_STATE_NAME,
        )
        return

    text_encoder_config = None
    pretrained_backbone = getattr(model, "pretrained_text_backbone", None)
    if pretrained_backbone is not None:
        raw_config = getattr(pretrained_backbone, "config_dict", None)
        if isinstance(raw_config, dict):
            text_encoder_config = dict(raw_config)

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
            "text_encoder_config": text_encoder_config,
            DATALOADER_STATE_KEY: dataloader_state,
            RUNTIME_STATE_KEY: runtime_state,
        },
        path,
    )


def _runtime_state_for_checkpoint(*, epoch: int, epoch_step: int) -> dict[str, int]:
    return {
        "epoch": int(epoch),
        "sampler_epoch": max(0, int(epoch) - 1),
        "epoch_step": int(epoch_step),
    }


def _collect_dataloader_state(
    loader: StatefulDataLoader,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> dict:
    local_state = loader.state_dict()
    if not distributed:
        return {
            "version": 1,
            "world_size": 1,
            "rank_states": [local_state],
        }

    rank_states: list[dict | None] = [None for _ in range(world_size)]
    dist.all_gather_object(rank_states, local_state)
    return {
        "version": 1,
        "world_size": int(world_size),
        "rank_states": rank_states,
        "saved_by_rank": int(rank),
    }


def _select_dataloader_state_for_rank(
    payload: dict,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> dict | None:
    state = payload.get(DATALOADER_STATE_KEY)
    if state is None:
        return None
    if not isinstance(state, dict):
        raise ValueError("Checkpoint dataloader_state must be a dictionary when present.")  # noqa: TRY004
    rank_states = state.get("rank_states")
    if not isinstance(rank_states, list):
        raise ValueError("Checkpoint dataloader_state.rank_states must be a list.")  # noqa: TRY004
    saved_world_size = int(state.get("world_size", len(rank_states)))
    expected_world_size = int(world_size) if distributed else 1
    if saved_world_size != expected_world_size or len(rank_states) != expected_world_size:
        raise ValueError(
            "Cannot restore dataloader state with a different world_size: "
            f"checkpoint={saved_world_size} current={expected_world_size}"
        )
    state_rank = int(rank) if distributed else 0
    rank_state = rank_states[state_rank]
    if rank_state is not None and not isinstance(rank_state, dict):
        raise ValueError(f"Checkpoint dataloader state for rank {state_rank} must be a dictionary.")
    return _move_state_tensors_to_cpu(rank_state)


def _move_state_tensors_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_state_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state_tensors_to_cpu(item) for item in value)
    return value


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


def maybe_save_best_val_loss_checkpoint(  # noqa: PLR0913
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
    dataloader_state: dict | None = None,
    runtime_state: dict | None = None,
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
        dataloader_state=dataloader_state,
        runtime_state=runtime_state,
    )
    checkpoints.append((float(val_loss), int(step), path))
    checkpoints = prune_best_val_loss_checkpoints(checkpoints, keep_best_n)
    return checkpoints, path


def _periodic_checkpoint_path(output_dir: Path, step: int, train_cfg: TrainConfig) -> Path:
    if train_cfg.speaker_inversion_enabled:
        return output_dir / f"checkpoint_{step:07d}{SPEAKER_INVERSION_SAFETENSORS_SUFFIX}"
    if train_config_uses_lora(train_cfg):
        return output_dir / f"checkpoint_{step:07d}"
    return output_dir / f"checkpoint_{step:07d}.pt"


def _best_checkpoint_path(
    output_dir: Path, *, step: int, val_loss: float, train_cfg: TrainConfig
) -> Path:
    if train_cfg.speaker_inversion_enabled:
        return (
            output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}"
            f"{SPEAKER_INVERSION_SAFETENSORS_SUFFIX}"
        )
    if train_config_uses_lora(train_cfg):
        return output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}"
    return output_dir / f"checkpoint_best_val_loss_{step:07d}_{val_loss:.6f}.pt"


def _final_checkpoint_path(output_dir: Path, train_cfg: TrainConfig) -> Path:
    if train_cfg.speaker_inversion_enabled:
        return output_dir / f"checkpoint_final{SPEAKER_INVERSION_SAFETENSORS_SUFFIX}"
    if train_config_uses_lora(train_cfg):
        return output_dir / "checkpoint_final"
    return output_dir / "checkpoint_final.pt"


def _load_checkpoint_payload(path: str | Path, *, map_location) -> dict:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        state_path = checkpoint_path / LORA_TRAINER_STATE_NAME
        payload = torch.load(state_path, map_location=map_location, weights_only=True)
    else:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")  # noqa: TRY004
    return payload
