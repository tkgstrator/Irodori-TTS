"""Model construction, pretrained initialization and checkpoint compatibility.

Backward-compatibility machinery: the order of the checks in
``_check_model_config_compatibility`` and the tiling in ``_upgrade_speaker_in_proj``
decide whether checkpoints already on disk still load.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from irodori_tts.config import ModelConfig, TrainConfig, merge_dataclass_overrides
from irodori_tts.lora import LORA_TRAIN_CONFIG_FIELDS
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer
from irodori_tts.training.cli_args import _lora_field_cli_explicit

SAFETENSORS_CONFIG_META_KEY = "config_json"
SAFETENSORS_TEXT_ENCODER_CONFIG_META_KEY = "text_encoder_config_json"
SAFETENSORS_INFERENCE_CONFIG_KEYS = {
    "max_text_len",
    "max_caption_len",
    "fixed_target_latent_steps",
    "ref_max_seconds",
}


def build_condition_tokenizer(
    *,
    repo_id: str,
    add_bos: bool,
    vocab_size: int | None,
    local_files_only: bool = False,
    revision: str | None = None,
) -> PretrainedTextTokenizer:
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=repo_id,
        add_bos=bool(add_bos),
        local_files_only=local_files_only,
        revision=revision,
    )
    if vocab_size is not None and tokenizer.vocab_size != vocab_size:
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
        vocab_size=(
            None if model_cfg.use_pretrained_text_encoder else int(model_cfg.text_vocab_size)
        ),
        local_files_only=local_files_only,
        revision=model_cfg.text_encoder_revision,
    )


def build_caption_tokenizer(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> PretrainedTextTokenizer:
    return build_condition_tokenizer(
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        add_bos=model_cfg.caption_add_bos_resolved,
        vocab_size=(
            None if model_cfg.use_pretrained_text_encoder else model_cfg.caption_vocab_size_resolved
        ),
        local_files_only=local_files_only,
        revision=model_cfg.text_encoder_revision,
    )


def validate_pretrained_backbone_dim(
    *,
    repo_id: str,
    expected_dim: int | None,
    local_files_only: bool = False,
    revision: str | None = None,
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
        revision=revision,
    )
    hidden_size = getattr(text_cfg, "hidden_size", None)
    if hidden_size is None:
        encoder_cfg = getattr(text_cfg, "encoder", None)
        hidden_size = getattr(encoder_cfg, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(encoder_cfg, "text_config", None), "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Could not read hidden_size from pretrained config: {repo_id}")
    hidden_size = int(hidden_size)
    if expected_dim is not None and hidden_size != expected_dim:
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
        expected_dim=(None if model_cfg.use_pretrained_text_encoder else int(model_cfg.text_dim)),
        local_files_only=local_files_only,
        revision=model_cfg.text_encoder_revision,
    )


def validate_caption_backbone_dim(
    model_cfg: ModelConfig,
    *,
    local_files_only: bool = False,
) -> int:
    return validate_pretrained_backbone_dim(
        repo_id=model_cfg.caption_tokenizer_repo_resolved,
        expected_dim=(
            None if model_cfg.use_pretrained_text_encoder else model_cfg.caption_dim_resolved
        ),
        local_files_only=local_files_only,
        revision=model_cfg.text_encoder_revision,
    )


def initialize_embedding_from_pretrained(
    embedding: torch.nn.Embedding,
    *,
    repo_id: str,
    local_files_only: bool = False,
    revision: str | None = None,
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
        revision=revision,
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
        revision=model_cfg.text_encoder_revision,
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
        revision=model_cfg.text_encoder_revision,
    )


def _load_model_state_from_checkpoint(  # noqa: C901
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict | None, dict | None, dict | None]:
    if path.suffix.lower() == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file as load_safetensors_file

        checkpoint_model_cfg = None
        text_encoder_config = None
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        from irodori_tts.quantization import parse_quantization_metadata

        if parse_quantization_metadata(metadata) is not None:
            raise ValueError(
                "Quantized checkpoints are inference-only and cannot be used with "
                "--init-checkpoint. Train LoRA against the matching full-precision base model, "
                "then merge and quantize it for inference."
            )
        config_json = metadata.get(SAFETENSORS_CONFIG_META_KEY)
        if config_json:
            parsed = json.loads(config_json)
            if isinstance(parsed, dict):
                checkpoint_model_cfg = {
                    key: value
                    for key, value in parsed.items()
                    if key not in SAFETENSORS_INFERENCE_CONFIG_KEYS
                }
        text_encoder_config_json = metadata.get(SAFETENSORS_TEXT_ENCODER_CONFIG_META_KEY)
        if text_encoder_config_json:
            parsed = json.loads(text_encoder_config_json)
            if isinstance(parsed, dict):
                text_encoder_config = parsed
        return (
            load_safetensors_file(str(path), device="cpu"),
            checkpoint_model_cfg,
            None,
            text_encoder_config,
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary, got {type(payload)!r}.")  # noqa: TRY004

    raw_model = payload.get("model")
    if raw_model is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
        raw_model = payload
    if not isinstance(raw_model, dict):
        raise ValueError(f"Checkpoint does not contain a model state dictionary: {path}")  # noqa: TRY004

    checkpoint_model_cfg = payload.get("model_config")
    if checkpoint_model_cfg is not None and not isinstance(checkpoint_model_cfg, dict):
        raise ValueError(f"Checkpoint model_config must be a dictionary when present: {path}")
    checkpoint_train_cfg = payload.get("train_config")
    if checkpoint_train_cfg is not None and not isinstance(checkpoint_train_cfg, dict):
        raise ValueError(f"Checkpoint train_config must be a dictionary when present: {path}")
    text_encoder_config = payload.get("text_encoder_config")
    if text_encoder_config is not None and not isinstance(text_encoder_config, dict):
        raise ValueError(f"Checkpoint text_encoder_config must be a dictionary: {path}")
    return raw_model, checkpoint_model_cfg, checkpoint_train_cfg, text_encoder_config


def _check_model_config_compatibility(  # noqa: C901, PLR0913
    checkpoint_path: Path,
    checkpoint_model_cfg: dict | None,
    current_model_cfg: ModelConfig,
    *,
    require_caption_match: bool,
    upgrade_speaker_patch: bool = False,
    upgrade_text_encoder: bool = False,
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
        ("text_dim", checkpoint_cfg.text_dim, current_model_cfg.text_dim),
        ("adaln_rank", checkpoint_cfg.adaln_rank, current_model_cfg.adaln_rank),
    ]
    if not upgrade_text_encoder:
        comparisons.append(
            (
                "text_encoder_type",
                checkpoint_cfg.text_encoder_type,
                current_model_cfg.text_encoder_type,
            )
        )
    if not upgrade_text_encoder and (
        checkpoint_cfg.use_pretrained_text_encoder or current_model_cfg.use_pretrained_text_encoder
    ):
        comparisons.append(
            (
                "text_tokenizer_repo",
                checkpoint_cfg.text_tokenizer_repo,
                current_model_cfg.text_tokenizer_repo,
            )
        )
        comparisons.extend(
            [
                (
                    "text_encoder_revision",
                    checkpoint_cfg.text_encoder_revision,
                    current_model_cfg.text_encoder_revision,
                ),
                (
                    "pretrained_projector_type",
                    checkpoint_cfg.pretrained_projector_type,
                    current_model_cfg.pretrained_projector_type,
                ),
                (
                    "pretrained_projector_hidden_ratio",
                    checkpoint_cfg.pretrained_projector_hidden_ratio,
                    current_model_cfg.pretrained_projector_hidden_ratio,
                ),
                (
                    "pretrained_projector_dropout",
                    checkpoint_cfg.pretrained_projector_dropout,
                    current_model_cfg.pretrained_projector_dropout,
                ),
            ]
        )
    elif not upgrade_text_encoder:
        comparisons.extend(
            [
                (
                    "text_vocab_size",
                    checkpoint_cfg.text_vocab_size,
                    current_model_cfg.text_vocab_size,
                ),
                ("text_layers", checkpoint_cfg.text_layers, current_model_cfg.text_layers),
                ("text_heads", checkpoint_cfg.text_heads, current_model_cfg.text_heads),
                (
                    "text_mlp_ratio",
                    checkpoint_cfg.text_mlp_ratio_resolved,
                    current_model_cfg.text_mlp_ratio_resolved,
                ),
            ]
        )
    if (
        checkpoint_cfg.use_speaker_condition_resolved
        and current_model_cfg.use_speaker_condition_resolved
    ):
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
            ]
        )
        if not upgrade_speaker_patch:
            comparisons.append(
                (
                    "speaker_patch_size",
                    checkpoint_cfg.speaker_patch_size,
                    current_model_cfg.speaker_patch_size,
                )
            )
        else:
            old_patch = int(checkpoint_cfg.speaker_patch_size)
            new_patch = int(current_model_cfg.speaker_patch_size)
            if (
                old_patch <= 0
                or new_patch <= 0
                or new_patch <= old_patch
                or new_patch % old_patch != 0
            ):
                raise ValueError(
                    "speaker_patch_size upgrade requires new patch to be a positive "
                    f"integer multiple of the checkpoint's: got old={old_patch} "
                    f"new={new_patch} ({checkpoint_path})"
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
                    checkpoint_cfg.use_speaker_condition_resolved,
                    current_model_cfg.use_speaker_condition_resolved,
                ),
                (
                    "caption_dim",
                    checkpoint_cfg.caption_dim_resolved,
                    current_model_cfg.caption_dim_resolved,
                ),
            ]
        )
        if not upgrade_text_encoder:
            comparisons.extend(
                [
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
        key.startswith(("caption_encoder.", "caption_norm."))
        or ".wk_caption." in key
        or ".wv_caption." in key
        for key in state_dict
    )


def checkpoint_uses_duration_predictor(
    checkpoint_model_cfg: dict | None,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    if checkpoint_model_cfg is not None:
        checkpoint_cfg = merge_dataclass_overrides(
            ModelConfig(),
            checkpoint_model_cfg,
            section="checkpoint model_config",
        )
        if checkpoint_cfg.use_duration_predictor:
            return True
    return any(key.startswith("duration_predictor.") for key in state_dict)


def load_model_state_partially(
    model: TextToLatentRFDiT,
    state_dict: dict[str, torch.Tensor],
    *,
    reinit_keys: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Load state_dict into ``model`` non-strictly, tolerating both extra and
    missing/renamed keys.

    Keys in ``reinit_keys`` are treated as an explicit "will be reinitialized
    by the caller" contract: they are NOT reported in ``skipped_shape`` even
    if the checkpoint tensor's shape differs from the current model's, and
    they are NOT loaded (the caller is expected to overwrite the parameter
    in-place afterwards, e.g. via a tile+scale upgrade).
    """
    reinit_keys = reinit_keys or set()
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
            if key not in reinit_keys:
                skipped_shape.append(key)
            continue
        filtered_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    if unexpected_keys:
        skipped_extra.extend(unexpected_keys)
    # ``reinit_keys`` will be initialized by the caller; suppress them from the
    # missing report so the validator does not treat them as an error.
    missing_keys = [key for key in missing_keys if key not in reinit_keys]
    return missing_keys, skipped_shape, skipped_extra


SPEAKER_IN_PROJ_WEIGHT_KEY = "speaker_encoder.in_proj.weight"


def _upgrade_speaker_in_proj(
    raw_model: torch.nn.Module,
    init_state: dict[str, torch.Tensor],
    *,
    old_patch: int,
    new_patch: int,
    is_main_process: bool,
) -> None:
    """
    Rebuild ``speaker_encoder.in_proj.weight`` when ``speaker_patch_size`` is
    increased by an integer factor ``k = new_patch // old_patch``.

    Strategy: tile the old weight ``k`` times along the input axis (the axis
    that grew due to speaker patching stacking ``k`` frames together in the
    channel dimension) and divide by ``k``. Averaging the ``k`` per-frame
    slices reproduces the old encoder's output when the ``k`` frames are
    identical, which gives a warm start close to the old model's behavior.

    Bias is unaffected by the patch change (shape ``(speaker_dim,)`` regardless
    of patch); it is loaded normally by ``load_model_state_partially``.
    """
    if new_patch <= old_patch or new_patch % old_patch != 0:
        raise ValueError(
            "speaker_patch_size upgrade must be a positive integer multiple: "
            f"old={old_patch} new={new_patch}"
        )
    factor = new_patch // old_patch
    old_weight = init_state.get(SPEAKER_IN_PROJ_WEIGHT_KEY)
    if old_weight is None:
        raise ValueError(
            f"Checkpoint is missing {SPEAKER_IN_PROJ_WEIGHT_KEY!r}; cannot upgrade "
            "speaker_patch_size."
        )
    speaker_encoder = getattr(raw_model, "speaker_encoder", None)
    if speaker_encoder is None or not hasattr(speaker_encoder, "in_proj"):
        raise RuntimeError(
            "Model does not expose speaker_encoder.in_proj; cannot upgrade speaker_patch_size."
        )
    target = speaker_encoder.in_proj.weight
    expected_out = int(target.shape[0])
    expected_in_new = int(target.shape[1])
    expected_in_old = expected_in_new // factor
    if tuple(old_weight.shape) != (expected_out, expected_in_old):
        raise ValueError(
            f"Checkpoint {SPEAKER_IN_PROJ_WEIGHT_KEY!r} has shape {tuple(old_weight.shape)}, "
            f"expected ({expected_out}, {expected_in_old}) for patch upgrade "
            f"{old_patch}->{new_patch}."
        )
    new_weight = old_weight.to(dtype=target.dtype).repeat(1, factor).contiguous() / float(factor)
    with torch.no_grad():
        target.data.copy_(new_weight.to(device=target.device))
    if is_main_process:
        print(
            f"Upgraded speaker_patch_size {old_patch}->{new_patch}: "
            f"reinitialized {SPEAKER_IN_PROJ_WEIGHT_KEY} by tiling old weight x{factor} "
            f"and scaling by 1/{factor}."
        )


def _canonical_parameter_key(key: str) -> str:
    prefix = "base_model.model."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def is_caption_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return (
        key.startswith(("caption_encoder.", "caption_norm."))
        or ".wk_caption." in key
        or ".wv_caption." in key
    )


def is_speaker_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return (
        key.startswith(("speaker_encoder.", "speaker_norm."))
        or ".wk_speaker." in key
        or ".wv_speaker." in key
    )


def is_duration_only_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return key.startswith("duration_predictor.")


def is_replaced_text_encoder_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return key.startswith(("pretrained_text_backbone.", "text_encoder.", "caption_encoder."))


def is_pretrained_projector_parameter(key: str) -> bool:
    key = _canonical_parameter_key(key)
    return key.startswith(("text_encoder.", "caption_encoder."))


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


def clear_non_pretrained_projector_grads(
    model: TextToLatentRFDiT,
) -> tuple[int, int]:
    projector_grad_params = 0
    cleared_grad_params = 0
    for key, param in model.named_parameters():
        if is_pretrained_projector_parameter(key):
            if param.grad is not None:
                projector_grad_params += 1
            continue
        if param.grad is not None:
            cleared_grad_params += 1
        param.grad = None
    return projector_grad_params, cleared_grad_params


def freeze_for_duration_only(model: torch.nn.Module) -> tuple[int, int]:
    trainable_params = 0
    frozen_params = 0
    for key, param in model.named_parameters():
        if is_duration_only_parameter(key):
            param.requires_grad_(True)
            trainable_params += param.numel()
        else:
            param.requires_grad_(False)
            frozen_params += param.numel()
    return trainable_params, frozen_params


def freeze_for_speaker_inversion(model: torch.nn.Module) -> tuple[int, int]:
    trainable_params = 0
    frozen_params = 0
    for key, param in model.named_parameters():
        if _canonical_parameter_key(key).startswith("speaker_inversion."):
            param.requires_grad_(True)
            trainable_params += param.numel()
        else:
            param.requires_grad_(False)
            frozen_params += param.numel()
    return trainable_params, frozen_params


def validate_checkpoint_upgrade_partial_load(  # noqa: PLR0913
    checkpoint_path: Path,
    missing_keys: list[str],
    skipped_shape: list[str],
    skipped_extra: list[str],
    *,
    allow_caption_missing: bool,
    allow_duration_missing: bool,
    allow_duration_extra: bool,
    allow_speaker_extra: bool,
    allow_text_encoder_replacement: bool = False,
) -> None:
    if skipped_shape:
        raise ValueError(
            "Checkpoint/config shape mismatch while upgrading checkpoint config: "
            f"{checkpoint_path} skipped_shape={skipped_shape[:8]}"
        )

    unexpected_extra = skipped_extra
    if allow_speaker_extra:
        unexpected_extra = [key for key in unexpected_extra if not is_speaker_only_parameter(key)]
    if allow_duration_extra:
        unexpected_extra = [key for key in unexpected_extra if not is_duration_only_parameter(key)]
    if allow_text_encoder_replacement:
        unexpected_extra = [
            key for key in unexpected_extra if not is_replaced_text_encoder_parameter(key)
        ]
    if unexpected_extra:
        raise ValueError(
            "Unexpected checkpoint keys while upgrading checkpoint config: "
            f"{checkpoint_path} skipped_extra={unexpected_extra[:8]}"
        )

    def _allowed_missing(key: str) -> bool:
        return (
            (allow_caption_missing and is_caption_only_parameter(key))
            or (allow_duration_missing and is_duration_only_parameter(key))
            or (allow_text_encoder_replacement and is_replaced_text_encoder_parameter(key))
        )

    unexpected_missing = [key for key in missing_keys if not _allowed_missing(key)]
    if unexpected_missing:
        raise ValueError(
            "Partial init from checkpoint left unexpected parameters missing: "
            f"{checkpoint_path} missing={unexpected_missing[:8]}"
        )


def _normalize_checkpoint_path(path: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))  # noqa: PTH100


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


def _initialize_base_model_from_pretrained_embeddings(  # noqa: C901
    raw_model: torch.nn.Module,
    *,
    model_cfg: ModelConfig,
    distributed: bool,
    is_main_process: bool,
) -> None:
    if model_cfg.use_pretrained_text_encoder:
        if is_main_process:
            print(
                "Using trainable pretrained text encoder with "
                f"condition projector(s): {model_cfg.text_tokenizer_repo}"
            )
        return
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


def _apply_base_initialization(  # noqa: C901, PLR0912, PLR0913, PLR0915
    raw_model: torch.nn.Module,
    *,
    model_cfg: ModelConfig,
    base_init: dict | None,
    distributed: bool,
    is_main_process: bool,
    preloaded_checkpoint: tuple[dict[str, torch.Tensor], dict | None, dict | None, dict | None]
    | None = None,
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
        if preloaded_checkpoint is None:
            init_state, init_model_cfg, _, _ = _load_model_state_from_checkpoint(init_path)
        else:
            init_state, init_model_cfg, _, _ = preloaded_checkpoint
        checkpoint_has_caption = checkpoint_uses_caption_condition(init_model_cfg, init_state)
        current_has_caption = bool(model_cfg.use_caption_condition)
        checkpoint_has_duration = checkpoint_uses_duration_predictor(init_model_cfg, init_state)
        current_has_duration = bool(model_cfg.use_duration_predictor)
        checkpoint_uses_pretrained_text_encoder = False
        if isinstance(init_model_cfg, dict):
            checkpoint_cfg = merge_dataclass_overrides(
                ModelConfig(),
                init_model_cfg,
                section="checkpoint model_config",
            )
            checkpoint_uses_pretrained_text_encoder = checkpoint_cfg.use_pretrained_text_encoder
        elif any(key.startswith("pretrained_text_backbone.") for key in init_state):
            checkpoint_uses_pretrained_text_encoder = True
        upgrade_text_encoder = bool(
            model_cfg.use_pretrained_text_encoder
            and not checkpoint_uses_pretrained_text_encoder
            and any(
                key.startswith(("text_encoder.text_embedding.", "text_encoder.blocks."))
                for key in init_state
            )
        )
        drop_duration = checkpoint_has_duration and not current_has_duration
        if checkpoint_has_caption and not current_has_caption:
            raise ValueError(
                "Caption-conditioned checkpoint cannot initialize a caption-free config. "
                "Use a caption-enabled config for this checkpoint."
            )
        if drop_duration and not (current_has_caption and not checkpoint_has_caption):
            raise ValueError(
                "Duration-predictor checkpoint cannot initialize a duration-free config. "
                "Use a duration-enabled config for this checkpoint, or initialize a "
                "caption-enabled phase-1 VoiceDesign model from a caption-free base checkpoint."
            )

        require_caption_match = checkpoint_has_caption and current_has_caption
        checkpoint_speaker_patch = None
        if isinstance(init_model_cfg, dict):
            checkpoint_speaker_patch = init_model_cfg.get("speaker_patch_size")
        upgrade_speaker_patch = bool(
            model_cfg.use_speaker_condition_resolved
            and checkpoint_speaker_patch is not None
            and int(checkpoint_speaker_patch) > 0
            and int(model_cfg.speaker_patch_size) > int(checkpoint_speaker_patch)
            and int(model_cfg.speaker_patch_size) % int(checkpoint_speaker_patch) == 0
        )
        _check_model_config_compatibility(
            init_path,
            init_model_cfg,
            model_cfg,
            require_caption_match=require_caption_match,
            upgrade_speaker_patch=upgrade_speaker_patch,
            upgrade_text_encoder=upgrade_text_encoder,
        )

        missing_keys: list[str] = []
        initialized_caption_embedding = False
        upgrade_caption = current_has_caption and not checkpoint_has_caption
        upgrade_duration = current_has_duration and not checkpoint_has_duration
        if (
            upgrade_caption
            or upgrade_duration
            or drop_duration
            or upgrade_speaker_patch
            or upgrade_text_encoder
        ):
            reinit_keys: set[str] = set()
            if upgrade_speaker_patch:
                reinit_keys.add(SPEAKER_IN_PROJ_WEIGHT_KEY)
            missing_keys, skipped_shape, skipped_extra = load_model_state_partially(
                raw_model,
                init_state,
                reinit_keys=reinit_keys,
            )
            validate_checkpoint_upgrade_partial_load(
                init_path,
                missing_keys,
                skipped_shape,
                skipped_extra,
                allow_caption_missing=upgrade_caption,
                allow_duration_missing=upgrade_duration,
                allow_duration_extra=drop_duration,
                allow_speaker_extra=(
                    upgrade_caption and not model_cfg.use_speaker_condition_resolved
                ),
                allow_text_encoder_replacement=upgrade_text_encoder,
            )
        else:
            raw_model.load_state_dict(init_state, strict=True)

        if upgrade_speaker_patch:
            _upgrade_speaker_in_proj(
                raw_model,
                init_state,
                old_patch=int(checkpoint_speaker_patch),
                new_patch=int(model_cfg.speaker_patch_size),
                is_main_process=is_main_process,
            )

        if upgrade_caption and not model_cfg.use_pretrained_text_encoder:
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

        if is_main_process:
            print(f"Initialized model weights from: {init_path}")
            if missing_keys:
                print(f"Partial load missing keys: {len(missing_keys)}")
            if current_has_duration and not checkpoint_has_duration:
                print("Duration predictor was randomly initialized.")
            if upgrade_text_encoder:
                print(
                    "Replaced checkpoint scratch text/caption encoders with "
                    "a trainable pretrained backbone and new projector(s): "
                    f"{model_cfg.text_tokenizer_repo}"
                )
            if initialized_caption_embedding:
                print("Caption embedding was initialized from its pretrained tokenizer backbone.")
        return

    raise ValueError(f"Unsupported base_init mode: {mode!r}")
