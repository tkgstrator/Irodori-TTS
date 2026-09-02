"""Speaker metadata resolution, sample prompt building and LoRA adapter metadata."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

from irodori_tts.config import SamplePromptConfig, TrainConfig


def _load_speaker_yaml(manifest_path: str | Path | None) -> dict | None:
    if manifest_path is None:
        return None
    cfg_path = Path(manifest_path).parent / "config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        import yaml

        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _resolve_speaker_name(manifest_path: str | Path | None) -> str | None:
    if manifest_path is None:
        return None
    speaker_dir = Path(manifest_path).parent
    speaker_id = speaker_dir.name or None
    data = _load_speaker_yaml(manifest_path)
    if data is None:
        return speaker_id
    speaker = (data.get("speaker") or {}) if isinstance(data, dict) else {}
    return speaker.get("label") or speaker.get("name") or speaker.get("id") or speaker_id


def _resolve_speaker_id(manifest_path: str | Path | None) -> str | None:
    if manifest_path is None:
        return None
    speaker_dir = Path(manifest_path).parent
    data = _load_speaker_yaml(manifest_path)
    if isinstance(data, dict):
        speaker = (data.get("speaker") or {}) if isinstance(data, dict) else {}
        value = speaker.get("id")
        if value is not None:
            text = str(value).strip()
            if text:
                return text

    raw = speaker_dir.name.strip()
    if not raw:
        return None
    return re.sub(r"_v\d+$", "", raw, flags=re.IGNORECASE).lower()


def _resolve_base_model_name(base_model: str | None) -> str | None:
    if base_model is None:
        return None
    text = str(base_model).strip()
    if not text:
        return None

    lowered = text.lower()
    if "irodori-tts" not in lowered:
        return None

    match = re.search(r"v\d+", lowered)
    if match is None:
        return None

    suffix = "-VoiceDesign" if "voicedesign" in lowered else ""
    return f"Irodori-TTS/{match.group(0)}{suffix}"


def _build_prompts_from_speaker_config(
    manifest_path: str | Path | None,
) -> list[SamplePromptConfig]:
    data = _load_speaker_yaml(manifest_path)
    if data is None:
        return []
    texts = data.get("sample_texts") or []
    if not isinstance(texts, list):
        return []
    prompts: list[SamplePromptConfig] = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            continue
        prompts.append(SamplePromptConfig(name=f"sample_{i:02d}", text=t.strip()))
    return prompts


def _autopick_prompts_from_manifest(
    manifest_path: str | Path | None,
    *,
    n: int = 5,
    min_len: int = 10,
    max_len: int = 60,
) -> list[SamplePromptConfig]:
    """Pick `n` length-balanced texts from the training manifest.

    Used as a fallback when data/<speaker>/config.yaml is missing or has no
    sample_texts. Filters to texts in [min_len, max_len] characters, sorts by
    length, and picks evenly-spaced quantiles so the sample set spans short /
    medium / long utterances. Deterministic for a given manifest.
    """
    if manifest_path is None:
        return []
    try:
        with Path(manifest_path).open(encoding="utf-8") as f:
            texts: list[str] = []
            seen: set[str] = set()
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = (item.get("text") or "").strip()
                if t and t not in seen:
                    seen.add(t)
                    texts.append(t)
    except OSError:
        return []
    candidates = [t for t in texts if min_len <= len(t) <= max_len]
    if not candidates:
        candidates = sorted(set(texts), key=len)
    if not candidates:
        return []
    candidates.sort(key=len)
    m = len(candidates)
    if m <= n:
        picks = candidates
    else:
        quantiles = [round((i + 0.5) / n * m) - 1 for i in range(n)]
        picks = [candidates[max(0, min(m - 1, q))] for q in quantiles]
    return [SamplePromptConfig(name=f"sample_{i:02d}", text=t) for i, t in enumerate(picks)]


def _build_lora_safetensors_metadata(  # noqa: PLR0913
    *,
    run_uuid: str | None,
    run_name: str | None,
    speaker_id: str | None,
    base_model: str | None,
    step: int,
    optim_steps_per_epoch: int | None,
    train_cfg: TrainConfig,
    val_loss: float | None,
) -> dict[str, str]:
    meta: dict[str, str] = {}
    if run_uuid:
        meta["uuid"] = str(run_uuid)
    if base_model:
        meta["base_model"] = str(base_model)
    resolved_model_name = _resolve_base_model_name(base_model)
    if resolved_model_name:
        meta["model_name"] = resolved_model_name
    elif run_name:
        meta["model_name"] = str(run_name)
    if run_name:
        meta["run_name"] = str(run_name)
    if speaker_id:
        meta["speaker"] = str(speaker_id)
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
        # safe_open exposes keys() but is not iterable, so SIM118's fix would break here.
        tensor_keys = f.keys()
        for key in tensor_keys:
            tensors[key] = f.get_tensor(key)
    merged = {**existing_meta, **{k: v for k, v in extra_metadata.items() if v is not None}}
    save_file(tensors, str(adapter_path), metadata=merged)
