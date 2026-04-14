from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .config import TrainConfig
from .model import TextToLatentRFDiT

LORA_TRAIN_CONFIG_FIELDS = (
    "lora_enabled",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_bias",
    "lora_target_modules",
)

LORA_ADAPTER_CONFIG_NAME = "adapter_config.json"
LORA_ADAPTER_STATE_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
LORA_TRAINER_STATE_NAME = "trainer_state.pt"
LORA_METADATA_NAME = "irodori_lora_metadata.json"

LORA_TARGET_PRESETS: dict[str, str] = {
    "text_attn_mlp": (
        r"^text_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))$"
    ),
    "caption_attn_mlp": (
        r"^caption_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))$"
    ),
    "speaker_attn_mlp": (
        r"^(speaker_encoder\.in_proj"
        r"|speaker_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3)))$"
    ),
    "diffusion_attn": (
        r"^blocks\.\d+\.attention\."
        r"(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)$"
    ),
    "diffusion_attn_mlp": (
        r"^blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
        r"|mlp\.(w1|w2|w3))$"
    ),
    "all_attn": (
        r"^(text_encoder\.blocks\.\d+\.attention\.(wq|wk|wv|wo|gate)"
        r"|caption_encoder\.blocks\.\d+\.attention\.(wq|wk|wv|wo|gate)"
        r"|speaker_encoder\.blocks\.\d+\.attention\.(wq|wk|wv|wo|gate)"
        r"|blocks\.\d+\.attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate))$"
    ),
    "diffusion_full": (
        r"^(cond_module\.(0|2|4)"
        r"|in_proj"
        r"|out_proj"
        r"|blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
        r"|mlp\.(w1|w2|w3)"
        r"|attention_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)"
        r"|mlp_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)))$"
    ),
    "adaln": (
        r"^blocks\.\d+\."
        r"(attention_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)"
        r"|mlp_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up))$"
    ),
    "conditioning": (
        r"^(cond_module\.(0|2|4)"
        r"|speaker_encoder\.in_proj"
        r"|blocks\.\d+\.attention\.(wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption))$"
    ),
    "all_attn_mlp": (
        r"^(text_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|caption_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|speaker_encoder\.in_proj"
        r"|speaker_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
        r"|mlp\.(w1|w2|w3)))$"
    ),
    "all_linear": (
        r"^(speaker_encoder\.in_proj"
        r"|cond_module\.(0|2|4)"
        r"|in_proj"
        r"|out_proj"
        r"|text_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|caption_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|speaker_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate|wo)"
        r"|mlp\.(w1|w2|w3)"
        r"|attention_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)"
        r"|mlp_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)))$"
    ),
}


def _require_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA fine-tuning requires `peft`. Install with `pip install peft` or `uv sync`."
        ) from exc
    return LoraConfig, PeftModel, get_peft_model


def _lookup_config_value(raw: TrainConfig | Mapping[str, Any] | None, field: str) -> Any:
    if raw is None:
        return getattr(TrainConfig(), field)
    if isinstance(raw, TrainConfig):
        return getattr(raw, field)
    if isinstance(raw, Mapping):
        if field in raw:
            return raw[field]
        return getattr(TrainConfig(), field)
    raise TypeError(f"Unsupported LoRA config source: {type(raw)!r}")


def train_config_uses_lora(raw: TrainConfig | Mapping[str, Any] | None) -> bool:
    return bool(_lookup_config_value(raw, "lora_enabled"))


def checkpoint_state_uses_lora(model_state: Mapping[str, torch.Tensor]) -> bool:
    return any(key.startswith("base_model.model.") or ".lora_" in key for key in model_state)


def resolve_lora_target_modules(spec: str | Sequence[str] | None) -> str | list[str]:
    if spec is None:
        spec = TrainConfig().lora_target_modules

    if isinstance(spec, str):
        value = spec.strip()
        if not value:
            raise ValueError("lora_target_modules must not be empty.")
        preset = LORA_TARGET_PRESETS.get(value)
        if preset is not None:
            return preset
        if "," in value:
            modules = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
            if not modules:
                raise ValueError(f"Invalid LoRA target_modules list: {spec!r}")
            return modules
        return value

    modules = [str(item).strip() for item in spec if str(item).strip()]
    if not modules:
        raise ValueError("LoRA target_modules sequence must not be empty.")
    return modules


def build_lora_config_kwargs(raw: TrainConfig | Mapping[str, Any]) -> dict[str, Any]:
    bias = str(_lookup_config_value(raw, "lora_bias")).strip().lower()
    if bias not in {"none", "all", "lora_only"}:
        raise ValueError(f"Unsupported lora_bias={bias!r}. Expected one of: none, all, lora_only.")

    return {
        "r": int(_lookup_config_value(raw, "lora_r")),
        "lora_alpha": int(_lookup_config_value(raw, "lora_alpha")),
        "lora_dropout": float(_lookup_config_value(raw, "lora_dropout")),
        "bias": bias,
        "target_modules": resolve_lora_target_modules(
            _lookup_config_value(raw, "lora_target_modules")
        ),
    }


def apply_lora(
    model: TextToLatentRFDiT,
    raw: TrainConfig | Mapping[str, Any],
) -> torch.nn.Module:
    if not train_config_uses_lora(raw):
        return model

    lora_config_cls, _, get_peft_model = _require_peft()
    peft_model = get_peft_model(
        model,
        lora_config_cls(
            task_type=None,
            inference_mode=False,
            **build_lora_config_kwargs(raw),
        ),
    )
    return peft_model


def is_lora_adapter_dir(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    if not (candidate / LORA_ADAPTER_CONFIG_NAME).is_file():
        return False
    return any((candidate / name).is_file() for name in LORA_ADAPTER_STATE_NAMES)


def is_lora_safetensors_file(path: str | Path) -> bool:
    """Return True if `path` is a standalone LoRA .safetensors export."""
    candidate = Path(path)
    if not candidate.is_file() or candidate.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(str(candidate), framework="pt") as f:
            meta = f.metadata() or {}
    except Exception:
        return False
    return "adapter_config" in meta


def is_lora_adapter_path(path: str | Path) -> bool:
    return is_lora_adapter_dir(path) or is_lora_safetensors_file(path)


def read_lora_safetensors_metadata(path: str | Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


def unpack_lora_safetensors(path: str | Path, *, dest_dir: str | Path | None = None) -> Path:
    """Materialize a single-file LoRA .safetensors into a PEFT-style directory.

    Returns the directory containing ``adapter_config.json`` and
    ``adapter_model.safetensors`` so PEFT's ``from_pretrained`` can load it.
    When ``dest_dir`` is ``None`` a throwaway temp directory is created; callers
    are responsible for cleaning it up if they care.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"LoRA safetensors file not found: {src}")

    with safe_open(str(src), framework="pt") as f:
        metadata = dict(f.metadata() or {})
        tensors = {key: f.get_tensor(key) for key in f.keys()}

    adapter_config_raw = metadata.get("adapter_config")
    if not adapter_config_raw:
        raise ValueError(
            f"{src} is not an Irodori-TTS LoRA export "
            f"(missing 'adapter_config' in safetensors metadata)."
        )
    adapter_config = json.loads(adapter_config_raw)

    if dest_dir is None:
        target = Path(tempfile.mkdtemp(prefix="irodori_lora_"))
    else:
        target = Path(dest_dir)
        target.mkdir(parents=True, exist_ok=True)

    (target / LORA_ADAPTER_CONFIG_NAME).write_text(
        json.dumps(adapter_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_file(tensors, str(target / "adapter_model.safetensors"))
    return target


def load_lora_adapter(
    model: TextToLatentRFDiT,
    adapter_path: str | Path,
    *,
    is_trainable: bool,
) -> torch.nn.Module:
    _, peft_model_cls, _ = _require_peft()
    resolved = Path(adapter_path)
    if is_lora_safetensors_file(resolved):
        resolved = unpack_lora_safetensors(resolved)
    return peft_model_cls.from_pretrained(model, str(resolved), is_trainable=is_trainable)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    total = sum(int(param.numel()) for param in model.parameters())
    return trainable, total
