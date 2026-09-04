"""Server configuration: speaker specs, YAML loading and checkpoint resolution."""

from __future__ import annotations

import json
import logging
import uuid as uuid_lib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from irodori_tts.lora import (
    is_lora_safetensors_file,
    read_lora_safetensors_metadata,
)

# Namespace for deterministic UUIDv5 derivation from LoRA filenames. Lets users
# drop .safetensors files into models/LoRA/ without assigning ids manually.
_LORA_UUID_NAMESPACE = uuid_lib.UUID("8e6d8a0e-5a52-4a1e-8c8d-4c3e2f6a1b9f")

# Upstream base checkpoints, keyed by the generation an adapter is trained
# against. Naming them here means a config picks a generation instead of
# restating a repo id, a filename and a local path that all have to agree.
_BASE_VERSIONS = {
    "v4.1-small": "Aratako/Irodori-TTS-v4.1-Small",
    "v4-small": "Aratako/Irodori-TTS-v4-Small",
    "v3": "Aratako/Irodori-TTS-500M-v3",
    "v2": "Aratako/Irodori-TTS-500M-v2",
    "v1": "Aratako/Irodori-TTS-500M",
}

logger = logging.getLogger("irodori_tts.server")


def _resolve_lora_display_name(meta: dict[str, str], fallback: str) -> str:
    for key in ("speaker.label", "name", "speaker"):
        value = meta.get(key)
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return fallback


@dataclass
class SpeakerSpec:
    uuid: str
    name: str
    adapter: str
    defaults: dict[str, Any] = field(default_factory=dict)
    category_id: str | None = None
    category_label: str | None = None
    cv: str | None = None


@dataclass
class ServerConfig:
    base_checkpoint: str | None
    base_hf_repo: str | None
    base_hf_filename: str
    model_device: str
    codec_device: str
    model_precision: str
    codec_precision: str
    codec_repo: str
    codec_deterministic_encode: bool
    codec_deterministic_decode: bool
    caption_checkpoint: str | None
    caption_hf_repo: str | None
    caption_hf_filename: str
    tail_window_size: int
    tail_std_threshold: float
    tail_mean_threshold: float
    show_timings: bool
    speakers: list[SpeakerSpec]


def _discover_lora_dir(lora_dir: Path) -> list[SpeakerSpec]:
    """Discover standalone .safetensors LoRA exports under ``lora_dir``.

    The search is recursive, so adapters may be grouped into subdirectories
    (the published set is laid out as ``<generation>/<category>/<speaker>``).

    Each file must carry Irodori-TTS metadata (``name``, ``uuid``,
    ``adapter_config``). ``defaults`` is optional.
    """
    if not lora_dir.is_dir():
        raise FileNotFoundError(f"lora_dir does not exist: {lora_dir}")
    specs: list[SpeakerSpec] = []
    for entry in sorted(lora_dir.rglob("*.safetensors")):
        if not is_lora_safetensors_file(entry):
            logger.warning("skipping non-LoRA safetensors file: %s", entry)
            continue
        try:
            meta = read_lora_safetensors_metadata(entry)
        except Exception as exc:
            logger.warning("failed to read metadata from %s: %s", entry, exc)
            continue
        name = _resolve_lora_display_name(meta, entry.stem)
        speaker_uuid = meta.get("uuid") or str(uuid_lib.uuid5(_LORA_UUID_NAMESPACE, entry.stem))
        defaults: dict[str, Any] = {}
        raw_defaults = meta.get("defaults")
        if raw_defaults:
            try:
                parsed = json.loads(raw_defaults)
                if isinstance(parsed, dict):
                    defaults = parsed
            except json.JSONDecodeError as exc:
                logger.warning("skipping defaults in %s: %s", entry, exc)
        category_id = str(meta.get("category.id") or "").strip() or None
        category_label = str(meta.get("category.label") or "").strip() or None
        cv = str(meta.get("speaker.cv") or "").strip() or None
        specs.append(
            SpeakerSpec(
                uuid=str(speaker_uuid),
                name=str(name),
                adapter=str(entry),
                defaults=defaults,
                category_id=category_id,
                category_label=category_label,
                cv=cv,
            )
        )
        logger.info("discovered LoRA: %s (uuid=%s)", name, speaker_uuid)
    return specs


def _resolve_base_repo(raw: dict[str, Any]) -> str | None:
    """Pick the base repo from ``base_version`` or an explicit ``base_hf_repo``.

    Setting both is rejected rather than silently ranked: the pair disagreeing
    is exactly the mistake the version alias exists to prevent.
    """
    version_raw = raw.get("base_version")
    version = str(version_raw).strip() if version_raw else ""
    explicit = str(raw["base_hf_repo"]) if raw.get("base_hf_repo") else None

    if not version:
        return explicit
    if explicit:
        raise ValueError(
            f"Set either base_version or base_hf_repo, not both (got {version!r} and {explicit!r})."
        )
    try:
        return _BASE_VERSIONS[version]
    except KeyError:
        known = ", ".join(sorted(_BASE_VERSIONS))
        raise ValueError(f"Unknown base_version: {version!r}. Known versions: {known}") from None


def load_config(path: Path) -> ServerConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # ValueError, not TypeError: matches how irodori_tts/config.py reports a bad config root.
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")  # noqa: TRY004

    speakers: list[SpeakerSpec] = []
    lora_dir_raw = raw.get("lora_dir")
    if lora_dir_raw:
        lora_dir = Path(str(lora_dir_raw))
        if not lora_dir.is_absolute():
            lora_dir = (path.parent / lora_dir).resolve() if not lora_dir.exists() else lora_dir
        speakers.extend(_discover_lora_dir(lora_dir))

    speakers.extend(
        SpeakerSpec(
            uuid=str(s["uuid"]),
            name=str(s["name"]),
            adapter=str(s["adapter"]),
            defaults=dict(s.get("defaults") or {}),
            category_id=(str(s["category_id"]).strip() or None) if s.get("category_id") else None,
            category_label=(str(s["category_label"]).strip() or None)
            if s.get("category_label")
            else None,
        )
        for s in raw.get("speakers") or []
    )

    return ServerConfig(
        base_checkpoint=(str(raw["base_checkpoint"]) if raw.get("base_checkpoint") else None),
        base_hf_repo=_resolve_base_repo(raw),
        base_hf_filename=str(raw.get("base_hf_filename", "model.safetensors")),
        model_device=str(raw.get("model_device", "cuda")),
        codec_device=str(raw.get("codec_device", "cuda")),
        model_precision=str(raw.get("model_precision", "bf16")),
        codec_precision=str(raw.get("codec_precision", "fp32")),
        codec_repo=str(raw.get("codec_repo", "Aratako/Semantic-DACVAE-Japanese-32dim")),
        codec_deterministic_encode=bool(raw.get("codec_deterministic_encode", True)),
        codec_deterministic_decode=bool(raw.get("codec_deterministic_decode", True)),
        caption_checkpoint=(
            str(raw["caption_checkpoint"]) if raw.get("caption_checkpoint") else None
        ),
        caption_hf_repo=(str(raw["caption_hf_repo"]) if raw.get("caption_hf_repo") else None),
        caption_hf_filename=str(raw.get("caption_hf_filename", "model.safetensors")),
        tail_window_size=int(raw.get("tail_window_size", 20)),
        tail_std_threshold=float(raw.get("tail_std_threshold", 0.05)),
        tail_mean_threshold=float(raw.get("tail_mean_threshold", 0.1)),
        show_timings=bool(raw.get("show_timings", True)),
        speakers=speakers,
    )


def _resolve_checkpoint(
    local_path: str | None,
    hf_repo: str | None,
    hf_filename: str,
    label: str,
) -> Path:
    local = Path(local_path) if local_path else None
    if local is not None and local.exists():
        logger.info("Using local %s checkpoint: %s", label, local)
        return local
    if not hf_repo:
        raise FileNotFoundError(
            f"{label} checkpoint not found at {local_path!r} and hf_repo is not set."
        )
    from huggingface_hub import hf_hub_download

    logger.info("Downloading %s checkpoint from HF: %s/%s", label, hf_repo, hf_filename)
    cached = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
    logger.info("%s checkpoint cached at: %s", label.capitalize(), cached)
    return Path(cached)


def resolve_base_checkpoint(cfg: ServerConfig) -> Path:
    return _resolve_checkpoint(
        cfg.base_checkpoint,
        cfg.base_hf_repo,
        cfg.base_hf_filename,
        "base",
    )
