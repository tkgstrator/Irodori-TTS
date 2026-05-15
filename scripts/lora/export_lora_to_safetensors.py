"""Export a PEFT LoRA checkpoint directory to a single .safetensors file.

The resulting file carries all PEFT metadata (adapter_config, base_init,
model_config, name/uuid/defaults) in the safetensors header so it can be
served by server.py without the surrounding checkpoint directory.

Example
-------
  uv run python scripts/lora/export_lora_to_safetensors.py \
      --input outputs/<speaker>_lora/checkpoint_xxxxxxx \
      --output models/LoRA/<speaker>.safetensors \
      --defaults '{"num_steps": 40, "cfg_scale_text": 3.0, "cfg_scale_speaker": 5.0}'

If --uuid is omitted, a deterministic UUIDv5 is derived from the output
filename so the server identifier stays stable across exports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml
from safetensors import safe_open
from safetensors.torch import save_file

EXPORT_FORMAT = "irodori-tts-lora/v1"
_NAMESPACE = uuid.UUID("8e6d8a0e-5a52-4a1e-8c8d-4c3e2f6a1b9f")


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_saved_path(raw_path: str, *, checkpoint_dir: Path) -> Path | None:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None

    search_roots = [Path.cwd(), checkpoint_dir, *checkpoint_dir.parents]
    seen: set[Path] = set()
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _infer_speaker_id_from_checkpoint_dir(checkpoint_dir: Path) -> str | None:
    for candidate in (checkpoint_dir, *checkpoint_dir.parents):
        name = candidate.name.strip()
        if name.endswith("_lora") and len(name) > len("_lora"):
            return name[: -len("_lora")]
    return None


def _resolve_dataset_config_path(checkpoint_dir: Path) -> Path | None:
    speaker_id = _infer_speaker_id_from_checkpoint_dir(checkpoint_dir)
    if not speaker_id:
        return None

    search_roots = [Path.cwd(), checkpoint_dir, *checkpoint_dir.parents]
    seen: set[Path] = set()
    for root in search_roots:
        candidate = (root / "data" / speaker_id / "config.yaml").resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _load_dataset_config(checkpoint_dir: Path) -> dict[str, Any] | None:
    saved_config = _read_json(checkpoint_dir / "config.json")
    config_path: Path | None = None

    if isinstance(saved_config, dict):
        train_cfg = saved_config.get("train")
        if isinstance(train_cfg, dict):
            manifest_path = train_cfg.get("manifest_path")
            if isinstance(manifest_path, str) and manifest_path.strip():
                resolved_manifest = _resolve_saved_path(manifest_path, checkpoint_dir=checkpoint_dir)
                if resolved_manifest is not None:
                    candidate = resolved_manifest.parent / "config.yaml"
                    if candidate.is_file():
                        config_path = candidate

    if config_path is None:
        config_path = _resolve_dataset_config_path(checkpoint_dir)
    if config_path is None:
        return None

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else None


def _flatten_dataset_metadata(dataset_config: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(dataset_config, dict):
        return {}

    metadata: dict[str, str] = {}

    speaker = dataset_config.get("speaker")
    if isinstance(speaker, dict):
        for field in ("label", "cv"):
            value = speaker.get(field)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                metadata[f"speaker.{field}"] = text

    category = dataset_config.get("category")
    if isinstance(category, dict):
        for field in ("id", "label"):
            value = category.get(field)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                metadata[f"category.{field}"] = text

    return metadata


def _resolve_export_speaker_id(
    *,
    dataset_config: dict[str, Any] | None,
    checkpoint_dir: Path,
    existing_metadata: dict[str, str],
) -> str | None:
    if isinstance(dataset_config, dict):
        speaker = dataset_config.get("speaker")
        if isinstance(speaker, dict):
            value = speaker.get("id")
            text = str(value).strip() if value is not None else ""
            if text:
                return text

    inferred = _infer_speaker_id_from_checkpoint_dir(checkpoint_dir)
    if inferred:
        return inferred

    raw = str(existing_metadata.get("speaker") or "").strip()
    if not raw:
        return None
    return re.sub(r"_v\d+$", "", raw, flags=re.IGNORECASE).lower()


def _resolve_model_name(base_model: str | None) -> str | None:
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


def _resolve_export_name(
    *,
    requested_name: str | None,
    dataset_config: dict[str, Any] | None,
) -> str:
    provided = str(requested_name or "").strip()

    if isinstance(dataset_config, dict):
        speaker = dataset_config.get("speaker")
        if isinstance(speaker, dict):
            for field in ("label", "name"):
                value = speaker.get(field)
                text = str(value).strip() if value is not None else ""
                if text:
                    if provided and provided != text:
                        print(
                            f"warning: ignoring --name={provided!r}; using canonical speaker.{field}={text!r}",
                            file=sys.stderr,
                        )
                    return text

    if provided:
        return provided

    raise ValueError(
        "Could not determine export name. Add speaker.label to data/<speaker>/config.yaml "
        "or pass --name explicitly."
    )


def export(
    *,
    checkpoint_dir: Path,
    output: Path,
    name: str,
    defaults: dict,
    speaker_uuid: str | None,
) -> None:
    weights_path = checkpoint_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"adapter_model.safetensors not found under {checkpoint_dir}")
    adapter_config = _read_json(checkpoint_dir / "adapter_config.json")
    if adapter_config is None:
        raise FileNotFoundError(f"adapter_config.json not found under {checkpoint_dir}")
    dataset_config = _load_dataset_config(checkpoint_dir)
    resolved_name = _resolve_export_name(requested_name=name, dataset_config=dataset_config)

    with safe_open(str(weights_path), framework="pt") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
        metadata = dict(f.metadata() or {})

    resolved_speaker_id = _resolve_export_speaker_id(
        dataset_config=dataset_config,
        checkpoint_dir=checkpoint_dir,
        existing_metadata=metadata,
    )
    resolved_model_name = _resolve_model_name(metadata.get("base_model"))

    metadata.update(_flatten_dataset_metadata(dataset_config))
    metadata.update({
        "format": EXPORT_FORMAT,
        "name": resolved_name,
        "adapter_config": json.dumps(adapter_config, ensure_ascii=False),
    })
    if resolved_speaker_id:
        metadata["speaker"] = resolved_speaker_id
    if resolved_model_name:
        metadata["model_name"] = resolved_model_name

    resolved_uuid = speaker_uuid
    if not resolved_uuid:
        resolved_uuid = str(uuid.uuid5(_NAMESPACE, output.stem))
    metadata["uuid"] = str(resolved_uuid)

    if defaults:
        metadata["defaults"] = json.dumps(defaults, ensure_ascii=False)

    base_init = _read_json(checkpoint_dir / "base_init.json")
    if base_init is not None:
        metadata["base_init"] = json.dumps(base_init, ensure_ascii=False)

    model_config = _read_json(checkpoint_dir / "config.json")
    if model_config is not None:
        metadata["model_config"] = json.dumps(model_config, ensure_ascii=False)

    manifest_size_path = checkpoint_dir / "manifest_size.txt"
    if manifest_size_path.is_file():
        value = manifest_size_path.read_text(encoding="utf-8").strip()
        if value:
            metadata["manifest_size"] = value

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata=metadata)
    print(f"wrote {output} ({len(tensors)} tensors, uuid={resolved_uuid})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="PEFT checkpoint directory")
    parser.add_argument("--output", required=True, type=Path, help="Output .safetensors path")
    parser.add_argument(
        "--name",
        default=None,
        help="Display name embedded in metadata when dataset config has no speaker.label",
    )
    parser.add_argument("--uuid", default=None, help="Explicit UUID (default: uuid5 from filename)")
    parser.add_argument(
        "--defaults",
        default="{}",
        help="JSON mapping of inference defaults embedded in metadata",
    )
    args = parser.parse_args(argv)

    try:
        defaults = json.loads(args.defaults) if args.defaults else {}
    except json.JSONDecodeError as exc:
        print(f"error: invalid --defaults JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(defaults, dict):
        print("error: --defaults must decode to a JSON object", file=sys.stderr)
        return 2

    try:
        export(
            checkpoint_dir=args.input,
            output=args.output,
            name=args.name,
            defaults=defaults,
            speaker_uuid=args.uuid,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
