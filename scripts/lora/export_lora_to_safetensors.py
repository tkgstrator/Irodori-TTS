"""Export a PEFT LoRA checkpoint directory to a single .safetensors file.

The resulting file carries all PEFT metadata (adapter_config, base_init,
model_config, name/uuid/defaults) in the safetensors header so it can be
served by tts_server without the surrounding checkpoint directory.

Example
-------
  uv run python scripts/lora/export_lora_to_safetensors.py \
      --input outputs/<speaker>_lora/checkpoint_xxxxxxx \
      --output models/LoRA/<speaker>.safetensors \
      --name "<display name>" \
      --defaults '{"num_steps": 40, "cfg_scale_text": 3.0, "cfg_scale_speaker": 5.0}'

If --uuid is omitted, a deterministic UUIDv5 is derived from the output
filename so the server identifier stays stable across exports.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

EXPORT_FORMAT = "irodori-tts-lora/v1"
_NAMESPACE = uuid.UUID("8e6d8a0e-5a52-4a1e-8c8d-4c3e2f6a1b9f")


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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

    with safe_open(str(weights_path), framework="pt") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}

    metadata: dict[str, str] = {
        "format": EXPORT_FORMAT,
        "name": str(name),
        "adapter_config": json.dumps(adapter_config, ensure_ascii=False),
    }

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
    parser.add_argument("--name", required=True, help="Display name embedded in metadata")
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

    export(
        checkpoint_dir=args.input,
        output=args.output,
        name=args.name,
        defaults=defaults,
        speaker_uuid=args.uuid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
