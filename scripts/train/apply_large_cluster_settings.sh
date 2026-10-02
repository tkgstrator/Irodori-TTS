#!/usr/bin/env bash
# Update only the next-run recipe; do not start or resume any training job.
set -euo pipefail
host=${1:-g20}

ssh -o BatchMode=yes "$host" 'docker run --rm -i --entrypoint python -v /home/smorimoto/Developer/Irodori-TTS-v4-large:/app irodori-tts-train:v4-large -' <<'PY'
from datetime import UTC, datetime
import os
from pathlib import Path
import re
import tempfile

path = Path("/app/configs/train_v4_large_lora.yaml")
original = path.read_text()
sections = re.split(r"(?m)^(?=[a-z_]+:\s*$)", original)
updates = {
    "train": {
        "batch_size": "40",
        "gradient_accumulation_steps": "2",
        "gradient_checkpointing": "false",
        "num_workers": "8",
        "dataloader_persistent_workers": "true",
        "save_every": "250",
        "valid_every": "100",
    },
    "sample_generation": {"every": "250", "on_best_val": "false"},
}
seen = set()
for index, section in enumerate(sections):
    for name, fields in updates.items():
        if not section.startswith(name + ":"):
            continue
        if name in seen:
            raise RuntimeError(f"Duplicate section: {name}")
        seen.add(name)
        for key, value in fields.items():
            pattern = rf"(?m)^  {re.escape(key)}:.*$"
            section, count = re.subn(pattern, f"  {key}: {value}", section)
            if count != 1:
                raise RuntimeError(f"Expected exactly one {name}.{key}, found {count}")
        sections[index] = section
if seen != set(updates):
    raise RuntimeError("Missing train or sample_generation section")
updated = "".join(sections)
if updated == original:
    print("Settings already applied; no files changed.")
else:
    backup_dir = Path("/app/outputs/config-backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = backup_dir / f"train_v4_large_lora-{stamp}.yaml"
    with backup.open("x") as handle:
        handle.write(original)
    stat = path.stat()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
        temporary.chmod(stat.st_mode & 0o777)
        os.chown(temporary, stat.st_uid, stat.st_gid)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    if path.read_text() != updated:
        raise RuntimeError("Configuration read-back did not match")
    print(f"Updated {path}; original saved at {backup}")
for section, fields in updates.items():
    for key, value in fields.items():
        print(f"{section}.{key}={value}")
print("Existing outputs/checkpoints and early-stop settings were not changed. No job was started.")
PY
