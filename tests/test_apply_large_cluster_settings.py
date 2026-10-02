"""Check configuration updates without connecting to the cluster."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _program(tmp_path: Path) -> str:
    script = (ROOT / "scripts/train/apply_large_cluster_settings.sh").read_text()
    source = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    return source.replace(
        "/app/configs/train_v4_large_lora.yaml", str(tmp_path / "config.yaml")
    ).replace("/app/outputs/config-backups", str(tmp_path / "backups"))


def test_updates_only_selected_values_and_keeps_backup(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs/train_v4_large_lora.yaml").read_text())
    config["model"]["flow_parameterization"] = "rf_velocity"
    config["train"].update(
        gradient_checkpointing=True,
        num_workers=16,
        dataloader_persistent_workers=False,
        save_every=100,
        valid_every=1000,
        early_stop_enabled=True,
        early_stop_patience=2,
    )
    config["sample_generation"].update(every=500, on_best_val=True)
    original = yaml.safe_dump(config, sort_keys=False)
    path = tmp_path / "config.yaml"
    path.write_text(original)
    exec(compile(_program(tmp_path), "apply-cluster-settings", "exec"), {})
    updated = yaml.safe_load(path.read_text())
    assert updated["train"]["gradient_checkpointing"] is False
    assert updated["train"]["batch_size"] == 40
    assert updated["train"]["gradient_accumulation_steps"] == 2
    assert updated["train"]["num_workers"] == 8
    assert updated["train"]["dataloader_persistent_workers"] is True
    assert updated["train"]["save_every"] == 250
    assert updated["train"]["valid_every"] == 100
    assert updated["sample_generation"]["every"] == 250
    assert updated["sample_generation"]["on_best_val"] is False
    assert updated["model"] == config["model"]
    assert updated["train"]["early_stop_patience"] == 2
    assert updated["train"]["early_stop_enabled"] is True
    backups = list((tmp_path / "backups").glob("*.yaml"))
    assert len(backups) == 1
    assert backups[0].read_text() == original
    exec(compile(_program(tmp_path), "apply-cluster-settings", "exec"), {})
    assert len(list((tmp_path / "backups").glob("*.yaml"))) == 1


def test_missing_key_refuses_to_write(tmp_path: Path) -> None:
    original = (
        (ROOT / "configs/train_v4_large_lora.yaml").read_text().replace("  num_workers: 8\n", "")
    )
    path = tmp_path / "config.yaml"
    path.write_text(original)
    with pytest.raises(RuntimeError, match=r"train\.num_workers"):
        exec(compile(_program(tmp_path), "apply-cluster-settings", "exec"), {})
    assert path.read_text() == original
    assert not (tmp_path / "backups").exists()
