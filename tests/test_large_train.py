from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.train.large_train import (
    Target,
    TrainSettings,
    build_command,
    latest_valid_checkpoint,
    load_targets,
    parse_gpus,
    run_training,
)

_REQUIRED = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "config.json",
    "manifest_size.txt",
    "trainer_state.pt",
)


def _target(tmp_path: Path, speaker_id: str, clips: int = 2) -> Target:
    root = tmp_path / "source" / speaker_id
    (root / "latents").mkdir(parents=True)
    manifest = root / "manifest.jsonl"
    manifest.write_text("{}\n" * clips, encoding="utf-8")
    return Target(speaker_id, "genshin_impact", manifest, clips)


def _checkpoint(path: Path, clips: int) -> Path:
    path.mkdir(parents=True)
    for name in _REQUIRED:
        (path / name).write_text(str(clips) if name == "manifest_size.txt" else "state")
    return path


def test_gpu_indices_must_be_explicit_and_distinct() -> None:
    assert parse_gpus("0 2,7") == [0, 2, 7]
    for value in ("", "2 2", "0 -1", "all"):
        with pytest.raises(ValueError):
            parse_gpus(value)


def test_targets_are_sorted_and_reject_unsafe_ids(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "gi_amber",
                        "category": "genshin_impact",
                        "manifest_path": "/data/a",
                        "clips": 4,
                    },
                    {
                        "id": "gi_ayaka",
                        "category": "genshin_impact",
                        "manifest_path": "/data/b",
                        "clips": 10,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert [target.id for target in load_targets(path)] == ["gi_ayaka", "gi_amber"]
    path.write_text(
        json.dumps([{"id": "../escape", "category": "x", "manifest_path": "/data/a", "clips": 1}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid canonical speaker id"):
        load_targets(path)


def test_checkpoint_picker_rejects_partial_and_wrong_manifest(tmp_path: Path) -> None:
    out = tmp_path / "output"
    _checkpoint(out / "checkpoint_0000100", 2)
    _checkpoint(out / "checkpoint_0000200", 3)
    (out / "checkpoint_0000300").mkdir()
    assert latest_valid_checkpoint(out, 2) == out / "checkpoint_0000100"
    with pytest.raises(ValueError, match="no complete checkpoint"):
        latest_valid_checkpoint(out, 4)


def test_resume_uses_same_large_base_and_generation_run_name(tmp_path: Path) -> None:
    target = _target(tmp_path, "gi_ayaka")
    resume = _checkpoint(tmp_path / "outputs" / "gi_ayaka_lora" / "checkpoint_0000100", 2)
    settings = TrainSettings(
        config=Path("configs/train_v4_large_lora.yaml"),
        base=Path("models/Irodori-TTS-v4-Large/model.safetensors"),
        output_root=resume.parent.parent,
        lock_root=tmp_path / "locks",
        project="irodori-tts-v4-large",
        cwd=tmp_path,
    )
    command = build_command(target, settings, resume.parent, resume)
    assert command[command.index("--resume") + 1] == str(resume)
    assert command[command.index("--init-checkpoint") + 1].endswith(
        "Irodori-TTS-v4-Large/model.safetensors"
    )
    assert command[command.index("--metrics-backend") + 1] == "atmos"
    assert command[command.index("--metrics-run-name") + 1] == "gi_ayaka_lora_v4_large"


def test_preflight_refuses_missing_credentials_without_claiming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target(tmp_path, "gi_ayaka")
    monkeypatch.delenv("ATMOS_TOKEN", raising=False)
    monkeypatch.delenv("ATMOS_BASE_URL", raising=False)
    command = Mock()
    monkeypatch.setattr(subprocess, "run", command)
    with pytest.raises(ValueError, match="ATMOS_TOKEN"):
        run_training(
            [target],
            [0],
            TrainSettings(
                config=tmp_path / "config.yaml",
                base=tmp_path / "model.safetensors",
                output_root=tmp_path / "outputs_v4_large",
                lock_root=tmp_path / "locks",
                project="irodori-tts-v4-large",
                cwd=tmp_path,
            ),
        )
    command.assert_not_called()
    assert not (tmp_path / "locks").exists()


def test_worker_marks_done_only_with_complete_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets = [_target(tmp_path, "gi_amber", 2), _target(tmp_path, "gi_ayaka", 4)]
    config = tmp_path / "config.yaml"
    config.write_text("model: large")
    base = tmp_path / "model.safetensors"
    base.write_text("base")
    monkeypatch.setenv("ATMOS_TOKEN", "test-token")
    monkeypatch.setenv("ATMOS_BASE_URL", "http://testserver")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        out = Path(command[command.index("--output-dir") + 1])
        clips = 4 if "gi_ayaka" in out.name else 2
        _checkpoint(out / "checkpoint_final", clips)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_training(
        targets,
        [1, 3],
        TrainSettings(
            config=config,
            base=base,
            output_root=tmp_path / "outputs_v4_large",
            lock_root=tmp_path / "locks",
            project="irodori-tts-v4-large",
            cwd=tmp_path,
        ),
    )
    assert result == 0
    assert len(calls) == 2
    assert {(tmp_path / "locks" / target.id / "done.json").is_file() for target in targets} == {
        True
    }
    assert not list((tmp_path / "locks").rglob("failed.json"))


def test_failed_job_keeps_claim_and_is_not_marked_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target(tmp_path, "gi_ayaka")
    config = tmp_path / "config.yaml"
    config.write_text("model: large")
    base = tmp_path / "model.safetensors"
    base.write_text("base")
    monkeypatch.setenv("ATMOS_TOKEN", "test-token")
    monkeypatch.setenv("ATMOS_BASE_URL", "http://testserver")
    run = Mock(return_value=subprocess.CompletedProcess([], 1))
    monkeypatch.setattr(subprocess, "run", run)
    settings = TrainSettings(
        config=config,
        base=base,
        output_root=tmp_path / "outputs_v4_large",
        lock_root=tmp_path / "locks",
        project="irodori-tts-v4-large",
        cwd=tmp_path,
    )
    assert run_training([target], [0], settings) == 1
    claim = tmp_path / "locks" / target.id
    assert (claim / "failed.json").is_file()
    assert not (claim / "done.json").exists()
    assert run_training([target], [0], settings) == 1
    run.assert_called_once()


def test_success_without_final_checkpoint_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target(tmp_path, "gi_ayaka")
    config = tmp_path / "config.yaml"
    config.write_text("model: large")
    base = tmp_path / "model.safetensors"
    base.write_text("base")
    monkeypatch.setenv("ATMOS_TOKEN", "test-token")
    monkeypatch.setenv("ATMOS_BASE_URL", "http://testserver")
    monkeypatch.setattr(
        subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0)
    )
    settings = TrainSettings(
        config=config,
        base=base,
        output_root=tmp_path / "outputs_v4_large",
        lock_root=tmp_path / "locks",
        project="irodori-tts-v4-large",
        cwd=tmp_path,
    )
    assert run_training([target], [0], settings) == 1
    claim = tmp_path / "locks" / target.id
    assert (claim / "failed.json").is_file()
    assert not (claim / "done.json").exists()


def test_existing_checkpoint_resumes_without_starting_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target(tmp_path, "gi_ayaka")
    config = tmp_path / "config.yaml"
    config.write_text("model: large")
    base = tmp_path / "model.safetensors"
    base.write_text("base")
    output = tmp_path / "outputs_v4_large" / f"{target.id}_lora"
    resume = _checkpoint(output / "checkpoint_0000100", target.clips)
    monkeypatch.setenv("ATMOS_TOKEN", "test-token")
    monkeypatch.setenv("ATMOS_BASE_URL", "http://testserver")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        _checkpoint(output / "checkpoint_final", target.clips)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    settings = TrainSettings(
        config=config,
        base=base,
        output_root=tmp_path / "outputs_v4_large",
        lock_root=tmp_path / "locks",
        project="irodori-tts-v4-large",
        cwd=tmp_path,
    )
    assert run_training([target], [0], settings) == 0
    assert calls[0][calls[0].index("--resume") + 1] == str(resume)
    assert (tmp_path / "locks" / target.id / "done.json").is_file()
