from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import make_dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, mock_open

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train/probe_large_batch.sh"


@pytest.fixture
def shell_env(tmp_path: Path) -> dict[str, str]:
    commands = {
        "ssh": """#!/usr/bin/env python3
import os, subprocess, sys
assert sys.argv[1:3] == ["-o", "BatchMode=yes"]
assert sys.argv[4:7] == ["bash", "-s", "--"]
result = subprocess.run(["bash", "-s", "--", *sys.argv[7:]], input=sys.stdin.read(), text=True)
sys.exit(result.returncode)
""",
        "nvidia-smi": "#!/bin/sh\nprintf '0\\n0\\n0\\n'\n",
        "docker": """#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
if args[0] == "inspect":
    print("/read-only-source" if ".Mounts" in args[-1] else "probe-image")
else:
    with open(os.environ["PROBE_DOCKER_LOG"], "w") as f:
        json.dump(args, f)
    i = args.index("-lc")
    sys.exit(subprocess.run(["bash", "-c", args[i + 1], *args[i + 2:]]).returncode)
""",
        "uv": """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["PROBE_UV_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if "train.py" in sys.argv:
    sys.exit(int(os.environ.get("PROBE_TRAIN_STATUS", "0")))
""",
    }
    for name, content in commands.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PROBE_DOCKER_LOG": str(tmp_path / "docker.json"),
        "PROBE_UV_LOG": str(tmp_path / "uv.jsonl"),
    }


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def _uv_calls(env: dict[str, str]) -> list[list[str]]:
    return [json.loads(line) for line in Path(env["PROBE_UV_LOG"]).read_text().splitlines()]


@pytest.mark.parametrize(("batch", "accumulation"), [("40", "2"), ("80", "1")])
def test_default_effective_batch_and_nested_shell_variables(
    shell_env: dict[str, str], batch: str, accumulation: str
) -> None:
    result = _run([batch], shell_env)
    assert result.returncode == 0, result.stderr
    calls = _uv_calls(shell_env)
    train = calls[-1]
    assert train[train.index("--batch-size") + 1] == batch
    assert train[train.index("--gradient-accumulation-steps") + 1] == accumulation
    assert train[train.index("--max-steps") + 1] == "30"
    assert train[train.index("--valid-ratio") + 1] == "0"
    assert train[train.index("--seed") + 1] == "42"
    assert calls[1] == ["run", "--no-sync", "python", "-", "false", "true", "false", "false"]
    assert "effective_batch=80" in result.stdout
    assert "first 10 optimizer steps" in result.stdout
    assert "not a shape-controlled GPU benchmark" in result.stdout


def test_explicit_options_propagate_and_preserve_positional_gpu_host(
    shell_env: dict[str, str],
) -> None:
    result = _run(
        [
            "40",
            "2",
            "fake-host",
            "--accumulation",
            "3",
            "--persistent-workers",
            "true",
            "--workers",
            "0",
            "--checkpoint",
            "false",
            "--compile",
            "true",
            "--steps",
            "60",
            "--dynamic-padding",
            "true",
        ],
        shell_env,
    )
    assert result.returncode == 0, result.stderr
    docker = json.loads(Path(shell_env["PROBE_DOCKER_LOG"]).read_text())
    assert docker[docker.index("--gpus") + 1] == "device=2"
    assert "/read-only-source:/app:ro" in docker
    calls = _uv_calls(shell_env)
    assert calls[1][-4:] == ["true", "false", "true", "true"]
    train = calls[-1]
    for option, value in (
        ("--gradient-accumulation-steps", "3"),
        ("--num-workers", "0"),
        ("--max-steps", "60"),
    ):
        assert train[train.index(option) + 1] == value
    assert "Capacity trial" in result.stdout


def test_busy_gpu_stops_before_container_run(shell_env: dict[str, str]) -> None:
    executable = Path(shell_env["PROBE_DOCKER_LOG"]).parent / "nvidia-smi"
    executable.write_text("#!/bin/sh\nprintf '0\\n4097\\n0\\n'\n", encoding="utf-8")
    result = _run(["80", "1"], shell_env)
    assert result.returncode == 1
    assert "GPU 1 is occupied or unavailable (used: 4097 MiB)" in result.stderr
    assert not Path(shell_env["PROBE_DOCKER_LOG"]).exists()
    assert not Path(shell_env["PROBE_UV_LOG"]).exists()


def test_training_failure_exit_is_preserved(shell_env: dict[str, str]) -> None:
    result = _run(["80", "--steps", "4"], {**shell_env, "PROBE_TRAIN_STATUS": "17"})
    assert result.returncode == 17
    assert "exit=17" in result.stdout
    assert "Short capacity probe" in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["0"],
        ["80", "-1"],
        ["80", "--workers", "-1"],
        ["80", "--compile", "yes"],
        ["80", "--steps"],
        ["80", "--accumulation", "0"],
        ["80", "--unknown"],
        ["80", "--dynamic-padding", "yes"],
    ],
)
def test_invalid_arguments_fail_before_ssh(shell_env: dict[str, str], args: list[str]) -> None:
    result = _run(args, shell_env)
    assert result.returncode == 2
    assert not Path(shell_env["PROBE_DOCKER_LOG"]).exists()


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_dynamic_padding_guard_in_generated_config(
    monkeypatch: pytest.MonkeyPatch, supported: bool, enabled: bool
) -> None:
    source = _SCRIPT.read_text().split("<<PY\n", 1)[1].split("\nPY", 1)[0]
    config_module = ModuleType("irodori_tts.config")
    config_module.TrainConfig = make_dataclass(  # type: ignore[attr-defined]
        "TrainConfig", [("dynamic_condition_padding", bool)] if supported else []
    )
    yaml_module = ModuleType("yaml")
    config = {
        "sample_generation": {"enabled": True},
        "train": {"dynamic_condition_padding": True},
    }
    yaml_module.safe_load = Mock(return_value=config)  # type: ignore[attr-defined]
    dump = Mock()
    yaml_module.safe_dump = dump  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "irodori_tts.config", config_module)
    monkeypatch.setitem(sys.modules, "yaml", yaml_module)
    monkeypatch.setattr(sys, "argv", ["-", "false", "true", "false", str(enabled).lower()])
    monkeypatch.setattr("builtins.open", mock_open())
    if enabled and not supported:
        with pytest.raises(SystemExit, match="sync the opt-in code"):
            exec(compile(source, "probe-config", "exec"), {})
        dump.assert_not_called()
    else:
        exec(compile(source, "probe-config", "exec"), {})
        dump.assert_called_once()
        assert config["sample_generation"]["enabled"] is False
        if supported:
            assert config["train"]["dynamic_condition_padding"] is enabled
        else:
            assert "dynamic_condition_padding" not in config["train"]


def test_speaker_option_reads_that_speakers_manifest_from_the_legacy_mount(
    shell_env: dict[str, str],
) -> None:
    result = _run(["16", "--speaker", "gi_aether", "--accumulation", "5"], shell_env)
    assert result.returncode == 0, result.stderr
    docker = json.loads(Path(shell_env["PROBE_DOCKER_LOG"]).read_text())
    assert any(item.endswith(":/legacy:ro") for item in docker)
    train = _uv_calls(shell_env)[-1]
    assert train[train.index("--manifest") + 1] == "/legacy/gi_aether/manifest.jsonl"
    assert train[train.index("--batch-size") + 1] == "16"
    assert "speaker=gi_aether" in result.stdout


def test_default_speaker_is_cherry_without_a_legacy_mount(shell_env: dict[str, str]) -> None:
    result = _run(["80"], shell_env)
    assert result.returncode == 0, result.stderr
    docker = json.loads(Path(shell_env["PROBE_DOCKER_LOG"]).read_text())
    assert not any(item.endswith(":/legacy:ro") for item in docker)
    train = _uv_calls(shell_env)[-1]
    assert train[train.index("--manifest") + 1] == "data/cherry/manifest.jsonl"


@pytest.mark.parametrize("speaker", ["Bad Name", "gi;rm", "../x", ""])
def test_unsafe_speaker_names_fail_before_ssh(shell_env: dict[str, str], speaker: str) -> None:
    result = _run(["80", "--speaker", speaker], shell_env)
    assert result.returncode == 2
    assert not Path(shell_env["PROBE_DOCKER_LOG"]).exists()
