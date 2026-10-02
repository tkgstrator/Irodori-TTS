"""Run the existing shell entrypoints with fake GPUs and a fake trainer."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def training_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    scripts = tmp_path / "scripts/train"
    scripts.mkdir(parents=True)
    for name in ("stream_pipeline.sh", "train_multi_speaker.sh"):
        shutil.copyfile(ROOT / "scripts/train" / name, scripts / name)
        (scripts / name).chmod(0o755)
    config = tmp_path / "configs/train_v4_large_lora.yaml"
    config.parent.mkdir()
    config.write_text("train: {}\n")
    base = tmp_path / "models/large/model.safetensors"
    base.parent.mkdir(parents=True)
    base.write_text("base")
    bins = tmp_path / "bin"
    bins.mkdir()
    commands = {
        "nvidia-smi": """#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = ' '.join(sys.argv[1:])
if 'query-compute-apps' in args:
    if os.environ.get('FAKE_QUERY_FAIL') == '1': sys.exit(1)
    if os.environ.get('FAKE_CONTEXT') == '1': print('GPU-0')
elif 'memory.used' in args:
    print('0, GPU-0, ' + ('9000' if os.environ.get('FAKE_BUSY_ONCE') == '1' and not Path(os.environ['TICKS']).exists() else '4'))
elif 'uuid' in args:
    print('0, GPU-0')
else:
    print('0')
""",
        "sleep": '#!/bin/sh\ntouch "$TICKS"\n/bin/sleep 0.01\n',
        "pgrep": "#!/bin/sh\nexit 1\n",
        "uv": """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(args) + '\\n')
count = len(Path(os.environ['CALLS']).read_text().splitlines())
if os.environ.get('FAKE_FAILURE') == '1': sys.exit(9)
if os.environ.get('FAKE_OOM_ONCE') == '1' and count == 1:
    print('torch.OutOfMemoryError: CUDA out of memory')
    sys.exit(1)
if os.environ.get('FAKE_OOM_ALWAYS') == '1':
    print('torch.OutOfMemoryError: CUDA out of memory')
    sys.exit(1)
if os.environ.get('FAKE_NO_FINAL') == '1': sys.exit(0)
out = Path(args[args.index('--output-dir') + 1]) / 'checkpoint_final'
out.mkdir(parents=True, exist_ok=True)
for name in ('adapter_config.json', 'adapter_model.safetensors', 'config.json', 'trainer_state.pt'):
    (out / name).write_text('saved')
manifest = Path(args[args.index('--manifest') + 1])
(out / 'manifest_size.txt').write_text(str(len(manifest.read_text().splitlines())))
""",
    }
    for name, program in commands.items():
        path = bins / name
        path.write_text(program)
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bins}:{os.environ['PATH']}",
        "CONFIG": str(config),
        "BASE_CKPT": str(base),
        "DATA_ROOT": str(tmp_path / "corpus"),
        "OUTPUT_ROOT": str(tmp_path / "results"),
        "LOCK_DIR": str(tmp_path / "locks"),
        "CALLS": str(tmp_path / "calls.jsonl"),
        "TICKS": str(tmp_path / "ticks"),
        "GPUS": "auto",
        "MIN_FREE_GB": "0",
        "GPU_POLL_SECONDS": "0",
        "ATMOS_TOKEN": "test-only",
        "METRICS_BACKEND": "atmos",
    }
    return tmp_path, env


def _speaker(root: Path, name: str, *, metadata: bool = False, frames: int | None = None) -> None:
    directory = root / "corpus" / name
    (directory / "latents").mkdir(parents=True)
    (directory / "latents/one.pt").write_text("latent")
    row = {"latent_path": "latents/one.pt", "text": "a"}
    if frames is not None:
        row["num_frames"] = frames
    (directory / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    if metadata:
        (directory / "metadata.jsonl").write_text('{"text":"a"}\n')


def _run(
    root: Path, env: dict[str, str], script: str, *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / "scripts/train" / script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _calls(root: Path) -> list[list[str]]:
    path = root / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_pipeline_waits_for_busy_gpu_and_trains_each_prepared_speaker_once(training_repo) -> None:
    root, env = training_repo
    _speaker(root, "gi_one", metadata=True)
    _speaker(root, "ema")
    result = _run(root, {**env, "FAKE_BUSY_ONCE": "1"}, "stream_pipeline.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "ticks").exists()
    calls = _calls(root)
    assert len(calls) == 2
    assert all("--max-steps" not in call for call in calls)
    assert all("--warmup-ratio" not in call for call in calls)
    assert all(call[call.index("--metrics-backend") + 1] == "atmos" for call in calls)
    assert (root / "locks/gi_one.done").exists()
    assert (root / "locks/ema.done").exists()
    assert not (root / "locks/GPU-0.gpu").exists()


def test_failure_is_not_marked_done(training_repo) -> None:
    root, env = training_repo
    _speaker(root, "gi_one")
    result = _run(root, {**env, "FAKE_FAILURE": "1"}, "stream_pipeline.sh")
    assert result.returncode == 1
    assert (root / "locks/gi_one.failed").exists()
    assert not (root / "locks/gi_one.done").exists()
    assert not (root / "locks/GPU-0.gpu").exists()


def test_zero_exit_without_final_checkpoint_is_failure(training_repo) -> None:
    root, env = training_repo
    _speaker(root, "gi_one")
    result = _run(root, {**env, "FAKE_NO_FINAL": "1"}, "stream_pipeline.sh")
    assert result.returncode == 1
    assert (root / "locks/gi_one.failed").exists()


def _batch(call: list[str]) -> tuple[str, str]:
    return (
        call[call.index("--batch-size") + 1],
        call[call.index("--gradient-accumulation-steps") + 1],
    )


@pytest.mark.parametrize(
    ("frames", "batch", "accumulation"),
    [(300, "40", "2"), (355, "40", "2"), (500, "20", "4"), (742, "16", "5"), (None, "16", "5")],
)
def test_batch_follows_the_longest_clip_and_keeps_effective_batch_80(
    training_repo, frames, batch, accumulation
) -> None:
    root, env = training_repo
    _speaker(root, "gi_one", frames=frames)
    result = _run(root, {**env, "GPUS": "0"}, "train_multi_speaker.sh", "gi_one")
    assert result.returncode == 0, result.stdout + result.stderr
    (call,) = _calls(root)
    assert _batch(call) == (batch, accumulation)
    assert int(batch) * int(accumulation) == 80
    assert "--gradient-checkpointing" not in call


def test_initial_oom_steps_down_the_batch_and_keeps_effective_batch(training_repo) -> None:
    root, env = training_repo
    _speaker(root, "gi_one", frames=742)
    result = _run(
        root, {**env, "GPUS": "0", "FAKE_OOM_ONCE": "1"}, "train_multi_speaker.sh", "gi_one"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    first, retry = _calls(root)
    assert _batch(first) == ("16", "5")
    assert _batch(retry) == ("10", "8")
    assert "--gradient-checkpointing" not in retry
    assert "gi_one" not in first


def test_oom_retries_stop_after_two_steps(training_repo) -> None:
    root, env = training_repo
    _speaker(root, "gi_one", frames=742)
    result = _run(
        root, {**env, "GPUS": "0", "FAKE_OOM_ALWAYS": "1"}, "train_multi_speaker.sh", "gi_one"
    )
    assert result.returncode != 0
    assert [_batch(call) for call in _calls(root)] == [("16", "5"), ("10", "8"), ("8", "10")]


def test_resume_keeps_the_checkpoints_own_settings_and_never_retries(training_repo) -> None:
    root, env = training_repo
    _speaker(root, "gi_one", frames=742)
    checkpoint = root / "results/gi_one_lora/checkpoint_0000100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest_size.txt").write_text("1")
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "train": {
                    "num_workers": 8,
                    "batch_size": 10,
                    "gradient_accumulation_steps": 8,
                    "gradient_checkpointing": False,
                }
            }
        )
    )
    result = _run(
        root, {**env, "GPUS": "0", "FAKE_OOM_ALWAYS": "1"}, "train_multi_speaker.sh", "gi_one"
    )
    assert result.returncode != 0
    (call,) = _calls(root)
    assert _batch(call) == ("10", "8")
    assert call[call.index("--num-workers") + 1] == "8"
    assert "--no-gradient-checkpointing" in call
    assert "--resume" in call


def test_finished_speaker_is_not_retrained(training_repo) -> None:
    root, env = training_repo
    _speaker(root, "cherry")
    result = _run(root, {**env, "GPUS": "0"}, "train_multi_speaker.sh", "cherry")
    assert result.returncode == 0
    result = _run(root, {**env, "GPUS": "0"}, "train_multi_speaker.sh", "cherry")
    assert result.returncode == 0
    assert len(_calls(root)) == 1
    assert "already finished" in result.stdout


@pytest.mark.parametrize(
    ("speaker", "canonical"),
    [("ema", "mgwt_ema"), ("cherry", "vtuber_cherry"), ("gi_ema", "gi_ema")],
)
def test_large_run_uses_canonical_output_and_metrics_name(
    training_repo, speaker, canonical
) -> None:
    root, env = training_repo
    _speaker(root, speaker)
    result = _run(root, {**env, "GPUS": "0"}, "train_multi_speaker.sh", speaker)
    assert result.returncode == 0, result.stdout + result.stderr
    call = _calls(root)[0]
    assert call[call.index("--output-dir") + 1].endswith(f"{canonical}_lora")
    assert call[call.index("--metrics-run-name") + 1] == f"{canonical}_lora_v4_large"


def test_completed_legacy_cherry_is_reused_without_renaming(training_repo) -> None:
    root, env = training_repo
    _speaker(root, "cherry")
    legacy = root / "results/cherry_lora/checkpoint_final"
    legacy.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "trainer_state.pt",
    ):
        (legacy / name).write_text("saved")
    (legacy / "manifest_size.txt").write_text("1")
    result = _run(root, {**env, "GPUS": "0"}, "train_multi_speaker.sh", "cherry")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _calls(root) == []
    assert legacy.exists()
    assert not (root / "results/vtuber_cherry_lora").exists()


@pytest.mark.parametrize("failure", ["FAKE_QUERY_FAIL", "FAKE_CONTEXT"])
def test_gpu_query_failure_and_context_block_launch(training_repo, failure: str) -> None:
    root, env = training_repo
    source = (root / "scripts/train/stream_pipeline.sh").read_text()
    function = source.split("gpu_is_free() {", 1)[1].split("\n}\n", 1)[0]
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -uo pipefail; IDLE_GPU_MEM_MIB=100; gpu_is_free() {"
            + function
            + "\n}; gpu_is_free 0",
        ],
        env={**env, failure: "1"},
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert _calls(root) == []


@pytest.mark.parametrize("node", ["g20", "g17"])
def test_cluster_mode_reuses_dockerfile_and_never_exposes_credentials(training_repo, node) -> None:
    root, env = training_repo
    fake = root / "bin/ssh"
    fake.write_text(
        """#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
body = sys.stdin.read()
if 'docker inspect' in ' '.join(args):
    print('ATMOS_TOKEN=secret-atmos\\nHF_TOKEN=secret-hf\\nPATH=/ignored')
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps({'args': args, 'body': body}) + '\\n')
"""
    )
    fake.chmod(0o755)
    result = _run(root, env, "stream_pipeline.sh", "--cluster", node)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    deployment = next(record for record in records if "DEPLOY_AUDIO=0" in " ".join(record["args"]))
    assert deployment["args"][2] == "g20"
    assert records[-1]["args"][2] == node
    script = records[-1]["body"]
    assert "docker build -f docker/train/Dockerfile" in script
    assert "docker compose" not in script
    assert "docker run -d" in script
    assert 'env_args+=(-e "$name")' in script
    assert "for name in ATMOS_TOKEN" in script
    assert '"${env_args[@]}"' in script
    assert "-e GPUS=auto" in script
    assert ":/app/data:ro" in script
    assert "data/cherry" not in script
    assert script.index("Model cache verified") < script.index(
        "exec scripts/train/stream_pipeline.sh"
    )
    assert "local_files_only=True" in script
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in script
    assert "secret-atmos" in script  # delivered through stdin only
    assert all("secret" not in " ".join(record["args"]) for record in records)
    assert "secret" not in result.stdout + result.stderr
    assert "PATH=/ignored" not in script


@pytest.mark.parametrize("mode", ["--deploy", "--deploy-audio"])
def test_deploy_replaces_files_atomically_and_skips_the_running_check(training_repo, mode) -> None:
    root, env = training_repo
    for name in (
        "irodori_tts/inference_runtime.py",
        "irodori_tts/server/config.py",
        "irodori_tts/server/registry.py",
        "pyproject.toml",
        "uv.lock",
        "requirements.txt",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    fake = root / "bin/ssh"
    fake.write_text(
        """#!/usr/bin/env python3
import io, json, os, sys, tarfile
body = sys.stdin.buffer.read()
with tarfile.open(fileobj=io.BytesIO(body)) as archive:
    members = archive.getnames()
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'members': members}) + '\\n')
"""
    )
    fake.chmod(0o755)
    result = _run(root, env, "stream_pipeline.sh", mode, "g20")
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    assert len(records) == 1  # no "is training already running" probe
    command = " ".join(records[0]["args"])
    members = records[0]["members"]
    assert members[:2] == [
        "scripts/train/stream_pipeline.sh",
        "scripts/train/train_multi_speaker.sh",
    ]
    if mode == "--deploy-audio":
        assert "irodori_tts/inference_runtime.py" in members
        assert "pyproject.toml" in members and "uv.lock" in members
        assert "DEPLOY_AUDIO=1" in command
    else:
        assert len(members) == 2
    assert not any(name.startswith("configs/") for name in members)
    assert ".new" in command and "mv -f" in command
    assert "script-backups" in command
    assert "compose" not in command


def test_deploy_rejects_unsafe_node_names(training_repo) -> None:
    root, env = training_repo
    result = _run(root, env, "stream_pipeline.sh", "--deploy", "g20;rm")
    assert result.returncode == 2
