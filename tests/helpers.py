"""Fixtures shared by the server tests: dummy configs, LoRA files and a fake runtime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from safetensors.torch import save_file

from irodori_tts.server import registry as registry_module

UUID_A = "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb"
UUID_B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

FAKE_SAMPLE_RATE = 48000
FAKE_HOP_LENGTH = 512
FAKE_AUDIO_SAMPLES = 4800


def write_config(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def write_lora(path: Path, metadata: dict[str, str] | None = None) -> Path:
    meta = {"adapter_config": "{}"}
    if metadata:
        meta.update(metadata)
    save_file({"lora_A.weight": torch.zeros(2, 2)}, str(path), metadata=meta)
    return path


class FakeTokenizer:
    def encode(self, text: str) -> torch.Tensor:
        return torch.zeros(len(text))


class FakeRuntime:
    """Stand-in for ``InferenceRuntime``: only what the registry and the speech path touch."""

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.model_cfg = SimpleNamespace(use_speaker_condition=True)
        self.codec = SimpleNamespace(
            sample_rate=FAKE_SAMPLE_RATE,
            model=SimpleNamespace(hop_length=FAKE_HOP_LENGTH),
        )
        self.watermarker = SimpleNamespace(model=object())
        self.tokenizer = FakeTokenizer()
        # Test hook: called with `self` from inside synthesize(), while the
        # registry's lock is (correctly) still held. Lets concurrency tests
        # observe/interfere at the point a real GPU call would block.
        self.on_synthesize: Callable[[FakeRuntime], None] | None = None

    def set_active_adapter(self, name: str) -> None:
        self.active_adapter = name

    def synthesize(self, req: Any, **_kwargs: Any) -> SimpleNamespace:
        self.last_request = req
        if self.on_synthesize is not None:
            self.on_synthesize(self)
        return SimpleNamespace(
            audio=torch.zeros(1, FAKE_AUDIO_SAMPLES),
            sample_rate=FAKE_SAMPLE_RATE,
            used_seed=7,
        )


def install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the runtime loader with a fake and record what it was asked for."""
    calls: dict[str, Any] = {"base": [], "slots": [], "made": {}}

    def from_base_with_adapters(
        *, key: Any, adapters: Any, default_adapter: Any, adapter_slots: int = 0
    ) -> FakeRuntime:
        del adapters, default_adapter
        calls["slots"].append(adapter_slots)
        calls["base"].append(key.checkpoint)
        runtime = FakeRuntime(key.checkpoint)
        calls["made"]["base"] = runtime
        return runtime

    monkeypatch.setattr(
        registry_module.InferenceRuntime, "from_base_with_adapters", from_base_with_adapters
    )
    return calls


def lora_test_config(tmp_path: Path, *, extra_speaker: bool = False, **extra: Any) -> Path:
    """Config with one discoverable LoRA (Alice/UUID_A) and an existing (dummy)
    base checkpoint file. Pass extra_speaker=True to also discover a second
    speaker (Bob/UUID_B), for tests that need more than one voice."""
    ckpt = tmp_path / "base.safetensors"
    ckpt.write_text("x", encoding="utf-8")
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    write_lora(
        lora_dir / "alice.safetensors",
        {
            "name": "Alice",
            "uuid": UUID_A,
            "speaker.cv": "Alice Actor",
            "category.id": "female",
            "category.label": "女性",
            "defaults": '{"num_steps": 30}',
        },
    )
    if extra_speaker:
        write_lora(
            lora_dir / "bob.safetensors",
            {
                "name": "Bob",
                "uuid": UUID_B,
                "speaker.cv": "Bob Actor",
                "category.id": "male",
                "category.label": "男性",
                "defaults": '{"num_steps": 30}',
            },
        )
    data: dict[str, Any] = {"base_checkpoint": str(ckpt), "lora_dir": str(lora_dir)}
    data.update(extra)
    return write_config(tmp_path / "c.yaml", data)
