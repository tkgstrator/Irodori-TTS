"""Tests for ``_load_model_state_from_checkpoint``.

Safetensors checkpoints carry only ``config_json`` and
``text_encoder_config_json``, so there is no train config to recover from them.
The loader must say so rather than hand back an indistinguishable ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from irodori_tts.training.model_init import (
    SAFETENSORS_CONFIG_META_KEY,
    SAFETENSORS_TEXT_ENCODER_CONFIG_META_KEY,
    _load_model_state_from_checkpoint,
)

STATE = {"blocks.0.weight": torch.zeros(2, 2)}


def _write_safetensors(path: Path, metadata: dict[str, str] | None = None) -> Path:
    save_file(STATE, str(path), metadata=metadata or {})
    return path


def test_safetensors_reports_train_config_as_unavailable(tmp_path: Path) -> None:
    path = _write_safetensors(
        tmp_path / "model.safetensors",
        {
            SAFETENSORS_CONFIG_META_KEY: json.dumps({"model_dim": 64}),
            SAFETENSORS_TEXT_ENCODER_CONFIG_META_KEY: json.dumps({"hidden_size": 8}),
        },
    )
    loaded = _load_model_state_from_checkpoint(path)
    assert loaded.train_config is None
    assert loaded.train_config_available is False
    assert loaded.model_config == {"model_dim": 64}
    assert loaded.text_encoder_config == {"hidden_size": 8}


def test_pt_reports_train_config_as_available(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model": STATE,
            "model_config": {"model_dim": 64},
            "train_config": {"lora_r": 8},
            "text_encoder_config": {"hidden_size": 8},
        },
        path,
    )
    loaded = _load_model_state_from_checkpoint(path)
    assert loaded.train_config == {"lora_r": 8}
    assert loaded.train_config_available is True


def test_pt_without_a_train_config_is_still_available(tmp_path: Path) -> None:
    path = tmp_path / "bare.pt"
    torch.save({"model": STATE}, path)
    loaded = _load_model_state_from_checkpoint(path)
    assert loaded.train_config is None
    assert loaded.train_config_available is True


def test_inference_only_config_keys_are_dropped_from_safetensors(tmp_path: Path) -> None:
    path = _write_safetensors(
        tmp_path / "inference.safetensors",
        {SAFETENSORS_CONFIG_META_KEY: json.dumps({"model_dim": 64, "max_text_len": 512})},
    )
    loaded = _load_model_state_from_checkpoint(path)
    assert loaded.model_config == {"model_dim": 64}


def test_model_state_is_returned_for_both_formats(tmp_path: Path) -> None:
    safetensors_path = _write_safetensors(tmp_path / "weights.safetensors")
    pt_path = tmp_path / "weights.pt"
    torch.save({"model": STATE}, pt_path)
    for path in (safetensors_path, pt_path):
        loaded = _load_model_state_from_checkpoint(path)
        assert set(loaded.model_state) == set(STATE)
