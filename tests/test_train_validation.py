"""Tests for training-mode handling in ``run_validation``.

The validation pass drops the model into ``eval()``; these pin down that the
prior mode is restored even when the loop blows up mid-pass.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from irodori_tts.config import ModelConfig, TrainConfig
from irodori_tts.training.validation import run_validation


class _StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = ModelConfig()


class _ExplodingLoader:
    def __iter__(self) -> Any:
        raise RuntimeError("loader exploded")


def _run(model: torch.nn.Module, loader: Any) -> dict[str, float]:
    return run_validation(
        model=model,
        loader=loader,
        train_cfg=TrainConfig(),
        device=torch.device("cpu"),
        use_bf16=False,
        distributed=False,
    )


def test_training_mode_is_restored_when_validation_raises() -> None:
    model = _StubModel()
    model.train()
    with pytest.raises(RuntimeError, match="loader exploded"):
        _run(model, _ExplodingLoader())
    assert model.training


def test_eval_mode_is_kept_when_validation_raises() -> None:
    model = _StubModel()
    model.eval()
    with pytest.raises(RuntimeError, match="loader exploded"):
        _run(model, _ExplodingLoader())
    assert not model.training


def test_training_mode_is_restored_on_success() -> None:
    model = _StubModel()
    model.train()
    metrics = _run(model, [])
    assert model.training
    assert metrics["num_samples"] == 0.0


def test_eval_mode_is_kept_on_success() -> None:
    model = _StubModel()
    model.eval()
    _run(model, [])
    assert not model.training
