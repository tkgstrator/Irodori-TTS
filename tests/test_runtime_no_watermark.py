"""Inference never creates or invokes a watermark backend."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import InferenceRuntime


def test_in_memory_runtime_does_not_create_a_watermark_backend() -> None:
    runtime = InferenceRuntime.from_components(
        model=torch.nn.Linear(2, 2),
        model_cfg=ModelConfig(),
        tokenizer=SimpleNamespace(),
        caption_tokenizer=None,
        codec=SimpleNamespace(device=torch.device("cpu")),
        model_device="cpu",
        codec_device="cpu",
    )
    assert not hasattr(runtime, "watermarker")


def test_synthesis_has_no_watermark_stage() -> None:
    source = inspect.getsource(InferenceRuntime.synthesize)
    assert "watermark" not in source.lower()
