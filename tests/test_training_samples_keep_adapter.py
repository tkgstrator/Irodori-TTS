"""Guards that mid-training samples are rendered through the adapter.

SamplingRequest.lora_adapter names an adapter to load for one request, and the
runtime reads "none named" as "serve the base model" — correct for the server,
wrong for sampling a model that is mid-training and already carries the adapter
being trained. Without keep_adapter every training sample came out identical to
the base model no matter how far the run had got, announced only by an info
line in the log.
"""

from __future__ import annotations

import inspect

from irodori_tts import training_samples
from irodori_tts.inference_runtime import InferenceRuntime, SamplingRequest


def test_request_defaults_to_replacing_the_adapter() -> None:
    """The server's default must not change: no adapter named means base model."""
    assert SamplingRequest(text="x").keep_adapter is False


def test_training_samples_ask_to_keep_the_adapter() -> None:
    source = inspect.getsource(training_samples.generate_training_samples)
    assert "keep_adapter=True" in source, (
        "generate_training_samples must pass keep_adapter=True; without it the "
        "runtime disables the adapter and every sample is base-model audio"
    )


def test_keep_adapter_skips_the_disable_path() -> None:
    """keep_adapter with no adapter path must not reach disable_adapter()."""
    calls: list[str] = []

    class _Model:
        def disable_adapter(self):  # pragma: no cover - must not be called
            calls.append("disabled")
            raise AssertionError("disable_adapter() called despite keep_adapter=True")

    runtime = InferenceRuntime.__new__(InferenceRuntime)
    runtime.model = _Model()

    context = InferenceRuntime._prepare_lora_for_request_inner(  # noqa: SLF001
        runtime,
        None,
        keep_adapter=True,
        messages=[],
        log_fn=lambda _msg: None,
    )
    with context:
        pass
    assert calls == []


def test_without_keep_adapter_the_adapter_is_disabled() -> None:
    """The inverse, so the guard above cannot pass by the path going missing."""
    calls: list[str] = []

    class _Model:
        def disable_adapter(self):
            calls.append("disabled")
            return _Null()

    class _Null:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    runtime = InferenceRuntime.__new__(InferenceRuntime)
    runtime.model = _Model()

    context = InferenceRuntime._prepare_lora_for_request_inner(  # noqa: SLF001
        runtime,
        None,
        keep_adapter=False,
        messages=[],
        log_fn=lambda _msg: None,
    )
    with context:
        pass
    assert calls == ["disabled"]
