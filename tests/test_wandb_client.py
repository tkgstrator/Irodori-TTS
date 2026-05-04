"""Tests for irodori_tts.wandb_client.

These exercise the public surface of WandbClient without requiring a real
wandb network connection. The real wandb module is replaced with a stub
inserted into sys.modules before WandbClient is constructed, so we can
inspect the kwargs that flow into wandb.init / Settings / run.log.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest

from irodori_tts.wandb_client import WandbClient, WandbConfig, from_env


class _StubSettings:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _StubAudio:
    def __init__(self, audio: Any, **kwargs: Any) -> None:
        self.audio = audio
        self.kwargs = kwargs


class _StubRun:
    def __init__(self, init_kwargs: dict[str, Any]) -> None:
        self.init_kwargs = init_kwargs
        self.name = init_kwargs.get("name") or "stub-run"
        self.logged: list[tuple[dict[str, Any], int | None]] = []
        self.summary: dict[str, Any] = {}
        self.finished_with: int | None | str = "<not-finished>"

    def log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        self.logged.append((dict(data), step))

    def finish(self, exit_code: int | None = None) -> None:
        self.finished_with = exit_code


def _install_stub_wandb(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, list[_StubRun]]:
    """Insert a fake `wandb` module into sys.modules and return it.

    The returned `runs` list captures every _StubRun that the stub init()
    creates — there should typically be exactly one per WandbClient.
    """
    runs: list[_StubRun] = []

    stub = types.ModuleType("wandb")

    def _init(**kwargs: Any) -> _StubRun:
        run = _StubRun(kwargs)
        runs.append(run)
        return run

    stub.init = _init  # type: ignore[attr-defined]
    stub.Settings = _StubSettings  # type: ignore[attr-defined]
    stub.Audio = _StubAudio  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", stub)
    return stub, runs


def test_disabled_client_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if wandb is importable, an enabled=False config must skip init.
    _install_stub_wandb(monkeypatch)
    client = WandbClient(WandbConfig(enabled=False))
    assert client.enabled is False
    assert client.run is None
    assert client.wandb is None
    # All public methods must be no-ops without raising.
    client.log({"x": 1}, step=10)
    client.set_summary("k", "v")
    assert client.Audio(b"\0\0", sample_rate=16000) is None
    client.finish(exit_code=0)


def test_enabled_init_threads_cf_access_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runs = _install_stub_wandb(monkeypatch)
    cfg = WandbConfig(
        enabled=True,
        project="irodori-tts",
        entity="tkgstrator",
        run_name="ayaka_lora",
        mode="online",
        base_url="https://wandb.tkgstrator.work",
        api_key="dummy-key",
        cf_access_client_id="cf-id",
        cf_access_client_secret="cf-secret",
    )
    client = WandbClient(cfg, config={"k": 1}, output_dir="/tmp/run")

    assert os.environ["WANDB_BASE_URL"] == "https://wandb.tkgstrator.work"
    assert os.environ["WANDB_API_KEY"] == "dummy-key"
    assert client.enabled is True
    assert len(runs) == 1
    init_kwargs = runs[0].init_kwargs
    assert init_kwargs["project"] == "irodori-tts"
    assert init_kwargs["entity"] == "tkgstrator"
    assert init_kwargs["name"] == "ayaka_lora"
    assert init_kwargs["mode"] == "online"
    assert init_kwargs["dir"] == "/tmp/run"
    assert init_kwargs["config"] == {"k": 1}
    settings = init_kwargs["settings"]
    assert isinstance(settings, _StubSettings)
    assert settings.kwargs["x_extra_http_headers"] == {
        "CF-Access-Client-Id": "cf-id",
        "CF-Access-Client-Secret": "cf-secret",
    }


def test_empty_entity_passes_none_to_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    """pyaml-env expands an unset ${WANDB_ENTITY} to "" — we must forward
    that as None so wandb falls back to the API key's default entity
    instead of trying to upsert into a literally empty namespace."""
    _, runs = _install_stub_wandb(monkeypatch)
    client = WandbClient(
        WandbConfig(enabled=True, entity="", project="irodori-tts"),
    )
    assert client.enabled is True
    assert runs[0].init_kwargs["entity"] is None


def test_no_cf_creds_omits_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runs = _install_stub_wandb(monkeypatch)
    WandbClient(WandbConfig(enabled=True, project="irodori-tts"))
    assert runs[0].init_kwargs["settings"] is None


def test_partial_cf_creds_omits_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """If only one of the two CF Access credentials is present, sending
    a half-configured header pair would just produce 403s — leave the
    Settings unset so the request goes out without CF Access at all."""
    _, runs = _install_stub_wandb(monkeypatch)
    WandbClient(
        WandbConfig(
            enabled=True,
            project="irodori-tts",
            cf_access_client_id="cf-id",
            cf_access_client_secret=None,
        ),
    )
    assert runs[0].init_kwargs["settings"] is None


def test_log_audio_summary_finish_route_through_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runs = _install_stub_wandb(monkeypatch)
    client = WandbClient(WandbConfig(enabled=True, project="irodori-tts"))

    audio = client.Audio(b"\x00\x01", sample_rate=24000, caption="step=42")
    assert isinstance(audio, _StubAudio)
    assert audio.kwargs == {"sample_rate": 24000, "caption": "step=42"}

    client.log({"train/loss": 0.5}, step=42)
    client.set_summary("train/final_step", 1234)
    client.finish(exit_code=0)

    run = runs[0]
    assert run.logged == [({"train/loss": 0.5}, 42)]
    assert run.summary == {"train/final_step": 1234}
    assert run.finished_with == 0
    # finish() flips the wrapper into the disabled state so subsequent calls
    # are inert (matches the "disabled client is inert" contract).
    assert client.run is None
    client.log({"after": 1})  # must not raise


def test_missing_wandb_module_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(RuntimeError, match="wandb"):
        WandbClient(WandbConfig(enabled=True, project="irodori-tts"))


def test_from_env_reads_documented_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_BASE_URL", "https://w.example")
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "id")
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", "secret")
    cfg = from_env(
        enabled=True,
        project="irodori-tts",
        entity=None,
        run_name="run",
        mode="online",
    )
    assert cfg.base_url == "https://w.example"
    assert cfg.api_key == "k"
    assert cfg.cf_access_client_id == "id"
    assert cfg.cf_access_client_secret == "secret"
    assert cfg.project == "irodori-tts"


def test_from_env_missing_cf_creds_yield_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CF_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("CF_ACCESS_CLIENT_SECRET", raising=False)
    cfg = from_env(
        enabled=False,
        project="irodori-tts",
        entity=None,
        run_name=None,
        mode="online",
    )
    assert cfg.cf_access_client_id is None
    assert cfg.cf_access_client_secret is None
