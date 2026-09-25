"""Tests for irodori_tts.atmos_client.

These exercise the public surface of AtmosClient without requiring a real
atmos network connection. The real atmos module is replaced with a stub
inserted into sys.modules before AtmosClient is constructed, so we can
inspect the kwargs that flow into atmos.init / run.log / run.log_audio.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from irodori_tts.atmos_client import AtmosClient, AtmosConfig, from_env


class _StubRun:
    def __init__(self, init_kwargs: dict[str, Any]) -> None:
        self.init_kwargs = init_kwargs
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.logged_audio: list[tuple[str, Any, int]] = []
        self.finished_with: str | None = "<not-finished>"

    def log(self, data: dict[str, Any], step: int) -> None:
        self.logged.append((dict(data), step))

    def log_audio(self, label: str, path: Any, step: int) -> None:
        self.logged_audio.append((label, path, step))

    def finish(self, status: str = "finished") -> None:
        self.finished_with = status


def _install_stub_atmos(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, list[_StubRun]]:
    """Insert a fake `atmos` module into sys.modules and return it.

    The returned `runs` list captures every _StubRun that the stub init()
    creates — there should typically be exactly one per AtmosClient.
    """
    runs: list[_StubRun] = []

    stub = types.ModuleType("atmos")

    def _init(project: str, **kwargs: Any) -> _StubRun:
        run = _StubRun({"project": project, **kwargs})
        runs.append(run)
        return run

    stub.init = _init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "atmos", stub)
    return stub, runs


def test_disabled_client_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if atmos is importable, an enabled=False config must skip init.
    _install_stub_atmos(monkeypatch)
    client = AtmosClient(AtmosConfig(enabled=False))
    assert client.enabled is False
    assert client.run is None
    # All public methods must be no-ops without raising.
    client.log({"x": 1.0}, step=10)
    client.log_audio("samples/x", "/tmp/x.wav", step=10)
    client.finish()


def test_enabled_init_threads_project_and_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runs = _install_stub_atmos(monkeypatch)
    cfg = AtmosConfig(
        enabled=True,
        project="irodori-tts",
        run_name="ayaka_lora",
        visibility="private",
        api_url="https://atmos-staging.qleap.jp",
        token="dummy-token",
        job_id="run-uuid",
    )
    client = AtmosClient(cfg, config={"k": 1})

    assert client.enabled is True
    assert len(runs) == 1
    init_kwargs = runs[0].init_kwargs
    assert init_kwargs["project"] == "irodori-tts"
    assert init_kwargs["name"] == "ayaka_lora"
    assert init_kwargs["visibility"] == "private"
    assert init_kwargs["api_url"] == "https://atmos-staging.qleap.jp"
    assert init_kwargs["token"] == "dummy-token"
    assert init_kwargs["job_id"] == "run-uuid"
    assert init_kwargs["config"] == {"k": 1}


def test_name_falls_back_to_configured_run_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """atmos's Run has no `.name` — AtmosClient.name reports the configured
    run_name instead of anything read back from the stub run object."""
    _install_stub_atmos(monkeypatch)
    client = AtmosClient(AtmosConfig(enabled=True, project="irodori-tts", run_name="ayaka_lora"))
    assert client.name == "ayaka_lora"


def test_log_and_log_audio_and_finish_route_through_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _, runs = _install_stub_atmos(monkeypatch)
    client = AtmosClient(AtmosConfig(enabled=True, project="irodori-tts"))

    client.log({"train/loss": 0.5}, step=42)
    client.log_audio("samples/prompt", "/tmp/prompt.wav", step=42)
    client.finish(status="finished")

    run = runs[0]
    assert run.logged == [({"train/loss": 0.5}, 42)]
    assert run.logged_audio == [("samples/prompt", "/tmp/prompt.wav", 42)]
    assert run.finished_with == "finished"
    # finish() flips the wrapper into the disabled state so subsequent calls
    # are inert (matches the "disabled client is inert" contract).
    assert client.run is None
    client.log({"after": 1.0}, step=43)  # must not raise


def test_missing_atmos_module_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "atmos", None)
    with pytest.raises(RuntimeError, match="atmos"):
        AtmosClient(AtmosConfig(enabled=True, project="irodori-tts"))


def test_from_env_reads_documented_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATMOS_API_URL", "https://atmos-staging.qleap.jp")
    monkeypatch.setenv("ATMOS_TOKEN", "k")
    cfg = from_env(
        enabled=True,
        project="irodori-tts",
        run_name="run",
        job_id="run-uuid",
    )
    assert cfg.api_url == "https://atmos-staging.qleap.jp"
    assert cfg.token == "k"
    assert cfg.project == "irodori-tts"
    assert cfg.job_id == "run-uuid"


def test_from_env_missing_creds_yield_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATMOS_API_URL", raising=False)
    monkeypatch.delenv("ATMOS_TOKEN", raising=False)
    cfg = from_env(enabled=False, project="irodori-tts", run_name=None)
    assert cfg.api_url is None
    assert cfg.token is None
