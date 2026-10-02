"""Tests for the backend-agnostic irodori_tts.metrics logging layer.

`irodori_tts.metrics.create_metrics_logger` is the only entry point train.py
uses. It must never import a concrete backend module until one is actually
requested, so plain `metrics_backend="none"` training runs (the default)
never touch atmos or any other optional dependency.

The atmos backend is exercised with a fake `atmos` module inserted into
`sys.modules`, mirroring the strategy the old irodori_tts.atmos_client tests
used, so this does not require real network credentials. `load_dotenv` is
stubbed out in those tests so they never touch a real repo-root `.env` file
or leak environment variables across the test session.
"""

from __future__ import annotations

import subprocess
import sys
import types
from typing import Any

import pytest

from irodori_tts.metrics import (
    MetricsLogger,
    NullMetricsLogger,
    create_metrics_logger,
)


def test_null_metrics_logger_is_inert() -> None:
    logger = NullMetricsLogger("run-name")
    assert logger.name == "run-name"
    assert logger.enabled is False
    # All public methods must be no-ops without raising.
    logger.log({"x": 1.0}, step=10)
    logger.log_audio("samples/x", "/tmp/x.wav", step=10)
    logger.finish()


def test_null_metrics_logger_defaults_name_to_none() -> None:
    assert NullMetricsLogger().name is None


@pytest.mark.parametrize("backend", ["none", "atmos"])
def test_disabled_returns_null_regardless_of_backend(backend: str) -> None:
    logger = create_metrics_logger(backend, project="p", run_name="run", enabled=False)
    assert isinstance(logger, NullMetricsLogger)
    assert logger.name == "run"
    assert logger.enabled is False


def test_none_backend_returns_null_even_when_enabled() -> None:
    logger = create_metrics_logger("none", project="p", run_name="run", enabled=True)
    assert isinstance(logger, NullMetricsLogger)
    assert logger.name == "run"


def test_unknown_backend_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown metrics_backend 'bogus'"):
        create_metrics_logger("bogus", project="p", run_name="run", enabled=True)


def test_importing_metrics_package_never_imports_atmos() -> None:
    # Run in a subprocess so this is independent of what earlier tests in
    # this session have already imported.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import irodori_tts.metrics; assert 'atmos' not in sys.modules",
        ],
        cwd=None,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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


def _install_stub_atmos(monkeypatch: pytest.MonkeyPatch) -> list[_StubRun]:
    """Insert a fake `atmos` module into sys.modules and return its runs.

    The returned list captures every _StubRun that the stub init() creates —
    there should typically be exactly one per AtmosMetricsLogger.
    """
    runs: list[_StubRun] = []

    stub = types.ModuleType("atmos")

    def _init(project: str, **kwargs: Any) -> _StubRun:
        run = _StubRun({"project": project, **kwargs})
        runs.append(run)
        return run

    stub.init = _init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "atmos", stub)
    return runs


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch a real repo-root `.env` file from these tests.

    Without this, AtmosMetricsLogger's load_dotenv(..., override=False) call
    would read the real file (if any) and could set process env vars that
    outlive the test (monkeypatch only tracks the vars *it* sets).
    """
    from irodori_tts.metrics import atmos as atmos_backend

    monkeypatch.setattr(atmos_backend, "load_dotenv", lambda *_args, **_kwargs: None)


def test_atmos_backend_threads_project_and_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _install_stub_atmos(monkeypatch)
    monkeypatch.setenv("ATMOS_BASE_URL", "https://atmos-staging.qleap.jp")
    monkeypatch.setenv("ATMOS_TOKEN", "dummy-token")
    monkeypatch.setenv("ATMOS_VISIBILITY", "private")

    logger = create_metrics_logger(
        "atmos",
        project="irodori-tts",
        run_name="ayaka_lora",
        run_id="run-uuid",
        config={"k": 1},
        enabled=True,
    )

    assert logger.enabled is True
    assert logger.name == "ayaka_lora"
    assert len(runs) == 1
    init_kwargs = runs[0].init_kwargs
    assert init_kwargs["project"] == "irodori-tts"
    assert init_kwargs["name"] == "ayaka_lora"
    assert init_kwargs["visibility"] == "private"
    assert init_kwargs["api_url"] == "https://atmos-staging.qleap.jp"
    assert init_kwargs["token"] == "dummy-token"
    assert init_kwargs["job_id"] == "run-uuid"
    assert init_kwargs["config"] == {"k": 1}


def test_atmos_backend_visibility_defaults_to_private(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _install_stub_atmos(monkeypatch)
    monkeypatch.delenv("ATMOS_VISIBILITY", raising=False)

    create_metrics_logger("atmos", project="irodori-tts", run_name=None, enabled=True)

    assert runs[0].init_kwargs["visibility"] == "private"


def test_atmos_backend_log_and_log_audio_and_finish_route_through_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _install_stub_atmos(monkeypatch)
    logger = create_metrics_logger("atmos", project="irodori-tts", run_name=None, enabled=True)

    logger.log({"train/loss": 0.5}, step=42)
    logger.log_audio("samples/prompt", "/tmp/prompt.wav", step=42)
    logger.finish(status="finished")

    run = runs[0]
    assert run.logged == [({"train/loss": 0.5}, 42)]
    assert run.logged_audio == [("samples/prompt", "/tmp/prompt.wav", 42)]
    assert run.finished_with == "finished"
    # finish() flips the wrapper into the disabled state so subsequent calls
    # are inert.
    assert logger.enabled is False
    logger.log({"after": 1.0}, step=43)  # must not raise


def test_atmos_backend_missing_module_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "atmos", None)
    with pytest.raises(RuntimeError, match=r"irodori-tts\[atmos\]"):
        create_metrics_logger("atmos", project="irodori-tts", run_name=None, enabled=True)


def test_atmos_backend_satisfies_metrics_logger_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_atmos(monkeypatch)
    logger = create_metrics_logger("atmos", project="irodori-tts", run_name="r", enabled=True)
    assert isinstance(logger, MetricsLogger)


class TestLegacyAtmosTrainKeys:
    def test_atmos_keys_are_migrated_to_metrics_keys(self) -> None:
        import pytest

        from irodori_tts.config import migrate_legacy_train_keys

        legacy = {
            "atmos_enabled": True,
            "atmos_project": "irodori-tts-v4",
            "atmos_run_name": "ayaka",
            "atmos_visibility": "public",
            "lr": 1e-4,
        }
        with pytest.warns(FutureWarning):
            migrated = migrate_legacy_train_keys(legacy, source="old.yaml")
        assert migrated == {
            "lr": 1e-4,
            "metrics_backend": "atmos",
            "metrics_project": "irodori-tts-v4",
            "metrics_run_name": "ayaka",
        }

    def test_disabled_atmos_maps_to_none_backend(self) -> None:
        import pytest

        from irodori_tts.config import migrate_legacy_train_keys

        with pytest.warns(FutureWarning):
            migrated = migrate_legacy_train_keys({"atmos_enabled": False}, source="old.yaml")
        assert migrated == {"metrics_backend": "none"}

    def test_new_keys_pass_through_untouched(self) -> None:
        from irodori_tts.config import migrate_legacy_train_keys

        train = {"metrics_backend": "atmos"}
        assert migrate_legacy_train_keys(train, source="new.yaml") is train

    def test_unknown_atmos_key_is_rejected(self) -> None:
        import pytest

        from irodori_tts.config import migrate_legacy_train_keys

        with pytest.raises(ValueError, match="atmos_typo"):
            migrate_legacy_train_keys({"atmos_typo": 1}, source="old.yaml")
