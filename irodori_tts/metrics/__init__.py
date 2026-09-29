"""Backend-agnostic metrics logging for train.py.

train.py only ever talks to this module: it builds a `MetricsLogger` through
`create_metrics_logger()` and calls `.log()` / `.log_audio()` / `.finish()`
on it. The default backend ("none") is `NullMetricsLogger`, so training runs
with zero metrics infrastructure configured unless a backend is explicitly
selected.

Concrete backends live in sibling modules (currently only `atmos`) and are
imported lazily by `create_metrics_logger`, keyed by name in `_BACKENDS`.
Importing this package never imports a backend module, so `metrics_backend
= "none"` (or any train.py import) never pulls in an optional backend's
dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

# backend name -> (module, class), resolved lazily inside create_metrics_logger.
_BACKENDS: dict[str, tuple[str, str]] = {
    "atmos": ("irodori_tts.metrics.atmos", "AtmosMetricsLogger"),
}


@runtime_checkable
class MetricsLogger(Protocol):
    """What train.py expects from any metrics backend."""

    name: str | None
    enabled: bool

    def log(self, data: dict[str, Any], *, step: int) -> None: ...

    def log_audio(self, label: str, path: Any, *, step: int) -> None: ...

    def finish(self, status: str = "finished") -> None: ...


class NullMetricsLogger:
    """The default backend: every call is a no-op.

    `.name` still reports the configured run name so callers can print/log
    run identity without branching on whether metrics logging is active.
    """

    def __init__(self, run_name: str | None = None) -> None:
        self.name = run_name
        self.enabled = False

    def log(self, data: dict[str, Any], *, step: int) -> None:  # noqa: ARG002
        return None

    def log_audio(self, label: str, path: Any, *, step: int) -> None:  # noqa: ARG002
        return None

    def finish(self, status: str = "finished") -> None:  # noqa: ARG002
        return None


def create_metrics_logger(  # noqa: PLR0913
    backend: str,
    *,
    project: str,
    run_name: str | None,
    run_id: str | None = None,
    config: dict[str, Any] | None = None,
    enabled: bool = True,
) -> MetricsLogger:
    """Build the `MetricsLogger` for `backend`.

    Returns a `NullMetricsLogger` (keeping `run_name` as `.name`) when
    `enabled` is False or `backend == "none"`. Any other unknown backend
    name is an error.
    """
    if not enabled or backend == "none":
        return NullMetricsLogger(run_name)

    target = _BACKENDS.get(backend)
    if target is None:
        raise ValueError(
            f"Unknown metrics_backend {backend!r}. Available: {sorted({'none', *_BACKENDS})}."
        )
    module_name, class_name = target
    backend_cls = getattr(importlib.import_module(module_name), class_name)
    return backend_cls(
        project=project,
        run_name=run_name,
        run_id=run_id,
        config=config,
        enabled=enabled,
    )
