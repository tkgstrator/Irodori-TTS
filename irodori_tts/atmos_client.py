"""Thin wrapper around the atmos SDK for training metrics.

Wraps `atmos.init` so callers pass project/run name/job id explicitly at
construction time instead of relying on ambient environment variables for
everything. atmos has no concept of an org/team namespace, access-proxy
headers, or an online/offline/disabled logging mode — the SDK always talks
to a single `api_url` and always sends metrics, so there is nothing to
thread through beyond the connection credentials and the run identity.

`AtmosClient.run` is exposed for code paths that already accept a raw
`atmos.Run` object (e.g. `irodori_tts.training_samples`); new code should
prefer `AtmosClient.log()` / `AtmosClient.log_audio()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AtmosConfig:
    enabled: bool
    project: str | None = None
    run_name: str | None = None
    visibility: str = "private"
    api_url: str | None = None
    token: str | None = None
    job_id: str | None = None


class AtmosClient:
    def __init__(
        self,
        cfg: AtmosConfig,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._run: Any | None = None
        self._cfg = cfg
        if not cfg.enabled:
            return

        try:
            import atmos
        except ImportError as exc:
            raise RuntimeError(
                "atmos logging is enabled, but `atmos` is not installed. "
                "Install it from GitHub, not PyPI (the PyPI `atmos` is unrelated): "
                "`pip install 'atmos @ git+https://github.com/qtmleap/atmos#subdirectory=packages/python-sdk'`."
            ) from exc

        self._run = atmos.init(
            cfg.project or "",
            name=cfg.run_name or None,
            config=config or {},
            visibility=cfg.visibility or "private",
            api_url=cfg.api_url or None,
            token=cfg.token or None,
            job_id=cfg.job_id or None,
        )

    @property
    def enabled(self) -> bool:
        return self._run is not None

    @property
    def run(self) -> Any | None:
        return self._run

    @property
    def name(self) -> str | None:
        return self._cfg.run_name

    @property
    def api_url(self) -> str | None:
        return self._cfg.api_url

    def log(self, data: dict[str, Any], *, step: int) -> None:
        if self._run is None:
            return
        self._run.log(data, step)

    def log_audio(self, label: str, path: Any, *, step: int) -> None:
        if self._run is None:
            return
        self._run.log_audio(label, path, step)

    def finish(self, status: str = "finished") -> None:
        if self._run is None:
            return
        self._run.finish(status=status)
        self._run = None


def from_env(
    *,
    enabled: bool,
    project: str | None,
    run_name: str | None,
    visibility: str = "private",
    job_id: str | None = None,
) -> AtmosConfig:
    """Build an AtmosConfig by reading credentials/URL from the environment.

    Centralizes the env var names so callers do not sprinkle
    `os.environ.get` throughout the codebase. `job_id` (a UUID v4) lets
    callers keep a stable identifier across restarts so atmos appends to the
    same job instead of starting a new one, mirroring a resumable-run id.
    """
    return AtmosConfig(
        enabled=enabled,
        project=project,
        run_name=run_name,
        visibility=visibility,
        api_url=os.environ.get("ATMOS_API_URL"),
        token=os.environ.get("ATMOS_TOKEN"),
        job_id=job_id,
    )
