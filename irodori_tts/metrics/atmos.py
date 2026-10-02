"""atmos backend for the generic metrics-logging interface.

Wraps `atmos.init` so callers pass project/run name/job id explicitly at
construction time instead of relying on ambient environment variables for
everything. atmos has no concept of an org/team namespace, access-proxy
headers, or an online/offline/disabled logging mode — the SDK always talks
to a single `api_url` and always sends metrics, so there is nothing to
thread through beyond the connection credentials and the run identity.

Credentials and the server URL come from the environment (ATMOS_BASE_URL,
ATMOS_TOKEN, ATMOS_VISIBILITY) rather than TrainConfig, since those are
atmos-specific and have no meaning for other backends. Before reading them,
a repo-root `.env` file is loaded (without overriding already-set env vars)
when python-dotenv is installed. The repo-root file deliberately overrides
inherited environment values so it remains the source of truth for local and
container runs. `pip install 'irodori-tts[atmos]'` covers local development
without an extra setup step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class AtmosMetricsLogger:
    """MetricsLogger backed by the atmos SDK."""

    def __init__(
        self,
        *,
        project: str,
        run_name: str | None,
        run_id: str | None,
        config: dict[str, Any] | None,
        enabled: bool,
    ) -> None:
        self._run: Any | None = None
        self._name = run_name
        self._api_url: str | None = None
        if not enabled:
            return

        if load_dotenv is not None:
            load_dotenv(_REPO_ROOT / ".env", override=True)

        try:
            import atmos
        except ImportError as exc:
            raise RuntimeError(
                "metrics_backend='atmos' is enabled, but `atmos` is not installed. "
                "Install it from GitHub, not PyPI (the PyPI `atmos` is unrelated): "
                "`pip install 'irodori-tts[atmos]'`, or directly with "
                "`pip install 'atmos @ git+https://github.com/qtmleap/atmos#subdirectory=packages/python-sdk'`."
            ) from exc

        self._api_url = os.environ.get("ATMOS_BASE_URL")
        self._run = atmos.init(
            project,
            name=run_name or None,
            config=config or {},
            visibility=os.environ.get("ATMOS_VISIBILITY") or "private",
            api_url=self._api_url or None,
            token=os.environ.get("ATMOS_TOKEN") or None,
            job_id=run_id or None,
        )

    @property
    def enabled(self) -> bool:
        return self._run is not None

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def api_url(self) -> str | None:
        return self._api_url

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
