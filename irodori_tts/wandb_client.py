"""Cloudflare Access-aware Weights & Biases wrapper.

Wraps `wandb.init` so callers pass W&B server URL, API key, and Cloudflare
Access service-token credentials explicitly at construction time instead of
relying on ambient environment variables. The class:

* exports `WANDB_BASE_URL` / `WANDB_API_KEY` to the process environment
  (the wandb SDK reads them at module init), and
* threads `CF-Access-Client-Id` / `CF-Access-Client-Secret` through
  `wandb.Settings(x_extra_http_headers=...)` so HTTP requests to a CF
  Access-fronted W&B instance authenticate correctly.

`WandbClient.run` is exposed for code paths that already accept a raw
`wandb.run` object (e.g. `irodori_tts.training_samples`); new code should
prefer `WandbClient.log()` / `WandbClient.Audio()`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    mode: str = "online"
    base_url: str | None = None
    api_key: str | None = None
    cf_access_client_id: str | None = None
    cf_access_client_secret: str | None = None


class WandbClient:
    def __init__(
        self,
        cfg: WandbConfig,
        *,
        config: dict[str, Any] | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self._run: Any | None = None
        self._wandb: Any | None = None
        self._cfg = cfg
        if not cfg.enabled:
            return

        if cfg.api_key:
            os.environ["WANDB_API_KEY"] = cfg.api_key
        if cfg.base_url:
            os.environ["WANDB_BASE_URL"] = cfg.base_url

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but `wandb` is not installed. "
                "Install it with `pip install wandb`."
            ) from exc
        self._wandb = wandb

        settings = None
        if cfg.cf_access_client_id and cfg.cf_access_client_secret:
            settings = wandb.Settings(
                x_extra_http_headers={
                    "CF-Access-Client-Id": cfg.cf_access_client_id,
                    "CF-Access-Client-Secret": cfg.cf_access_client_secret,
                }
            )

        self._run = wandb.init(
            project=cfg.project or None,
            entity=cfg.entity or None,
            name=cfg.run_name or None,
            mode=cfg.mode or "online",
            dir=str(output_dir) if output_dir is not None else None,
            config=config or {},
            settings=settings,
        )

    @property
    def enabled(self) -> bool:
        return self._run is not None

    @property
    def run(self) -> Any | None:
        return self._run

    @property
    def wandb(self) -> Any | None:
        return self._wandb

    @property
    def name(self) -> str | None:
        if self._run is None:
            return self._cfg.run_name
        return getattr(self._run, "name", None)

    @property
    def base_url(self) -> str | None:
        return self._cfg.base_url

    def log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        if self._run is None:
            return
        self._run.log(data, step=step)

    def Audio(
        self,
        audio: Any,
        *,
        sample_rate: int,
        caption: str | None = None,
    ) -> Any | None:
        if self._wandb is None:
            return None
        return self._wandb.Audio(audio, sample_rate=sample_rate, caption=caption)

    def finish(self, exit_code: int | None = None) -> None:
        if self._run is None:
            return
        self._run.finish(exit_code=exit_code)
        self._run = None


def from_env(
    *,
    enabled: bool,
    project: str | None,
    entity: str | None,
    run_name: str | None,
    mode: str,
) -> WandbConfig:
    """Build a WandbConfig by reading credentials/URL from the environment.

    Centralizes the env var names so callers do not sprinkle `os.environ.get`
    throughout the codebase.
    """
    return WandbConfig(
        enabled=enabled,
        project=project,
        entity=entity,
        run_name=run_name,
        mode=mode,
        base_url=os.environ.get("WANDB_BASE_URL"),
        api_key=os.environ.get("WANDB_API_KEY"),
        cf_access_client_id=os.environ.get("CF_ACCESS_CLIENT_ID"),
        cf_access_client_secret=os.environ.get("CF_ACCESS_CLIENT_SECRET"),
    )
