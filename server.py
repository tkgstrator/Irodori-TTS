"""FastAPI TTS server for Irodori-TTS speaker LoRAs."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import Response

from irodori_tts.server.config import load_config
from irodori_tts.server.registry import RuntimeRegistry
from irodori_tts.server.schemas import SynthRequest
from irodori_tts.server.synthesis import _handle_synth, _handle_synth_vds


def build_app(cfg_path: Path, *, eager_load: bool = True) -> FastAPI:
    cfg = load_config(cfg_path)
    registry = RuntimeRegistry(cfg)

    app = FastAPI(title="Irodori-TTS Server", version="0.1.0")

    if eager_load:
        registry.load()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "speakers": len(cfg.speakers),
            "caption": registry.caption_available,
        }

    @app.get("/speakers")
    def list_speakers() -> dict[str, Any]:
        return {
            "speakers": [
                {
                    "uuid": s.uuid,
                    "name": s.name,
                    "cv": s.cv,
                    "defaults": s.defaults,
                    "category": {
                        "id": s.category_id,
                        "label": s.category_label,
                    },
                }
                for s in registry.list_speakers()
            ]
        }

    @app.post(
        "/synth",
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                    "audio/pcm": {"schema": {"type": "string", "format": "binary"}},
                },
                "description": "Accept: audio/wav for WAV file, audio/pcm (default) "
                "for raw PCM16 mono. Both single-cue and drama mode supported.",
            }
        },
    )
    def synth(req: SynthRequest, request: Request) -> Response:
        return _handle_synth(registry, cfg, req, request)

    @app.post(
        "/synth/vds",
        responses={
            200: {
                "content": {
                    "audio/pcm": {"schema": {"type": "string", "format": "binary"}},
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                },
                "description": "Drama mode from .vds text upload. "
                "Accept: audio/pcm (default, stream) or audio/wav.",
            }
        },
    )
    async def synth_vds(file: UploadFile, request: Request) -> Response:
        content = await file.read()
        return _handle_synth_vds(registry, cfg, content, request)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("TTS_CONFIG", "config.yaml"))
    parser.add_argument("--host", default=os.environ.get("TTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", "8765")))
    parser.add_argument("--no-eager-load", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = build_app(Path(args.config), eager_load=not args.no_eager_load)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
