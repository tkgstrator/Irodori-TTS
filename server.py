"""OpenAI-compatible TTS server for Irodori-TTS speaker LoRAs."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

from irodori_tts.server.config import ServerConfig, load_config
from irodori_tts.server.encoding import MEDIA_TYPES, encode, sse_events
from irodori_tts.server.errors import ApiError, install_error_handlers
from irodori_tts.server.registry import RuntimeRegistry
from irodori_tts.server.schemas import SpeechRequest
from irodori_tts.server.synthesis import synthesize

_BINARY_SCHEMA = {"schema": {"type": "string", "format": "binary"}}


def _model_object(cfg: ServerConfig, created: int) -> dict[str, Any]:
    return {
        "id": cfg.model_id,
        "object": "model",
        "created": created,
        "owned_by": "irodori-tts",
    }


def _require_model(cfg: ServerConfig, model: str) -> None:
    if model != cfg.model_id:
        raise ApiError(
            f"The model `{model}` does not exist. This server serves `{cfg.model_id}`.",
            status_code=404,
            param="model",
            code="model_not_found",
        )


def build_app(cfg_path: Path, *, eager_load: bool = True) -> FastAPI:
    cfg = load_config(cfg_path)
    registry = RuntimeRegistry(cfg)
    created = int(time.time())

    app = FastAPI(title="Irodori-TTS Server", version="1.0.0")
    install_error_handlers(app)

    if eager_load:
        registry.load()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": cfg.model_id, "voices": len(cfg.speakers)}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [_model_object(cfg, created)]}

    @app.get("/v1/models/{model}")
    def retrieve_model(model: str) -> dict[str, Any]:
        _require_model(cfg, model)
        return _model_object(cfg, created)

    @app.get("/v1/audio/voices")
    def list_voices() -> dict[str, Any]:
        """Speaker UUIDs to put in the `voice` field. Not part of the OpenAI API."""
        return {
            "object": "list",
            "data": [
                {
                    "id": s.uuid,
                    "object": "voice",
                    "name": s.name,
                    "cv": s.cv,
                    "category": {"id": s.category_id, "label": s.category_label},
                    "defaults": s.defaults,
                }
                for s in registry.list_speakers()
            ],
        }

    @app.post(
        "/v1/audio/speech",
        responses={
            200: {
                "content": dict.fromkeys(MEDIA_TYPES.values(), _BINARY_SCHEMA)
                | {"text/event-stream": {"schema": {"type": "string"}}},
                "description": "The encoded audio, or speech.audio.* events when "
                "stream_format is sse.",
            }
        },
    )
    def create_speech(req: SpeechRequest) -> Response:
        _require_model(cfg, req.model)
        result = synthesize(registry, cfg, req)
        audio_bytes, media_type = encode(result.audio, result.sample_rate, req.response_format)
        headers = {
            "X-TTS-Voice-Id": result.speaker_id,
            "X-TTS-Used-Seed": str(result.used_seed),
            "X-TTS-Sample-Rate": str(result.sample_rate),
        }
        if req.stream_format == "sse":
            usage = {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            }
            return StreamingResponse(
                sse_events(audio_bytes, usage),
                media_type="text/event-stream",
                headers=headers,
            )
        return Response(content=audio_bytes, media_type=media_type, headers=headers)

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
