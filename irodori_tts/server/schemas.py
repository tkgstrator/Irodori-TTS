"""Request schema for POST /v1/audio/speech and speaker-defaults merging."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from irodori_tts.server.errors import ApiError

MAX_INPUT_CHARS = 4096


class VoiceRef(BaseModel):
    """The object form of ``voice`` that the API also accepts."""

    id: str = Field(..., description="Speaker UUID.")


class SpeechRequest(BaseModel):
    # `model` would otherwise collide with pydantic's own protected prefix.
    model_config = ConfigDict(protected_namespaces=())

    model: str = Field(
        ...,
        description="Model id, as listed by GET /v1/models.",
        examples=["irodori-tts-v4.1-small"],
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_CHARS,
        description="Text to synthesize. Supports {shortcode} emoji annotations.",
        examples=["こんにちは、今日はいい天気ですね。"],
    )
    voice: str | VoiceRef = Field(
        ...,
        description="Speaker UUID, as listed by GET /v1/audio/voices.",
        examples=["7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb"],
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="Audio format. pcm is 24 kHz 16-bit mono, headerless.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Playback rate multiplier applied to the predicted duration.",
    )
    stream_format: Literal["audio", "sse"] = Field(
        default="audio",
        description="audio returns the encoded file; sse streams speech.audio.* events.",
    )
    instructions: str | None = Field(
        default=None,
        description="Not supported by this server: voice design was removed.",
    )

    # --- Extensions. Not part of the OpenAI schema; pass them via extra_body. ---
    seed: int | None = Field(
        default=None,
        description="Sampling seed. Omit or set <0 for random. Fix it to keep one "
        "speaker sounding the same across separate requests.",
    )
    num_steps: int | None = Field(default=None, description="RF sampling steps.")
    cfg_scale_text: float | None = Field(default=None, description="Text CFG scale.")
    cfg_scale_speaker: float | None = Field(default=None, description="Speaker CFG scale.")
    speaker_kv_scale: float | None = Field(
        default=None, description="Speaker KV scale (>1 strengthens identity)."
    )
    truncation_factor: float | None = Field(
        default=None, description="Noise truncation, e.g. 0.8. Omit or set <=0 to disable."
    )
    seconds: float | None = Field(
        default=None, gt=0, description="Fixed duration, overriding the duration predictor."
    )
    min_seconds: float | None = Field(
        default=None, gt=0, description="Lower bound for the predicted duration. Default 0.5."
    )
    max_seconds: float | None = Field(
        default=None, gt=0, description="Upper bound for the predicted duration. Default 30.0."
    )

    @property
    def voice_id(self) -> str:
        return self.voice.id if isinstance(self.voice, VoiceRef) else self.voice

    @model_validator(mode="after")
    def _check_duration_bounds(self) -> SpeechRequest:
        if (
            self.min_seconds is not None
            and self.max_seconds is not None
            and self.min_seconds > self.max_seconds
        ):
            raise ValueError(
                f"min_seconds ({self.min_seconds}) must be <= max_seconds ({self.max_seconds})"
            )
        return self


_POSITIVE_ONLY = {
    "num_steps",
    "cfg_scale_text",
    "cfg_scale_speaker",
    "speaker_kv_scale",
    "truncation_factor",
}


def _merge_defaults(req: SpeechRequest, defaults: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "num_steps": 40,
        "cfg_scale_text": 3.0,
        "cfg_scale_speaker": 5.0,
        "speaker_kv_scale": None,
        "truncation_factor": None,
        "seconds": None,
        "min_seconds": 0.5,
        "max_seconds": 30.0,
        "duration_scale": 1.0,
        "seed": None,
    }
    for k, v in defaults.items():
        if k in resolved:
            resolved[k] = v
    for k in list(resolved.keys()):
        override = getattr(req, k, None)
        if override is None:
            continue
        if k in _POSITIVE_ONLY and float(override) <= 0:
            continue
        resolved[k] = override
    # A negative seed means "random" both in the request and in speaker defaults.
    seed = resolved["seed"]
    resolved["seed"] = int(seed) if seed is not None and int(seed) >= 0 else None
    if resolved["duration_scale"] is not None and float(resolved["duration_scale"]) <= 0:
        raise ApiError(
            f"resolved duration_scale ({resolved['duration_scale']}) must be > 0 "
            "after merging speaker defaults",
            param="voice",
        )
    if (
        resolved["min_seconds"] is not None
        and resolved["max_seconds"] is not None
        and float(resolved["min_seconds"]) > float(resolved["max_seconds"])
    ):
        raise ApiError(
            f"resolved min_seconds ({resolved['min_seconds']}) > "
            f"max_seconds ({resolved['max_seconds']}) after merging speaker defaults",
            param="min_seconds",
        )
    return resolved
