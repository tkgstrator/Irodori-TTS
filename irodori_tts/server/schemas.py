"""Request schemas for the TTS server and speaker-defaults merging."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# VDS-JSON schema models (OpenAPI documentation)
# ---------------------------------------------------------------------------


class VdsLoraSpeaker(BaseModel):
    type: Literal["lora"]
    uuid: str = Field(
        ...,
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        description="Speaker LoRA adapter UUID.",
        examples=["7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb"],
    )


class VdsCaptionSpeaker(BaseModel):
    type: Literal["caption"]
    caption: str = Field(
        ...,
        min_length=1,
        description="Natural-language voice description for VoiceDesign synthesis.",
        examples=["落ち着いた女性の声で、やわらかく自然に"],
    )


VdsSpeakerRef = Annotated[
    VdsLoraSpeaker | VdsCaptionSpeaker,
    Field(discriminator="type"),
]


class VdsSynthOptions(BaseModel):
    seed: int | None = Field(default=None, description="Sampling seed.")
    num_steps: int | None = Field(default=None, description="RF sampling steps.")
    cfg_scale_text: float | None = Field(default=None, description="Text CFG scale.")
    cfg_scale_speaker: float | None = Field(default=None, description="Speaker CFG scale.")
    speaker_kv_scale: float | None = Field(default=None, description="Speaker KV scale.")
    truncation_factor: float | None = Field(default=None, description="Noise truncation factor.")
    seconds: float | None = Field(
        default=None,
        gt=0,
        description="Manual synthesis duration in seconds (overrides the duration predictor).",
    )
    min_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Lower bound for the duration predictor output (default 0.5).",
    )
    max_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Upper bound for the duration predictor output (default 30.0).",
    )
    duration_scale: float | None = Field(
        default=None,
        gt=0,
        description="Multiplier applied to the predicted duration (default 1.0).",
    )


class VdsSpeechCue(BaseModel):
    kind: Literal["speech"]
    speaker: str = Field(..., description="Speaker alias defined in the speakers map.")
    text: str = Field(
        ...,
        min_length=1,
        description="Text to synthesize. Supports {shortcode} emoji annotations.",
    )
    options: VdsSynthOptions | None = Field(
        default=None,
        description="Per-cue synthesis parameter overrides.",
    )


class VdsPauseCue(BaseModel):
    kind: Literal["pause"]
    duration: float = Field(..., gt=0, description="Pause duration in seconds.")


class VdsSceneCue(BaseModel):
    kind: Literal["scene"]
    name: str = Field(..., min_length=1, description="Scene marker name (not synthesized).")


VdsCue = Annotated[
    VdsSpeechCue | VdsPauseCue | VdsSceneCue,
    Field(discriminator="kind"),
]


class VdsDefaults(BaseModel):
    gap: float = Field(
        default=1.0,
        ge=0,
        description="Gap between consecutive speech cues in seconds.",
    )
    num_steps: int | None = Field(default=None, description="Default RF sampling steps.")
    cfg_scale_text: float | None = Field(default=None, description="Default text CFG scale.")
    cfg_scale_speaker: float | None = Field(default=None, description="Default speaker CFG scale.")
    speaker_kv_scale: float | None = Field(default=None, description="Default speaker KV scale.")
    truncation_factor: float | None = Field(
        default=None, description="Default noise truncation factor."
    )
    seed: int | None = Field(default=None, description="Default sampling seed.")
    seconds: float | None = Field(
        default=None,
        gt=0,
        description="Default manual synthesis duration in seconds (overrides the predictor).",
    )
    min_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Default lower bound for the duration predictor output.",
    )
    max_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Default upper bound for the duration predictor output.",
    )
    duration_scale: float | None = Field(
        default=None,
        gt=0,
        description="Default multiplier applied to the predicted duration.",
    )


class VdsScriptBody(BaseModel):
    version: Literal[1] = Field(..., description="VDS format version. Must be 1.")
    title: str | None = Field(default=None, description="Script title.")
    defaults: VdsDefaults | None = Field(
        default=None,
        description="Default synthesis parameters applied to all cues.",
    )
    speakers: dict[str, VdsSpeakerRef] = Field(
        ...,
        description="Map of speaker aliases to speaker definitions (LoRA UUID or VoiceDesign caption).",
    )
    cues: list[VdsCue] = Field(
        ...,
        description="Ordered list of cues to synthesize.",
    )


class SynthRequest(BaseModel):
    speaker_id: str | None = Field(
        default=None,
        description="Registered speaker UUID. Required for single-cue mode.",
        examples=["7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb"],
    )
    text: str | None = Field(
        default=None,
        min_length=1,
        description="Text to synthesize. Required for single-cue mode.",
        examples=["こんにちは、今日はいい天気ですね。"],
    )
    seed: int | None = Field(default=None, description="Sampling seed. Omit or set <0 for random.")
    num_steps: int | None = Field(
        default=None, description="RF sampling steps. Omit or set <=0 to use speaker default."
    )
    cfg_scale_text: float | None = Field(
        default=None, description="Text CFG scale. Omit or set <=0 to use speaker default."
    )
    cfg_scale_speaker: float | None = Field(
        default=None, description="Speaker CFG scale. Omit or set <=0 to use speaker default."
    )
    caption: str | None = Field(
        default=None,
        description="Natural-language voice description for VoiceDesign mode. "
        "Alternative to speaker_id (mutually exclusive).",
        examples=["落ち着いた女性の声で、やわらかく自然に"],
    )
    cfg_scale_caption: float | None = Field(
        default=None, description="Caption CFG scale (VoiceDesign mode). Default 3.0."
    )
    speaker_kv_scale: float | None = Field(
        default=None,
        description="Speaker KV scale (>1 strengthens identity). Omit or set <=0 to disable.",
    )
    truncation_factor: float | None = Field(
        default=None, description="Noise truncation (e.g. 0.8). Omit or set <=0 to disable."
    )
    script: VdsScriptBody | None = Field(
        default=None,
        description="VDS-JSON script object for drama mode. "
        "If provided, speaker_id/text/caption are ignored.",
    )
    seconds: float | None = Field(
        default=None,
        gt=0,
        description="Manual synthesis duration in seconds. "
        "When set, overrides the duration predictor; clamped by min/max_seconds.",
    )
    min_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Lower bound for the duration predictor output. "
        "Default 0.5s. Useful when very short text yields too-short audio.",
    )
    max_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Upper bound for the duration predictor output. Default 30.0s.",
    )
    duration_scale: float | None = Field(
        default=None,
        gt=0,
        description="Multiplier applied to the predicted duration. Default 1.0.",
    )

    @model_validator(mode="after")
    def _check_duration_bounds(self) -> SynthRequest:
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


def _merge_defaults(req: SynthRequest, defaults: dict[str, Any]) -> dict[str, Any]:
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
    resolved["seed"] = req.seed if (req.seed is not None and req.seed >= 0) else None
    if (
        resolved["min_seconds"] is not None
        and resolved["max_seconds"] is not None
        and float(resolved["min_seconds"]) > float(resolved["max_seconds"])
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"resolved min_seconds ({resolved['min_seconds']}) > "
                f"max_seconds ({resolved['max_seconds']}) after merging speaker defaults"
            ),
        )
    return resolved
