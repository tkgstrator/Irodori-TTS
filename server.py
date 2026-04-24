"""FastAPI TTS server for Irodori-TTS speaker LoRAs."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import threading
import urllib.parse
import uuid as uuid_lib
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import soundfile as sf
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    resolve_cfg_scales,
)
from irodori_tts.lora import (
    is_lora_safetensors_file,
    read_lora_safetensors_metadata,
)
from irodori_tts.vds import (
    CaptionSpeaker,
    PauseCue,
    SceneCue,
    SpeechCue,
    VdsScript,
    parse_json,
    parse_text,
)
from irodori_tts.vds.shortcodes import expand_shortcodes
from irodori_tts.vds.parser import ParseError

FIXED_SECONDS = 30.0
_FADE_MS = 50


def _apply_fade(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    fade_samples = int(sample_rate * _FADE_MS / 1000)
    if fade_samples <= 0 or len(audio) < fade_samples * 2:
        return audio
    audio = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio.dtype)
    audio[:fade_samples] *= fade_in
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=audio.dtype)
    audio[-fade_samples:] *= fade_out
    return audio


# Namespace for deterministic UUIDv5 derivation from LoRA filenames. Lets users
# drop .safetensors files into models/LoRA/ without assigning ids manually.
_LORA_UUID_NAMESPACE = uuid_lib.UUID("8e6d8a0e-5a52-4a1e-8c8d-4c3e2f6a1b9f")

logger = logging.getLogger("irodori_tts.server")


@dataclass
class SpeakerSpec:
    uuid: str
    name: str
    adapter: str
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerConfig:
    base_checkpoint: str | None
    base_hf_repo: str | None
    base_hf_filename: str
    model_device: str
    codec_device: str
    model_precision: str
    codec_precision: str
    codec_repo: str
    codec_deterministic_encode: bool
    codec_deterministic_decode: bool
    enable_watermark: bool
    caption_checkpoint: str | None
    caption_hf_repo: str | None
    caption_hf_filename: str
    tail_window_size: int
    tail_std_threshold: float
    tail_mean_threshold: float
    show_timings: bool
    speakers: list[SpeakerSpec]


def _discover_lora_dir(lora_dir: Path) -> list[SpeakerSpec]:
    """Discover standalone .safetensors LoRA exports under ``lora_dir``.

    Each file must carry Irodori-TTS metadata (``name``, ``uuid``,
    ``adapter_config``). ``defaults`` is optional.
    """
    if not lora_dir.is_dir():
        raise FileNotFoundError(f"lora_dir does not exist: {lora_dir}")
    specs: list[SpeakerSpec] = []
    for entry in sorted(lora_dir.glob("*.safetensors")):
        if not is_lora_safetensors_file(entry):
            logger.warning("skipping non-LoRA safetensors file: %s", entry)
            continue
        try:
            meta = read_lora_safetensors_metadata(entry)
        except Exception as exc:
            logger.warning("failed to read metadata from %s: %s", entry, exc)
            continue
        name = meta.get("name") or entry.stem
        speaker_uuid = meta.get("uuid") or str(uuid_lib.uuid5(_LORA_UUID_NAMESPACE, entry.stem))
        defaults: dict[str, Any] = {}
        raw_defaults = meta.get("defaults")
        if raw_defaults:
            try:
                parsed = json.loads(raw_defaults)
                if isinstance(parsed, dict):
                    defaults = parsed
            except json.JSONDecodeError as exc:
                logger.warning("skipping defaults in %s: %s", entry, exc)
        specs.append(
            SpeakerSpec(
                uuid=str(speaker_uuid),
                name=str(name),
                adapter=str(entry),
                defaults=defaults,
            )
        )
        logger.info("discovered LoRA: %s (uuid=%s)", name, speaker_uuid)
    return specs


def load_config(path: Path) -> ServerConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    speakers: list[SpeakerSpec] = []
    lora_dir_raw = raw.get("lora_dir")
    if lora_dir_raw:
        lora_dir = Path(str(lora_dir_raw))
        if not lora_dir.is_absolute():
            lora_dir = (path.parent / lora_dir).resolve() if not lora_dir.exists() else lora_dir
        speakers.extend(_discover_lora_dir(lora_dir))

    for s in raw.get("speakers") or []:
        speakers.append(
            SpeakerSpec(
                uuid=str(s["uuid"]),
                name=str(s["name"]),
                adapter=str(s["adapter"]),
                defaults=dict(s.get("defaults") or {}),
            )
        )

    return ServerConfig(
        base_checkpoint=(str(raw["base_checkpoint"]) if raw.get("base_checkpoint") else None),
        base_hf_repo=(str(raw["base_hf_repo"]) if raw.get("base_hf_repo") else None),
        base_hf_filename=str(raw.get("base_hf_filename", "model.safetensors")),
        model_device=str(raw.get("model_device", "cuda")),
        codec_device=str(raw.get("codec_device", "cuda")),
        model_precision=str(raw.get("model_precision", "bf16")),
        codec_precision=str(raw.get("codec_precision", "fp32")),
        codec_repo=str(raw.get("codec_repo", "Aratako/Semantic-DACVAE-Japanese-32dim")),
        codec_deterministic_encode=bool(raw.get("codec_deterministic_encode", True)),
        codec_deterministic_decode=bool(raw.get("codec_deterministic_decode", True)),
        enable_watermark=bool(raw.get("enable_watermark", False)),
        caption_checkpoint=(str(raw["caption_checkpoint"]) if raw.get("caption_checkpoint") else None),
        caption_hf_repo=(str(raw["caption_hf_repo"]) if raw.get("caption_hf_repo") else None),
        caption_hf_filename=str(raw.get("caption_hf_filename", "model.safetensors")),
        tail_window_size=int(raw.get("tail_window_size", 20)),
        tail_std_threshold=float(raw.get("tail_std_threshold", 0.05)),
        tail_mean_threshold=float(raw.get("tail_mean_threshold", 0.1)),
        show_timings=bool(raw.get("show_timings", True)),
        speakers=speakers,
    )


def _resolve_checkpoint(
    local_path: str | None, hf_repo: str | None, hf_filename: str, label: str,
) -> Path:
    local = Path(local_path) if local_path else None
    if local is not None and local.exists():
        logger.info("Using local %s checkpoint: %s", label, local)
        return local
    if not hf_repo:
        raise FileNotFoundError(
            f"{label} checkpoint not found at {local_path!r} and hf_repo is not set."
        )
    from huggingface_hub import hf_hub_download

    logger.info("Downloading %s checkpoint from HF: %s/%s", label, hf_repo, hf_filename)
    cached = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
    logger.info("%s checkpoint cached at: %s", label.capitalize(), cached)
    return Path(cached)


def resolve_base_checkpoint(cfg: ServerConfig) -> Path:
    return _resolve_checkpoint(
        cfg.base_checkpoint, cfg.base_hf_repo, cfg.base_hf_filename, "base",
    )


class RuntimeRegistry:
    """LoRA runtime + optional caption (VoiceDesign) runtime."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._by_uuid: dict[str, SpeakerSpec] = {s.uuid: s for s in cfg.speakers}
        self._runtime: InferenceRuntime | None = None
        self._caption_runtime: InferenceRuntime | None = None
        self._lock = threading.Lock()

    def list_speakers(self) -> list[SpeakerSpec]:
        return list(self.cfg.speakers)

    def get_spec(self, uuid: str) -> SpeakerSpec:
        spec = self._by_uuid.get(uuid)
        if spec is None:
            raise KeyError(uuid)
        return spec

    @property
    def caption_available(self) -> bool:
        return self._caption_runtime is not None

    def _make_key(self, checkpoint: str) -> RuntimeKey:
        return RuntimeKey(
            checkpoint=checkpoint,
            model_device=self.cfg.model_device,
            codec_repo=self.cfg.codec_repo,
            model_precision=self.cfg.model_precision,
            codec_device=self.cfg.codec_device,
            codec_precision=self.cfg.codec_precision,
            codec_deterministic_encode=self.cfg.codec_deterministic_encode,
            codec_deterministic_decode=self.cfg.codec_deterministic_decode,
            enable_watermark=self.cfg.enable_watermark,
            compile_model=False,
            compile_dynamic=False,
        )

    def load(self) -> None:
        if self.cfg.speakers:
            base_path = resolve_base_checkpoint(self.cfg)
            adapters = {s.uuid: s.adapter for s in self.cfg.speakers}
            logger.info("Loading base + %d LoRA adapters", len(adapters))
            self._runtime = InferenceRuntime.from_base_with_adapters(
                key=self._make_key(str(base_path)),
                adapters=adapters,
                default_adapter=self.cfg.speakers[0].uuid,
            )
        else:
            logger.warning("No LoRA speakers configured — LoRA synthesis disabled")

        if self.cfg.caption_checkpoint or self.cfg.caption_hf_repo:
            caption_path = _resolve_checkpoint(
                self.cfg.caption_checkpoint,
                self.cfg.caption_hf_repo,
                self.cfg.caption_hf_filename,
                "caption",
            )
            logger.info("Loading caption (VoiceDesign) runtime")
            self._caption_runtime = InferenceRuntime.from_key(
                self._make_key(str(caption_path))
            )

    def acquire(self, uuid: str) -> tuple[InferenceRuntime, SpeakerSpec]:
        spec = self.get_spec(uuid)
        if self._runtime is None:
            raise RuntimeError("Registry not loaded. Call load() first.")
        with self._lock:
            self._runtime.set_active_adapter(uuid)
            return self._runtime, spec

    def acquire_caption(self) -> InferenceRuntime:
        if self._caption_runtime is None:
            raise RuntimeError("Caption runtime not configured.")
        return self._caption_runtime


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


class VdsSpeechCue(BaseModel):
    kind: Literal["speech"]
    speaker: str = Field(..., description="Speaker alias defined in the speakers map.")
    text: str = Field(
        ...,
        min_length=1,
        description="Text to synthesize. Supports {shortcode} emoji annotations.",
    )
    options: VdsSynthOptions | None = Field(
        default=None, description="Per-cue synthesis parameter overrides.",
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
        default=1.0, ge=0,
        description="Gap between consecutive speech cues in seconds.",
    )
    num_steps: int | None = Field(default=None, description="Default RF sampling steps.")
    cfg_scale_text: float | None = Field(default=None, description="Default text CFG scale.")
    cfg_scale_speaker: float | None = Field(default=None, description="Default speaker CFG scale.")
    speaker_kv_scale: float | None = Field(default=None, description="Default speaker KV scale.")
    truncation_factor: float | None = Field(default=None, description="Default noise truncation factor.")
    seed: int | None = Field(default=None, description="Default sampling seed.")


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
    return resolved


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
                {"uuid": s.uuid, "name": s.name, "defaults": s.defaults}
                for s in registry.list_speakers()
            ]
        }

    def _caption_sampling_req(
        text: str,
        caption_text: str,
        *,
        num_steps: int = 40,
        cfg_scale_text: float = 3.0,
        cfg_scale_caption: float = 3.0,
        truncation_factor: float | None = None,
        seed: int | None = None,
    ) -> SamplingRequest:
        cfg_text, cfg_cap, _, _ = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=0.0,
            cfg_scale=None,
            use_caption_condition=True,
            use_speaker_condition=False,
        )
        return SamplingRequest(
            text=text,
            caption=caption_text,
            ref_wav=None,
            ref_latent=None,
            no_ref=True,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            seconds=FIXED_SECONDS,
            max_ref_seconds=30.0,
            max_text_len=None,
            max_caption_len=None,
            num_steps=num_steps,
            cfg_scale_text=cfg_text,
            cfg_scale_caption=cfg_cap,
            cfg_scale_speaker=0.0,
            cfg_guidance_mode="independent",
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=truncation_factor,
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            seed=seed,
            trim_tail=True,
            tail_window_size=cfg.tail_window_size,
            tail_std_threshold=cfg.tail_std_threshold,
            tail_mean_threshold=cfg.tail_mean_threshold,
        )

    def _synth_single(req: SynthRequest, request: Request) -> Response:
        """Single-cue synthesis. Returns WAV by default, raw PCM16 mono when Accept: audio/pcm."""
        if not req.text:
            raise HTTPException(status_code=422, detail="'text' is required")
        req.text = expand_shortcodes(req.text)
        if req.speaker_id and req.caption:
            raise HTTPException(
                status_code=422,
                detail="'speaker_id' and 'caption' are mutually exclusive",
            )
        if not req.speaker_id and not req.caption:
            raise HTTPException(
                status_code=422,
                detail="either 'speaker_id' or 'caption' is required",
            )

        if req.caption:
            try:
                runtime = registry.acquire_caption()
            except RuntimeError as err:
                raise HTTPException(
                    status_code=501, detail="caption runtime not configured"
                ) from err

            num_steps = int(req.num_steps) if req.num_steps and req.num_steps > 0 else 40
            cfg_text = float(req.cfg_scale_text) if req.cfg_scale_text and req.cfg_scale_text > 0 else 3.0
            cfg_cap = float(req.cfg_scale_caption) if req.cfg_scale_caption and req.cfg_scale_caption > 0 else 3.0
            trunc = req.truncation_factor if req.truncation_factor and req.truncation_factor > 0 else None
            seed = req.seed if req.seed is not None and req.seed >= 0 else None

            sampling_req = _caption_sampling_req(
                req.text, req.caption,
                num_steps=num_steps,
                cfg_scale_text=cfg_text,
                cfg_scale_caption=cfg_cap,
                truncation_factor=trunc,
                seed=seed,
            )

            try:
                result = runtime.synthesize(sampling_req, log_fn=logger.debug if cfg.show_timings else None)
            except Exception as e:
                logger.exception("caption synthesis failed")
                raise HTTPException(status_code=500, detail=f"synthesis failed: {e}") from e

            audio = result.audio
            audio_np = audio.squeeze(0).cpu().numpy() if audio.ndim == 2 else audio.cpu().numpy()
            audio_np = _apply_fade(audio_np, int(result.sample_rate))
            headers = {
                "X-TTS-Used-Seed": str(int(result.used_seed)),
                "X-TTS-Sample-Rate": str(int(result.sample_rate)),
            }
            if _wants_wav(request):
                buf = io.BytesIO()
                sf.write(buf, audio_np, int(result.sample_rate), format="WAV", subtype="PCM_16")
                return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)
            return Response(content=_to_pcm16(audio_np), media_type="audio/pcm", headers=headers)

        # LoRA speaker path
        try:
            runtime, spec = registry.acquire(req.speaker_id)  # type: ignore[arg-type]
        except KeyError as err:
            raise HTTPException(
                status_code=404, detail=f"unknown speaker_id: {req.speaker_id}"
            ) from err

        params = _merge_defaults(req, spec.defaults)

        use_speaker = bool(runtime.model_cfg.use_speaker_condition)
        cfg_text, cfg_caption, cfg_speaker, _messages = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=float(params["cfg_scale_text"]),
            cfg_scale_caption=3.0,
            cfg_scale_speaker=float(params["cfg_scale_speaker"]),
            cfg_scale=None,
            use_caption_condition=False,
            use_speaker_condition=use_speaker,
        )

        sampling_req = SamplingRequest(
            text=req.text,
            caption=None,
            ref_wav=None,
            ref_latent=None,
            no_ref=True,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            seconds=FIXED_SECONDS,
            max_ref_seconds=30.0,
            max_text_len=None,
            max_caption_len=None,
            num_steps=int(params["num_steps"]),
            cfg_scale_text=cfg_text,
            cfg_scale_caption=cfg_caption,
            cfg_scale_speaker=cfg_speaker,
            cfg_guidance_mode="independent",
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=params["truncation_factor"],
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=params["speaker_kv_scale"],
            speaker_kv_min_t=0.9 if params["speaker_kv_scale"] is not None else None,
            speaker_kv_max_layers=None,
            seed=params["seed"],
            trim_tail=True,
            tail_window_size=cfg.tail_window_size,
            tail_std_threshold=cfg.tail_std_threshold,
            tail_mean_threshold=cfg.tail_mean_threshold,
        )

        try:
            result = runtime.synthesize(sampling_req, log_fn=logger.debug if cfg.show_timings else None)
        except Exception as e:
            logger.exception("synthesis failed")
            raise HTTPException(status_code=500, detail=f"synthesis failed: {e}") from e

        audio = result.audio
        if audio.ndim == 2:
            audio_np = audio.squeeze(0).cpu().numpy()
        else:
            audio_np = audio.cpu().numpy()
        audio_np = _apply_fade(audio_np, int(result.sample_rate))

        headers = {
            "X-TTS-Speaker-Id": spec.uuid,
            "X-TTS-Speaker-Name": urllib.parse.quote(spec.name),
            "X-TTS-Used-Seed": str(int(result.used_seed)),
            "X-TTS-Sample-Rate": str(int(result.sample_rate)),
        }
        if _wants_wav(request):
            buf = io.BytesIO()
            sf.write(buf, audio_np, int(result.sample_rate), format="WAV", subtype="PCM_16")
            return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)
        return Response(content=_to_pcm16(audio_np), media_type="audio/pcm", headers=headers)

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
        if req.script is not None:
            try:
                script, warnings = parse_json(req.script.model_dump(exclude_none=True))
            except ParseError as e:
                raise HTTPException(status_code=422, detail=str(e)) from e
            for w in warnings:
                logger.warning("VDS warning: %s", w)
            return _render_drama(script, request)
        return _synth_single(req, request)

    def _synth_cue(
        cue: SpeechCue,
        script: VdsScript,
    ) -> tuple[np.ndarray, int]:
        """Synthesize a single speech cue, returning (pcm_array, sample_rate)."""
        ref = script.speakers[cue.speaker]

        synth_defaults = script.defaults.synth
        cue_num_steps = int(synth_defaults.num_steps) if synth_defaults.num_steps else 40
        cue_cfg_text = float(synth_defaults.cfg_scale_text) if synth_defaults.cfg_scale_text else 3.0
        cue_trunc = synth_defaults.truncation_factor
        cue_seed: int | None = int(synth_defaults.seed) if synth_defaults.seed is not None else None
        if cue.options:
            if cue.options.num_steps is not None:
                cue_num_steps = int(cue.options.num_steps)
            if cue.options.cfg_scale_text is not None:
                cue_cfg_text = float(cue.options.cfg_scale_text)
            if cue.options.truncation_factor is not None:
                cue_trunc = cue.options.truncation_factor
            if cue.options.seed is not None:
                cue_seed = int(cue.options.seed)

        if isinstance(ref, CaptionSpeaker):
            runtime = registry.acquire_caption()
            sampling_req = _caption_sampling_req(
                cue.text, ref.caption,
                num_steps=cue_num_steps,
                cfg_scale_text=cue_cfg_text,
                cfg_scale_caption=3.0,
                truncation_factor=cue_trunc,
                seed=cue_seed if cue_seed is not None and cue_seed >= 0 else None,
            )
            result = runtime.synthesize(sampling_req, log_fn=logger.debug if cfg.show_timings else None)
            audio = result.audio
            audio_np = audio.squeeze(0).cpu().numpy() if audio.ndim == 2 else audio.cpu().numpy()
            return _apply_fade(audio_np, int(result.sample_rate)), int(result.sample_rate)

        # LoRA speaker path
        try:
            runtime, spec = registry.acquire(ref.uuid)
        except KeyError as err:
            raise RuntimeError(
                f"unknown speaker UUID: {ref.uuid} (alias: {cue.speaker!r})"
            ) from err

        merged_defaults = dict(spec.defaults)
        for key in ("num_steps", "cfg_scale_text", "cfg_scale_speaker",
                     "speaker_kv_scale", "truncation_factor", "seed"):
            val = getattr(synth_defaults, key, None)
            if val is not None:
                merged_defaults[key] = val

        mock_req_fields: dict[str, Any] = {"speaker_id": ref.uuid, "text": cue.text}
        if cue.options:
            for key in ("seed", "num_steps", "cfg_scale_text", "cfg_scale_speaker",
                         "speaker_kv_scale", "truncation_factor"):
                val = getattr(cue.options, key, None)
                if val is not None:
                    mock_req_fields[key] = val
        mock_req = SynthRequest(**mock_req_fields)
        params = _merge_defaults(mock_req, merged_defaults)

        use_speaker = bool(runtime.model_cfg.use_speaker_condition)
        cfg_text, cfg_caption, cfg_speaker, _ = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=float(params["cfg_scale_text"]),
            cfg_scale_caption=3.0,
            cfg_scale_speaker=float(params["cfg_scale_speaker"]),
            cfg_scale=None,
            use_caption_condition=False,
            use_speaker_condition=use_speaker,
        )

        sampling_req = SamplingRequest(
            text=cue.text,
            caption=None,
            ref_wav=None,
            ref_latent=None,
            no_ref=True,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            seconds=FIXED_SECONDS,
            max_ref_seconds=30.0,
            max_text_len=None,
            max_caption_len=None,
            num_steps=int(params["num_steps"]),
            cfg_scale_text=cfg_text,
            cfg_scale_caption=cfg_caption,
            cfg_scale_speaker=cfg_speaker,
            cfg_guidance_mode="independent",
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=params["truncation_factor"],
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=params["speaker_kv_scale"],
            speaker_kv_min_t=0.9 if params["speaker_kv_scale"] is not None else None,
            speaker_kv_max_layers=None,
            seed=params["seed"],
            trim_tail=True,
            tail_window_size=cfg.tail_window_size,
            tail_std_threshold=cfg.tail_std_threshold,
            tail_mean_threshold=cfg.tail_mean_threshold,
        )

        result = runtime.synthesize(sampling_req, log_fn=logger.debug if cfg.show_timings else None)
        audio = result.audio
        audio_np = audio.squeeze(0).cpu().numpy() if audio.ndim == 2 else audio.cpu().numpy()
        return _apply_fade(audio_np, int(result.sample_rate)), int(result.sample_rate)

    def _to_pcm16(audio_np: np.ndarray) -> bytes:
        pcm = np.clip(audio_np, -1.0, 1.0)
        return (pcm * 32767).astype(np.int16).tobytes()

    def _silence_pcm(duration: float, sample_rate: int) -> bytes:
        return b"\x00\x00" * int(duration * sample_rate)

    def _stream_drama_pcm(script: VdsScript) -> Generator[bytes, None, None]:
        """Yield raw PCM16 mono bytes, with gap/pause as silence."""
        sample_rate: int | None = None
        prev_was_speech = False

        for cue in script.cues:
            if isinstance(cue, SceneCue):
                continue

            if isinstance(cue, PauseCue):
                if sample_rate is not None:
                    yield _silence_pcm(cue.duration, sample_rate)
                prev_was_speech = False
                continue

            if isinstance(cue, SpeechCue):
                if prev_was_speech and script.defaults.gap > 0 and sample_rate is not None:
                    yield _silence_pcm(script.defaults.gap, sample_rate)

                try:
                    audio_np, sr = _synth_cue(cue, script)
                except Exception:
                    logger.exception("synthesis failed for cue (speaker=%s), skipping", cue.speaker)
                    prev_was_speech = False
                    continue

                if sample_rate is None:
                    sample_rate = sr
                yield _to_pcm16(audio_np)
                prev_was_speech = True

    def _validate_drama(script: VdsScript) -> list[SpeechCue]:
        """Pre-flight checks before streaming. Returns speech cues."""
        speech_cues = [c for c in script.cues if isinstance(c, SpeechCue)]
        if not speech_cues:
            raise HTTPException(status_code=422, detail="no speech cues in script")

        for cue in speech_cues:
            ref = script.speakers[cue.speaker]
            if isinstance(ref, CaptionSpeaker):
                if not registry.caption_available:
                    raise HTTPException(
                        status_code=501,
                        detail=f"caption runtime not configured (alias: {cue.speaker!r})",
                    )
            else:
                try:
                    registry.get_spec(ref.uuid)
                except KeyError as err:
                    raise HTTPException(
                        status_code=404,
                        detail=f"unknown speaker UUID: {ref.uuid} (alias: {cue.speaker!r})",
                    ) from err
        return speech_cues

    def _get_sample_rate() -> int:
        for rt in (registry._runtime, registry._caption_runtime):
            if rt is not None:
                return int(rt.codec.sample_rate)
        return 24000

    def _render_drama_wav(script: VdsScript, speech_cues: list[SpeechCue]) -> Response:
        """Synthesize all cues and return a single concatenated WAV."""
        segments: list[np.ndarray] = []
        sample_rate: int | None = None
        prev_was_speech = False

        for cue in script.cues:
            if isinstance(cue, SceneCue):
                continue
            if isinstance(cue, PauseCue):
                if sample_rate is not None:
                    segments.append(np.zeros(int(cue.duration * sample_rate), dtype=np.float32))
                prev_was_speech = False
                continue
            if isinstance(cue, SpeechCue):
                if prev_was_speech and script.defaults.gap > 0 and sample_rate is not None:
                    segments.append(np.zeros(int(script.defaults.gap * sample_rate),
                                             dtype=np.float32))
                try:
                    audio_np, sr = _synth_cue(cue, script)
                except Exception:
                    logger.exception("synthesis failed for cue (speaker=%s), skipping", cue.speaker)
                    prev_was_speech = False
                    continue
                if sample_rate is None:
                    sample_rate = sr
                segments.append(audio_np)
                prev_was_speech = True

        if not segments or sample_rate is None:
            raise HTTPException(status_code=500, detail="all cues failed to synthesize")

        combined = np.concatenate(segments)
        buf = io.BytesIO()
        sf.write(buf, combined, sample_rate, format="WAV", subtype="PCM_16")
        return Response(
            content=buf.getvalue(),
            media_type="audio/wav",
            headers={
                "X-TTS-Sample-Rate": str(sample_rate),
                "X-TTS-Cue-Count": str(len(speech_cues)),
            },
        )

    def _wants_wav(request: Request) -> bool:
        accept = request.headers.get("accept", "")
        return "audio/wav" in accept

    def _render_drama(
        script: VdsScript, request: Request
    ) -> Response:
        speech_cues = _validate_drama(script)

        if _wants_wav(request):
            return _render_drama_wav(script, speech_cues)

        sample_rate = _get_sample_rate()
        return StreamingResponse(
            _stream_drama_pcm(script),
            media_type="audio/pcm",
            headers={
                "X-TTS-Sample-Rate": str(sample_rate),
                "X-TTS-Cue-Count": str(len(speech_cues)),
            },
        )

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
    async def synth_vds(
        file: UploadFile, request: Request
    ) -> Response:
        content = await file.read()
        try:
            source = content.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=422, detail="file must be UTF-8 encoded") from e
        try:
            script, warnings = parse_text(source)
        except ParseError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        for w in warnings:
            logger.warning("VDS warning: %s", w)
        return _render_drama(script, request)

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
