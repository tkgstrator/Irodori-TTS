"""Synthesis for /v1/audio/speech: one LoRA speaker, one utterance."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from irodori_tts.inference_runtime import SamplingRequest, resolve_cfg_scales
from irodori_tts.server.audio import _apply_fade
from irodori_tts.server.config import ServerConfig
from irodori_tts.server.errors import ApiError
from irodori_tts.server.registry import RuntimeRegistry
from irodori_tts.server.schemas import SpeechRequest, _merge_defaults
from irodori_tts.server.shortcodes import expand_shortcodes

logger = logging.getLogger("irodori_tts.server")


@dataclass(frozen=True)
class Synthesized:
    audio: np.ndarray
    sample_rate: int
    used_seed: int
    speaker_id: str
    speaker_name: str
    input_tokens: int
    output_tokens: int


def synthesize(registry: RuntimeRegistry, cfg: ServerConfig, req: SpeechRequest) -> Synthesized:
    if req.instructions is not None:
        raise ApiError(
            "instructions is not supported by this server: voice design was removed. "
            "Pick a voice from GET /v1/audio/voices instead.",
            param="instructions",
        )

    voice = req.voice_id
    try:
        with registry.acquire(voice) as (runtime, spec):
            text = expand_shortcodes(req.input)
            params = _merge_defaults(req, spec.defaults)

            cfg_text, cfg_caption, cfg_speaker, _messages = resolve_cfg_scales(
                cfg_guidance_mode="independent",
                cfg_scale_text=float(params["cfg_scale_text"]),
                cfg_scale_caption=3.0,
                cfg_scale_speaker=float(params["cfg_scale_speaker"]),
                cfg_scale=None,
                use_caption_condition=False,
                use_speaker_condition=bool(runtime.model_cfg.use_speaker_condition),
            )

            sampling_req = SamplingRequest(
                # The adapter registry.acquire() just activated must survive
                # synthesize(): without keep_adapter, _prepare_lora_for_request()
                # disables it and every speaker comes out as the base voice.
                keep_adapter=True,
                text=text,
                caption=None,
                ref_wav=None,
                ref_latent=None,
                no_ref=True,
                ref_normalize_db=-16.0,
                ref_ensure_max=True,
                num_candidates=1,
                decode_mode="sequential",
                seconds=params["seconds"],
                # speed divides the duration the predictor asked for, and stacks on top
                # of whatever scale the speaker's own defaults carry.
                duration_scale=float(params["duration_scale"]) / float(req.speed),
                min_seconds=float(params["min_seconds"]),
                max_seconds=float(params["max_seconds"]),
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
                result = runtime.synthesize(
                    sampling_req, log_fn=logger.debug if cfg.show_timings else None
                )
            except Exception as e:
                logger.exception("synthesis failed")
                raise ApiError(
                    f"synthesis failed: {e}", status_code=500, error_type="server_error"
                ) from e

            audio = result.audio
            audio_np = (
                audio.squeeze(0).cpu().float().numpy()
                if audio.ndim == 2
                else audio.cpu().float().numpy()
            )
            sample_rate = int(result.sample_rate)
            audio_np = _apply_fade(audio_np, sample_rate)

            hop_length = int(runtime.codec.model.hop_length)
            return Synthesized(
                audio=audio_np,
                sample_rate=sample_rate,
                used_seed=int(result.used_seed),
                speaker_id=spec.uuid,
                speaker_name=spec.name,
                input_tokens=int(runtime.tokenizer.encode(text).numel()),
                output_tokens=math.ceil(len(audio_np) / hop_length),
            )
    except KeyError as err:
        raise ApiError(f"unknown voice: {voice}", param="voice", code="voice_not_found") from err
