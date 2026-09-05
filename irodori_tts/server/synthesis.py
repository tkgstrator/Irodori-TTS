"""Synthesis path for the TTS server: single-cue, drama rendering and PCM streaming."""

from __future__ import annotations

import io
import logging
import urllib.parse
from collections.abc import Generator
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from irodori_tts.inference_runtime import SamplingRequest, resolve_cfg_scales
from irodori_tts.server.audio import _apply_fade
from irodori_tts.server.config import ServerConfig
from irodori_tts.server.registry import RuntimeRegistry
from irodori_tts.server.schemas import SynthRequest, _merge_defaults
from irodori_tts.vds import (
    CaptionSpeaker,
    PauseCue,
    SceneCue,
    SpeechCue,
    VdsScript,
    parse_json,
    parse_text,
)
from irodori_tts.vds.parser import ParseError
from irodori_tts.vds.shortcodes import expand_shortcodes

logger = logging.getLogger("irodori_tts.server")


def _wants_wav(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "audio/wav" in accept


def _to_pcm16(audio_np: np.ndarray) -> bytes:
    pcm = np.clip(audio_np, -1.0, 1.0)
    return (pcm * 32767).astype(np.int16).tobytes()


def _silence_pcm(duration: float, sample_rate: int) -> bytes:
    return b"\x00\x00" * int(duration * sample_rate)


def _caption_sampling_req(  # noqa: PLR0913
    cfg: ServerConfig,
    text: str,
    caption_text: str,
    *,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    truncation_factor: float | None = None,
    seed: int | None = None,
    seconds: float | None = None,
    min_seconds: float = 0.5,
    max_seconds: float = 30.0,
    duration_scale: float = 1.0,
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
        seconds=seconds,
        duration_scale=duration_scale,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
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


def _synth_single(  # noqa: C901, PLR0915
    registry: RuntimeRegistry,
    cfg: ServerConfig,
    req: SynthRequest,
    request: Request,
) -> Response:
    """Single-cue synthesis. Returns WAV by default, raw PCM16 mono when Accept: audio/pcm."""
    if not req.text:
        raise HTTPException(status_code=422, detail="'text' is required")
    text = expand_shortcodes(req.text)
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
            raise HTTPException(status_code=501, detail="caption runtime not configured") from err

        num_steps = int(req.num_steps) if req.num_steps and req.num_steps > 0 else 40
        cfg_text = (
            float(req.cfg_scale_text) if req.cfg_scale_text and req.cfg_scale_text > 0 else 3.0
        )
        cfg_cap = (
            float(req.cfg_scale_caption)
            if req.cfg_scale_caption and req.cfg_scale_caption > 0
            else 3.0
        )
        trunc = (
            req.truncation_factor if req.truncation_factor and req.truncation_factor > 0 else None
        )
        seed = req.seed if req.seed is not None and req.seed >= 0 else None

        sampling_req = _caption_sampling_req(
            cfg,
            text,
            req.caption,
            num_steps=num_steps,
            cfg_scale_text=cfg_text,
            cfg_scale_caption=cfg_cap,
            truncation_factor=trunc,
            seed=seed,
            seconds=req.seconds,
            min_seconds=req.min_seconds if req.min_seconds is not None else 0.5,
            max_seconds=req.max_seconds if req.max_seconds is not None else 30.0,
            duration_scale=req.duration_scale if req.duration_scale is not None else 1.0,
        )

        try:
            result = runtime.synthesize(
                sampling_req, log_fn=logger.debug if cfg.show_timings else None
            )
        except Exception as e:
            logger.exception("caption synthesis failed")
            raise HTTPException(status_code=500, detail=f"synthesis failed: {e}") from e

        audio = result.audio
        audio_np = (
            audio.squeeze(0).cpu().float().numpy()
            if audio.ndim == 2
            else audio.cpu().float().numpy()
        )
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
        duration_scale=float(params["duration_scale"]),
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
        result = runtime.synthesize(sampling_req, log_fn=logger.debug if cfg.show_timings else None)
    except Exception as e:
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}") from e

    audio = result.audio
    if audio.ndim == 2:
        audio_np = audio.squeeze(0).cpu().float().numpy()
    else:
        audio_np = audio.cpu().float().numpy()
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


def _synth_cue(  # noqa: C901, PLR0912, PLR0915
    registry: RuntimeRegistry,
    cfg: ServerConfig,
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
    cue_seconds = synth_defaults.seconds
    cue_min_seconds = synth_defaults.min_seconds
    cue_max_seconds = synth_defaults.max_seconds
    cue_duration_scale = synth_defaults.duration_scale
    if cue.options:
        if cue.options.num_steps is not None:
            cue_num_steps = int(cue.options.num_steps)
        if cue.options.cfg_scale_text is not None:
            cue_cfg_text = float(cue.options.cfg_scale_text)
        if cue.options.truncation_factor is not None:
            cue_trunc = cue.options.truncation_factor
        if cue.options.seed is not None:
            cue_seed = int(cue.options.seed)
        if cue.options.seconds is not None:
            cue_seconds = cue.options.seconds
        if cue.options.min_seconds is not None:
            cue_min_seconds = cue.options.min_seconds
        if cue.options.max_seconds is not None:
            cue_max_seconds = cue.options.max_seconds
        if cue.options.duration_scale is not None:
            cue_duration_scale = cue.options.duration_scale

    if isinstance(ref, CaptionSpeaker):
        runtime = registry.acquire_caption()
        sampling_req = _caption_sampling_req(
            cfg,
            cue.text,
            ref.caption,
            num_steps=cue_num_steps,
            cfg_scale_text=cue_cfg_text,
            cfg_scale_caption=3.0,
            truncation_factor=cue_trunc,
            seed=cue_seed if cue_seed is not None and cue_seed >= 0 else None,
            seconds=cue_seconds,
            min_seconds=float(cue_min_seconds) if cue_min_seconds is not None else 0.5,
            max_seconds=float(cue_max_seconds) if cue_max_seconds is not None else 30.0,
            duration_scale=float(cue_duration_scale) if cue_duration_scale is not None else 1.0,
        )
        result = runtime.synthesize(sampling_req, log_fn=logger.debug if cfg.show_timings else None)
        audio = result.audio
        audio_np = (
            audio.squeeze(0).cpu().float().numpy()
            if audio.ndim == 2
            else audio.cpu().float().numpy()
        )
        return _apply_fade(audio_np, int(result.sample_rate)), int(result.sample_rate)

    # LoRA speaker path
    try:
        runtime, spec = registry.acquire(ref.uuid)
    except KeyError as err:
        raise RuntimeError(f"unknown speaker UUID: {ref.uuid} (alias: {cue.speaker!r})") from err

    merged_defaults = dict(spec.defaults)
    for key in (
        "num_steps",
        "cfg_scale_text",
        "cfg_scale_speaker",
        "speaker_kv_scale",
        "truncation_factor",
        "seed",
        "seconds",
        "min_seconds",
        "max_seconds",
        "duration_scale",
    ):
        val = getattr(synth_defaults, key, None)
        if val is not None:
            merged_defaults[key] = val

    mock_req_fields: dict[str, Any] = {"speaker_id": ref.uuid, "text": cue.text}
    if cue.options:
        for key in (
            "seed",
            "num_steps",
            "cfg_scale_text",
            "cfg_scale_speaker",
            "speaker_kv_scale",
            "truncation_factor",
            "seconds",
            "min_seconds",
            "max_seconds",
            "duration_scale",
        ):
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
        # See _synth_single: the acquired adapter must survive synthesize().
        keep_adapter=True,
        text=cue.text,
        caption=None,
        ref_wav=None,
        ref_latent=None,
        no_ref=True,
        ref_normalize_db=-16.0,
        ref_ensure_max=True,
        num_candidates=1,
        decode_mode="sequential",
        seconds=params["seconds"],
        duration_scale=float(params["duration_scale"]),
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

    result = runtime.synthesize(sampling_req, log_fn=logger.debug if cfg.show_timings else None)
    audio = result.audio
    audio_np = (
        audio.squeeze(0).cpu().float().numpy() if audio.ndim == 2 else audio.cpu().float().numpy()
    )
    return _apply_fade(audio_np, int(result.sample_rate)), int(result.sample_rate)


def _stream_drama_pcm(
    registry: RuntimeRegistry,
    cfg: ServerConfig,
    script: VdsScript,
) -> Generator[bytes, None, None]:
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
                audio_np, sr = _synth_cue(registry, cfg, cue, script)
            except Exception:
                logger.exception("synthesis failed for cue (speaker=%s), skipping", cue.speaker)
                prev_was_speech = False
                continue

            if sample_rate is None:
                sample_rate = sr
            yield _to_pcm16(audio_np)
            prev_was_speech = True


def _validate_drama(registry: RuntimeRegistry, script: VdsScript) -> list[SpeechCue]:
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


def _get_sample_rate(registry: RuntimeRegistry) -> int:
    sr = registry.sample_rate
    return sr if sr is not None else 24000


def _render_drama_wav(
    registry: RuntimeRegistry,
    cfg: ServerConfig,
    script: VdsScript,
    speech_cues: list[SpeechCue],
) -> Response:
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
                segments.append(np.zeros(int(script.defaults.gap * sample_rate), dtype=np.float32))
            try:
                audio_np, sr = _synth_cue(registry, cfg, cue, script)
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


def _render_drama(
    registry: RuntimeRegistry,
    cfg: ServerConfig,
    script: VdsScript,
    request: Request,
) -> Response:
    speech_cues = _validate_drama(registry, script)

    if _wants_wav(request):
        return _render_drama_wav(registry, cfg, script, speech_cues)

    sample_rate = _get_sample_rate(registry)
    return StreamingResponse(
        _stream_drama_pcm(registry, cfg, script),
        media_type="audio/pcm",
        headers={
            "X-TTS-Sample-Rate": str(sample_rate),
            "X-TTS-Cue-Count": str(len(speech_cues)),
        },
    )


def _handle_synth(
    registry: RuntimeRegistry,
    cfg: ServerConfig,
    req: SynthRequest,
    request: Request,
) -> Response:
    if req.script is not None:
        try:
            script, warnings = parse_json(req.script.model_dump(exclude_none=True))
        except ParseError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        for w in warnings:
            logger.warning("VDS warning: %s", w)
        return _render_drama(registry, cfg, script, request)
    return _synth_single(registry, cfg, req, request)


def _handle_synth_vds(
    registry: RuntimeRegistry,
    cfg: ServerConfig,
    content: bytes,
    request: Request,
) -> Response:
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
    return _render_drama(registry, cfg, script, request)
