"""Server-side building blocks for the Irodori-TTS FastAPI app."""

from irodori_tts.server.audio import _FADE_MS, _apply_fade
from irodori_tts.server.config import (
    _LORA_UUID_NAMESPACE,
    ServerConfig,
    SpeakerSpec,
    _discover_lora_dir,
    _resolve_checkpoint,
    _resolve_lora_display_name,
    load_config,
    resolve_base_checkpoint,
)
from irodori_tts.server.registry import RuntimeRegistry
from irodori_tts.server.schemas import (
    _POSITIVE_ONLY,
    SynthRequest,
    VdsCaptionSpeaker,
    VdsCue,
    VdsDefaults,
    VdsLoraSpeaker,
    VdsPauseCue,
    VdsSceneCue,
    VdsScriptBody,
    VdsSpeakerRef,
    VdsSpeechCue,
    VdsSynthOptions,
    _merge_defaults,
)

__all__ = [
    "RuntimeRegistry",
    "ServerConfig",
    "SpeakerSpec",
    "SynthRequest",
    "VdsCaptionSpeaker",
    "VdsCue",
    "VdsDefaults",
    "VdsLoraSpeaker",
    "VdsPauseCue",
    "VdsSceneCue",
    "VdsScriptBody",
    "VdsSpeakerRef",
    "VdsSpeechCue",
    "VdsSynthOptions",
    "load_config",
    "resolve_base_checkpoint",
]
