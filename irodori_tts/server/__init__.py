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
from irodori_tts.server.encoding import MEDIA_TYPES, encode, sse_events, to_pcm16
from irodori_tts.server.errors import ApiError, install_error_handlers
from irodori_tts.server.registry import RuntimeRegistry
from irodori_tts.server.schemas import (
    _POSITIVE_ONLY,
    SpeechRequest,
    VoiceRef,
    _merge_defaults,
)
from irodori_tts.server.shortcodes import SHORTCODE_MAP, expand_shortcodes
from irodori_tts.server.synthesis import Synthesized, synthesize

__all__ = [
    "MEDIA_TYPES",
    "SHORTCODE_MAP",
    "ApiError",
    "RuntimeRegistry",
    "ServerConfig",
    "SpeakerSpec",
    "SpeechRequest",
    "Synthesized",
    "VoiceRef",
    "encode",
    "expand_shortcodes",
    "install_error_handlers",
    "load_config",
    "resolve_base_checkpoint",
    "sse_events",
    "synthesize",
    "to_pcm16",
]
