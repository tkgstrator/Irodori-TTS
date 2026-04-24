"""Voice Drama Script (VDS) parser and types."""

from .parser import ParseError, ParseWarning, parse_json, parse_text
from .shortcodes import SHORTCODE_MAP, expand_shortcodes
from .types import (
    CaptionSpeaker,
    Cue,
    Defaults,
    LoraSpeaker,
    PauseCue,
    SceneCue,
    SpeakerRef,
    SpeechCue,
    SynthOptions,
    VdsScript,
)

__all__ = [
    "CaptionSpeaker",
    "Cue",
    "Defaults",
    "LoraSpeaker",
    "ParseError",
    "ParseWarning",
    "PauseCue",
    "SHORTCODE_MAP",
    "SceneCue",
    "SpeakerRef",
    "SpeechCue",
    "SynthOptions",
    "VdsScript",
    "expand_shortcodes",
    "parse_json",
    "parse_text",
]
