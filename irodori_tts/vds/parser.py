"""VDS text and JSON parsers.

Both formats produce the same ``VdsScript`` internal representation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .shortcodes import expand_shortcodes
from .types import (
    SYNTH_OPTION_KEYS,
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

# ---------------------------------------------------------------------------
# Error / warning types
# ---------------------------------------------------------------------------

ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
PAUSE_RE = re.compile(r"^\(pause\s+([\d.]+)\)$")
CAPTION_RE = re.compile(r'^caption\s+"((?:[^"\\]|\\.)*)"$')
CUE_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_-]*)"  # alias
    r"(?:\s*\[([^\]]*)\])?"  # optional [k=v, ...]
    r"\s*:\s*"  # colon separator
    r"(.*)$"  # text
)
DIRECTIVE_RE = re.compile(r"^@(\w+)\s*[: ]\s*(.*)$", re.DOTALL)
SPEAKER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*(.+)$")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,]+)")


@dataclass
class ParseError(Exception):
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


@dataclass
class ParseWarning:
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


# ---------------------------------------------------------------------------
# Text parser
# ---------------------------------------------------------------------------

_CUE_LENGTH_WARN_CHARS = 120


def _parse_number(raw: str) -> int | float:
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"not a number: {raw!r}") from exc


def _parse_synth_options(raw: str, lineno: int) -> SynthOptions:
    pairs: dict[str, int | float] = {}
    for m in KV_RE.finditer(raw):
        key, val_str = m.group(1), m.group(2).strip()
        if key not in SYNTH_OPTION_KEYS:
            raise ParseError(lineno, f"unknown option key: {key!r}")
        try:
            pairs[key] = _parse_number(val_str)
        except ValueError as exc:
            raise ParseError(lineno, f"option value for {key!r} is not a number: {val_str!r}") from exc
    return SynthOptions(**pairs)  # type: ignore[arg-type]


def _parse_defaults(raw: str, lineno: int) -> Defaults:
    gap: float = 1.0
    synth_pairs: dict[str, int | float] = {}
    for m in KV_RE.finditer(raw):
        key, val_str = m.group(1), m.group(2).strip()
        if key == "gap":
            try:
                gap = float(val_str)
            except ValueError as exc:
                raise ParseError(lineno, f"gap value is not a number: {val_str!r}") from exc
            if gap < 0:
                raise ParseError(lineno, f"gap must be non-negative, got {gap}")
            continue
        if key not in SYNTH_OPTION_KEYS:
            # §6.2: unknown keys in defaults are warnings, not errors
            continue
        try:
            synth_pairs[key] = _parse_number(val_str)
        except ValueError as exc:
            raise ParseError(lineno, f"defaults value for {key!r} is not a number: {val_str!r}") from exc
    return Defaults(gap=gap, synth=SynthOptions(**synth_pairs))  # type: ignore[arg-type]


def _parse_speaker_ref(raw: str, lineno: int) -> SpeakerRef:
    cm = CAPTION_RE.match(raw)
    if cm:
        caption = cm.group(1).replace('\\"', '"').replace("\\\\", "\\")
        return CaptionSpeaker(caption=caption)
    uuid_val = raw.strip()
    if UUID_RE.match(uuid_val):
        return LoraSpeaker(uuid=uuid_val)
    raise ParseError(lineno, f"speaker value must be a UUID or caption \"...\", got: {raw!r}")


def parse_text(source: str) -> tuple[VdsScript, list[ParseWarning]]:
    """Parse a VDS plain-text script into internal representation.

    Returns the parsed script and a list of warnings.
    Raises ``ParseError`` on fatal validation failures.
    """
    warnings: list[ParseWarning] = []
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    version: int | None = None
    title: str | None = None
    defaults = Defaults()
    speakers: dict[str, SpeakerRef] = {}
    cues: list[Cue] = []
    first_cue_seen = False
    prev_was_pause = False

    for lineno_0, raw_line in enumerate(lines):
        lineno = lineno_0 + 1
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        # --- directives ---
        if line.startswith("@"):
            dm = DIRECTIVE_RE.match(line)
            if not dm:
                raise ParseError(lineno, f"malformed directive: {line!r}")
            key, value = dm.group(1), dm.group(2).strip()

            if key == "version":
                try:
                    version = int(value)
                except ValueError as exc:
                    raise ParseError(lineno, f"version must be an integer, got {value!r}") from exc
                if version != 1:
                    raise ParseError(lineno, f"unsupported version: {version}")
                continue

            if key == "title":
                title = value
                continue

            if key == "scene":
                cues.append(SceneCue(name=value))
                prev_was_pause = False
                continue

            if key == "speaker":
                if first_cue_seen:
                    raise ParseError(lineno, "@speaker must appear before the first cue")
                sm = SPEAKER_RE.match(value)
                if not sm:
                    raise ParseError(lineno, f"malformed @speaker: {value!r}")
                alias, ref_raw = sm.group(1), sm.group(2).strip()
                if not ALIAS_RE.match(alias):
                    raise ParseError(lineno, f"invalid alias: {alias!r}")
                if alias in speakers:
                    raise ParseError(lineno, f"duplicate speaker alias: {alias!r}")
                speakers[alias] = _parse_speaker_ref(ref_raw, lineno)
                continue

            if key == "defaults":
                if first_cue_seen:
                    raise ParseError(lineno, "@defaults must appear before the first cue")
                defaults = _parse_defaults(value, lineno)
                continue

            raise ParseError(lineno, f"unknown directive: @{key}")

        # --- pause ---
        pm = PAUSE_RE.match(line)
        if pm:
            try:
                duration = float(pm.group(1))
            except ValueError as exc:
                raise ParseError(lineno, f"pause duration is not a number: {pm.group(1)!r}") from exc
            if duration <= 0:
                raise ParseError(lineno, f"pause duration must be positive, got {duration}")
            if prev_was_pause:
                warnings.append(ParseWarning(lineno, "consecutive pauses (values will be summed)"))
            cues.append(PauseCue(duration=duration))
            prev_was_pause = True
            continue

        # --- cue ---
        cm = CUE_RE.match(line)
        if cm:
            alias, opts_raw, text = cm.group(1), cm.group(2), cm.group(3).strip()
            if alias not in speakers:
                raise ParseError(lineno, f"undefined speaker alias: {alias!r}")
            if not text:
                raise ParseError(lineno, "cue text must not be empty")
            options = _parse_synth_options(opts_raw, lineno) if opts_raw else None
            text = expand_shortcodes(text)
            cues.append(SpeechCue(speaker=alias, text=text, options=options))
            first_cue_seen = True
            prev_was_pause = False

            if len(text) > _CUE_LENGTH_WARN_CHARS:
                warnings.append(
                    ParseWarning(lineno, f"cue text is {len(text)} chars (may exceed 30s)")
                )
            continue

        raise ParseError(lineno, f"unrecognised line: {line!r}")

    # --- post-parse validation ---
    if version is None:
        raise ParseError(0, "@version directive is required")

    used_aliases = {c.speaker for c in cues if isinstance(c, SpeechCue)}
    for alias in speakers:
        if alias not in used_aliases:
            warnings.append(ParseWarning(0, f"unused speaker alias: {alias!r}"))

    speech_count = sum(1 for c in cues if isinstance(c, SpeechCue))
    if speech_count > 100:
        warnings.append(
            ParseWarning(0, f"script has {speech_count} speech cues (recommended max: 100)")
        )

    return VdsScript(
        version=version,
        title=title,
        defaults=defaults,
        speakers=speakers,
        cues=cues,
    ), warnings


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------


def _json_speaker_ref(raw: dict, path: str) -> SpeakerRef:
    ref_type = raw.get("type")
    if ref_type == "lora":
        uuid = raw.get("uuid")
        if not isinstance(uuid, str) or not UUID_RE.match(uuid):
            raise ParseError(0, f"{path}: 'uuid' must be a valid UUID string")
        extra = set(raw.keys()) - {"type", "uuid"}
        if extra:
            raise ParseError(0, f"{path}: unexpected fields: {extra}")
        return LoraSpeaker(uuid=uuid)
    if ref_type == "caption":
        caption = raw.get("caption")
        if not isinstance(caption, str) or not caption:
            raise ParseError(0, f"{path}: 'caption' must be a non-empty string")
        extra = set(raw.keys()) - {"type", "caption"}
        if extra:
            raise ParseError(0, f"{path}: unexpected fields: {extra}")
        return CaptionSpeaker(caption=caption)
    raise ParseError(0, f"{path}: 'type' must be \"lora\" or \"caption\", got {ref_type!r}")


def _json_synth_options(raw: dict, path: str) -> SynthOptions:
    pairs: dict[str, int | float] = {}
    for k, v in raw.items():
        if k == "gap":
            continue
        if k not in SYNTH_OPTION_KEYS:
            raise ParseError(0, f"{path}: unknown option key: {k!r}")
        if not isinstance(v, (int, float)):
            raise ParseError(0, f"{path}.{k}: expected a number, got {type(v).__name__}")
        pairs[k] = v
    return SynthOptions(**pairs)  # type: ignore[arg-type]


def _json_cue(raw: dict, idx: int) -> Cue:
    path = f"cues[{idx}]"
    kind = raw.get("kind")
    if kind == "speech":
        speaker = raw.get("speaker")
        if not isinstance(speaker, str):
            raise ParseError(0, f"{path}: 'speaker' must be a string")
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ParseError(0, f"{path}: 'text' must be a non-empty string")
        opts = None
        if "options" in raw:
            if not isinstance(raw["options"], dict):
                raise ParseError(0, f"{path}.options: must be an object")
            opts = _json_synth_options(raw["options"], f"{path}.options")
        extra = set(raw.keys()) - {"kind", "speaker", "text", "options"}
        if extra:
            raise ParseError(0, f"{path}: unexpected fields: {extra}")
        return SpeechCue(speaker=speaker, text=expand_shortcodes(text.strip()), options=opts)
    if kind == "pause":
        duration = raw.get("duration")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ParseError(0, f"{path}: 'duration' must be a positive number")
        extra = set(raw.keys()) - {"kind", "duration"}
        if extra:
            raise ParseError(0, f"{path}: unexpected fields: {extra}")
        return PauseCue(duration=float(duration))
    if kind == "scene":
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ParseError(0, f"{path}: 'name' must be a non-empty string")
        extra = set(raw.keys()) - {"kind", "name"}
        if extra:
            raise ParseError(0, f"{path}: unexpected fields: {extra}")
        return SceneCue(name=name.strip())
    raise ParseError(0, f"{path}: 'kind' must be \"speech\", \"pause\", or \"scene\", got {kind!r}")


def parse_json(source: str | dict) -> tuple[VdsScript, list[ParseWarning]]:
    """Parse a VDS-JSON document into internal representation.

    *source* can be a JSON string or an already-parsed dict.
    Returns the parsed script and a list of warnings.
    Raises ``ParseError`` on validation failures.
    """
    if isinstance(source, str):
        try:
            data = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ParseError(0, f"invalid JSON: {exc}") from exc
    else:
        data = source

    if not isinstance(data, dict):
        raise ParseError(0, "top-level value must be an object")

    # version (required)
    version = data.get("version")
    if version is None:
        raise ParseError(0, "'version' is required")
    if version != 1:
        raise ParseError(0, f"unsupported version: {version}")

    # title (optional)
    title = data.get("title")
    if title is not None and not isinstance(title, str):
        raise ParseError(0, "'title' must be a string")

    # defaults (optional)
    defaults = Defaults()
    if "defaults" in data:
        raw_defaults = data["defaults"]
        if not isinstance(raw_defaults, dict):
            raise ParseError(0, "'defaults' must be an object")
        gap = raw_defaults.get("gap", 1.0)
        if not isinstance(gap, (int, float)) or gap < 0:
            raise ParseError(0, "'defaults.gap' must be a non-negative number")
        synth = _json_synth_options(raw_defaults, "defaults")
        defaults = Defaults(gap=float(gap), synth=synth)

    # speakers (required)
    raw_speakers = data.get("speakers")
    if not isinstance(raw_speakers, dict):
        raise ParseError(0, "'speakers' must be an object")
    speakers: dict[str, SpeakerRef] = {}
    for alias, ref_raw in raw_speakers.items():
        if not ALIAS_RE.match(alias):
            raise ParseError(0, f"invalid speaker alias: {alias!r}")
        if not isinstance(ref_raw, dict):
            raise ParseError(0, f"speakers.{alias}: must be an object")
        speakers[alias] = _json_speaker_ref(ref_raw, f"speakers.{alias}")

    # cues (required)
    raw_cues = data.get("cues")
    if not isinstance(raw_cues, list):
        raise ParseError(0, "'cues' must be an array")
    cues: list[Cue] = []
    for idx, raw_cue in enumerate(raw_cues):
        if not isinstance(raw_cue, dict):
            raise ParseError(0, f"cues[{idx}]: must be an object")
        cues.append(_json_cue(raw_cue, idx))

    # --- post-parse validation ---
    warnings: list[ParseWarning] = []

    for idx, cue in enumerate(cues):
        if isinstance(cue, SpeechCue) and cue.speaker not in speakers:
            raise ParseError(0, f"cues[{idx}]: undefined speaker alias: {cue.speaker!r}")

    used_aliases = {c.speaker for c in cues if isinstance(c, SpeechCue)}
    for alias in speakers:
        if alias not in used_aliases:
            warnings.append(ParseWarning(0, f"unused speaker alias: {alias!r}"))

    speech_count = sum(1 for c in cues if isinstance(c, SpeechCue))
    if speech_count > 100:
        warnings.append(
            ParseWarning(0, f"script has {speech_count} speech cues (recommended max: 100)")
        )

    prev_pause = False
    for idx, cue in enumerate(cues):
        if isinstance(cue, PauseCue):
            if prev_pause:
                warnings.append(
                    ParseWarning(0, f"cues[{idx}]: consecutive pauses (values will be summed)")
                )
            prev_pause = True
        else:
            prev_pause = False

    for idx, cue in enumerate(cues):
        if isinstance(cue, SpeechCue) and len(cue.text) > _CUE_LENGTH_WARN_CHARS:
            warnings.append(
                ParseWarning(0, f"cues[{idx}]: cue text is {len(cue.text)} chars (may exceed 30s)")
            )

    return VdsScript(
        version=version,
        title=title,
        defaults=defaults,
        speakers=speakers,
        cues=cues,
    ), warnings
