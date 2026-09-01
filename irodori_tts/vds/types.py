"""Internal representation for VDS scripts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SynthOptions:
    seed: int | None = None
    num_steps: int | None = None
    cfg_scale_text: float | None = None
    cfg_scale_speaker: float | None = None
    speaker_kv_scale: float | None = None
    truncation_factor: float | None = None
    seconds: float | None = None
    min_seconds: float | None = None
    max_seconds: float | None = None
    duration_scale: float | None = None


SYNTH_OPTION_KEYS = frozenset(f.name for f in SynthOptions.__dataclass_fields__.values())


@dataclass(frozen=True)
class Defaults:
    gap: float = 1.0
    synth: SynthOptions = field(default_factory=SynthOptions)


@dataclass(frozen=True)
class LoraSpeaker:
    uuid: str


@dataclass(frozen=True)
class CaptionSpeaker:
    caption: str


SpeakerRef = LoraSpeaker | CaptionSpeaker


@dataclass(frozen=True)
class SpeechCue:
    speaker: str
    text: str
    options: SynthOptions | None = None


@dataclass(frozen=True)
class PauseCue:
    duration: float


@dataclass(frozen=True)
class SceneCue:
    name: str


Cue = SpeechCue | PauseCue | SceneCue


@dataclass
class VdsScript:
    version: int
    title: str | None
    defaults: Defaults
    speakers: dict[str, SpeakerRef]
    cues: list[Cue]

    def to_dict(self) -> dict:
        d: dict = {"version": self.version}
        if self.title is not None:
            d["title"] = self.title
        defaults_d = _defaults_to_dict(self.defaults)
        if defaults_d:
            d["defaults"] = defaults_d
        d["speakers"] = {alias: _speaker_ref_to_dict(ref) for alias, ref in self.speakers.items()}
        d["cues"] = [_cue_to_dict(cue) for cue in self.cues]
        return d


def _synth_options_to_dict(options: SynthOptions) -> dict:
    return {key: val for key in SYNTH_OPTION_KEYS if (val := getattr(options, key)) is not None}


def _defaults_to_dict(defaults: Defaults) -> dict:
    defaults_d: dict = {}
    if defaults.gap != 1.0:
        defaults_d["gap"] = defaults.gap
    defaults_d.update(_synth_options_to_dict(defaults.synth))
    return defaults_d


def _speaker_ref_to_dict(ref: SpeakerRef) -> dict:
    if isinstance(ref, LoraSpeaker):
        return {"type": "lora", "uuid": ref.uuid}
    return {"type": "caption", "caption": ref.caption}


def _cue_to_dict(cue: Cue) -> dict:
    if isinstance(cue, SpeechCue):
        cue_d: dict = {"kind": "speech", "speaker": cue.speaker, "text": cue.text}
        if cue.options is not None:
            opts_d = _synth_options_to_dict(cue.options)
            if opts_d:
                cue_d["options"] = opts_d
        return cue_d
    if isinstance(cue, PauseCue):
        return {"kind": "pause", "duration": cue.duration}
    return {"kind": "scene", "name": cue.name}
