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
        defaults_d: dict = {}
        if self.defaults.gap != 1.0:
            defaults_d["gap"] = self.defaults.gap
        for key in SYNTH_OPTION_KEYS:
            val = getattr(self.defaults.synth, key)
            if val is not None:
                defaults_d[key] = val
        if defaults_d:
            d["defaults"] = defaults_d
        speakers_d: dict = {}
        for alias, ref in self.speakers.items():
            if isinstance(ref, LoraSpeaker):
                speakers_d[alias] = {"type": "lora", "uuid": ref.uuid}
            else:
                speakers_d[alias] = {"type": "caption", "caption": ref.caption}
        d["speakers"] = speakers_d
        cues_list: list[dict] = []
        for cue in self.cues:
            if isinstance(cue, SpeechCue):
                cue_d: dict = {"kind": "speech", "speaker": cue.speaker, "text": cue.text}
                if cue.options is not None:
                    opts_d: dict = {}
                    for key in SYNTH_OPTION_KEYS:
                        val = getattr(cue.options, key)
                        if val is not None:
                            opts_d[key] = val
                    if opts_d:
                        cue_d["options"] = opts_d
                cues_list.append(cue_d)
            elif isinstance(cue, PauseCue):
                cues_list.append({"kind": "pause", "duration": cue.duration})
            elif isinstance(cue, SceneCue):
                cues_list.append({"kind": "scene", "name": cue.name})
        d["cues"] = cues_list
        return d
