"""Tests for the VDS parser (text and JSON formats)."""

from __future__ import annotations

import pytest

from irodori_tts.vds import (
    CaptionSpeaker,
    LoraSpeaker,
    ParseError,
    PauseCue,
    SceneCue,
    SpeechCue,
    SynthOptions,
    expand_shortcodes,
    parse_json,
    parse_text,
)

UUID_A = "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb"
UUID_B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

MINIMAL_VDS = f"""\
@version: 1
@speaker alice = {UUID_A}

alice: こんにちは
"""

FULL_VDS = f"""\
@version: 1
@title: テスト台本
@speaker alice = {UUID_A}
@speaker bob = caption "落ち着いた男性の声"
@defaults num_steps=50, gap=0.3

@scene: 第1幕
alice: おはよう。
bob: おはようございます。

(pause 1.0)

alice [seed=42, cfg_scale_text=3.5]: 今日はいい天気ですね。
"""


# ===================================================================
# Text parser – happy path
# ===================================================================


class TestParseTextHappyPath:
    def test_minimal(self):
        script, warnings = parse_text(MINIMAL_VDS)
        assert script.version == 1
        assert script.title is None
        assert script.defaults.gap == 1.0
        assert script.speakers == {"alice": LoraSpeaker(uuid=UUID_A)}
        assert len(script.cues) == 1
        assert script.cues[0] == SpeechCue(speaker="alice", text="こんにちは")
        assert warnings == []

    def test_full(self):
        script, warnings = parse_text(FULL_VDS)
        assert script.version == 1
        assert script.title == "テスト台本"
        assert script.defaults.gap == 0.3
        assert script.defaults.synth.num_steps == 50
        assert isinstance(script.speakers["alice"], LoraSpeaker)
        assert isinstance(script.speakers["bob"], CaptionSpeaker)
        assert script.speakers["bob"].caption == "落ち着いた男性の声"

        assert isinstance(script.cues[0], SceneCue)
        assert script.cues[0].name == "第1幕"
        assert isinstance(script.cues[1], SpeechCue)
        assert script.cues[1].speaker == "alice"
        assert isinstance(script.cues[2], SpeechCue)
        assert script.cues[2].speaker == "bob"
        assert isinstance(script.cues[3], PauseCue)
        assert script.cues[3].duration == 1.0
        assert isinstance(script.cues[4], SpeechCue)
        assert script.cues[4].options == SynthOptions(seed=42, cfg_scale_text=3.5)
        assert warnings == []

    def test_crlf(self):
        source = MINIMAL_VDS.replace("\n", "\r\n")
        script, _ = parse_text(source)
        assert len(script.cues) == 1

    def test_comments_and_blanks_ignored(self):
        source = f"""\
@version: 1
# this is a comment
@speaker a = {UUID_A}

# another comment

a: hello
"""
        script, _ = parse_text(source)
        assert len(script.cues) == 1

    def test_gap_zero(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}
@defaults gap=0

a: hello
"""
        script, _ = parse_text(source)
        assert script.defaults.gap == 0.0

    def test_caption_with_escapes(self):
        source = """\
@version: 1
@speaker a = caption "声に\\"特徴\\"がある"

a: test
"""
        script, _ = parse_text(source)
        assert isinstance(script.speakers["a"], CaptionSpeaker)
        assert script.speakers["a"].caption == '声に"特徴"がある'

    def test_multiple_scenes(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

@scene: シーン1
a: hello
@scene: シーン2
a: world
"""
        script, _ = parse_text(source)
        scenes = [c for c in script.cues if isinstance(c, SceneCue)]
        assert len(scenes) == 2
        assert scenes[0].name == "シーン1"
        assert scenes[1].name == "シーン2"

    def test_cue_options_all_keys(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a [seed=1, num_steps=30, cfg_scale_text=2.5, cfg_scale_speaker=4.0, speaker_kv_scale=1.2, truncation_factor=0.8]: test
"""
        script, _ = parse_text(source)
        opts = script.cues[0].options
        assert opts.seed == 1
        assert opts.num_steps == 30
        assert opts.cfg_scale_text == 2.5
        assert opts.cfg_scale_speaker == 4.0
        assert opts.speaker_kv_scale == 1.2
        assert opts.truncation_factor == 0.8

    def test_negative_seed(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a [seed=-1]: test
"""
        script, _ = parse_text(source)
        assert script.cues[0].options.seed == -1

    def test_alias_with_hyphen_and_underscore(self):
        source = f"""\
@version: 1
@speaker my_speaker-1 = {UUID_A}

my_speaker-1: hello
"""
        script, _ = parse_text(source)
        assert "my_speaker-1" in script.speakers

    def test_colon_spacing_flexible(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a:hello
"""
        script, _ = parse_text(source)
        assert script.cues[0].text == "hello"

    def test_hash_in_text_not_comment(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a: C# is a programming language
"""
        script, _ = parse_text(source)
        assert script.cues[0].text == "C# is a programming language"


# ===================================================================
# Text parser – warnings
# ===================================================================


class TestParseTextWarnings:
    def test_consecutive_pause_warning(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a: hello
(pause 1.0)
(pause 0.5)
a: world
"""
        _, warnings = parse_text(source)
        assert len(warnings) == 1
        assert "consecutive" in warnings[0].message

    def test_unused_speaker_warning(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}
@speaker b = {UUID_B}

a: hello
"""
        _, warnings = parse_text(source)
        assert len(warnings) == 1
        assert "unused" in warnings[0].message
        assert "'b'" in warnings[0].message

    def test_long_text_warning(self):
        long_text = "あ" * 150
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a: {long_text}
"""
        _, warnings = parse_text(source)
        assert any("chars" in w.message for w in warnings)

    def test_many_cues_warning(self):
        cue_lines = "\n".join(f"a: line {i}" for i in range(101))
        source = f"""\
@version: 1
@speaker a = {UUID_A}

{cue_lines}
"""
        _, warnings = parse_text(source)
        assert any("101" in w.message and "100" in w.message for w in warnings)


# ===================================================================
# Text parser – errors
# ===================================================================


class TestParseTextErrors:
    def test_missing_version(self):
        source = f"""\
@speaker a = {UUID_A}

a: hello
"""
        with pytest.raises(ParseError, match="@version.*required"):
            parse_text(source)

    def test_unsupported_version(self):
        with pytest.raises(ParseError, match="unsupported version"):
            parse_text("@version: 2\n")

    def test_undefined_speaker(self):
        source = """\
@version: 1

alice: hello
"""
        with pytest.raises(ParseError, match="undefined speaker"):
            parse_text(source)

    def test_empty_text(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a:
"""
        with pytest.raises(ParseError, match="text must not be empty"):
            parse_text(source)

    def test_empty_text_whitespace_only(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a:   \t
"""
        with pytest.raises(ParseError, match="text must not be empty"):
            parse_text(source)

    def test_unknown_directive(self):
        with pytest.raises(ParseError, match="unknown directive.*@foo"):
            parse_text("@version: 1\n@foo: bar\n")

    def test_unknown_option_key(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a [badkey=1]: hello
"""
        with pytest.raises(ParseError, match="unknown option key.*badkey"):
            parse_text(source)

    def test_option_not_a_number(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a [seed=abc]: hello
"""
        with pytest.raises(ParseError, match="not a number"):
            parse_text(source)

    def test_pause_non_positive(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

(pause 0)
a: hello
"""
        with pytest.raises(ParseError, match="must be positive"):
            parse_text(source)

    def test_pause_negative(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

(pause -1)
a: hello
"""
        with pytest.raises(ParseError, match="unrecognised line"):
            parse_text(source)

    def test_pause_not_a_number(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

(pause abc)
a: hello
"""
        with pytest.raises(ParseError, match="unrecognised line"):
            parse_text(source)

    def test_speaker_after_cue(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a: hello
@speaker b = {UUID_B}
"""
        with pytest.raises(ParseError, match="@speaker must appear before"):
            parse_text(source)

    def test_defaults_after_cue(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a: hello
@defaults num_steps=50
"""
        with pytest.raises(ParseError, match="@defaults must appear before"):
            parse_text(source)

    def test_duplicate_alias(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}
@speaker a = {UUID_B}

a: hello
"""
        with pytest.raises(ParseError, match="duplicate speaker"):
            parse_text(source)

    def test_invalid_speaker_ref(self):
        source = """\
@version: 1
@speaker a = not-a-uuid

a: hello
"""
        with pytest.raises(ParseError, match="must be a UUID or caption"):
            parse_text(source)

    def test_negative_gap(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}
@defaults gap=-1

a: hello
"""
        with pytest.raises(ParseError, match="gap must be non-negative"):
            parse_text(source)


# ===================================================================
# JSON parser – happy path
# ===================================================================

MINIMAL_JSON = {
    "version": 1,
    "speakers": {
        "alice": {"type": "lora", "uuid": UUID_A},
    },
    "cues": [
        {"kind": "speech", "speaker": "alice", "text": "こんにちは"},
    ],
}

FULL_JSON = {
    "version": 1,
    "title": "テスト台本",
    "defaults": {"num_steps": 50, "gap": 0.3},
    "speakers": {
        "alice": {"type": "lora", "uuid": UUID_A},
        "bob": {"type": "caption", "caption": "落ち着いた男性の声"},
    },
    "cues": [
        {"kind": "scene", "name": "第1幕"},
        {"kind": "speech", "speaker": "alice", "text": "おはよう。"},
        {"kind": "speech", "speaker": "bob", "text": "おはようございます。"},
        {"kind": "pause", "duration": 1.0},
        {
            "kind": "speech",
            "speaker": "alice",
            "text": "今日はいい天気ですね。",
            "options": {"seed": 42, "cfg_scale_text": 3.5},
        },
    ],
}


class TestParseJsonHappyPath:
    def test_minimal(self):
        script, warnings = parse_json(MINIMAL_JSON)
        assert script.version == 1
        assert script.title is None
        assert script.defaults.gap == 1.0
        assert len(script.cues) == 1
        assert script.cues[0] == SpeechCue(speaker="alice", text="こんにちは")
        assert warnings == []

    def test_full(self):
        script, warnings = parse_json(FULL_JSON)
        assert script.version == 1
        assert script.title == "テスト台本"
        assert script.defaults.gap == 0.3
        assert script.defaults.synth.num_steps == 50
        assert isinstance(script.speakers["bob"], CaptionSpeaker)
        assert isinstance(script.cues[0], SceneCue)
        assert isinstance(script.cues[3], PauseCue)
        assert script.cues[4].options.seed == 42
        assert warnings == []

    def test_from_json_string(self):
        import json

        source = json.dumps(MINIMAL_JSON, ensure_ascii=False)
        script, _ = parse_json(source)
        assert len(script.cues) == 1

    def test_integer_pause_duration(self):
        data = {
            "version": 1,
            "speakers": {"a": {"type": "lora", "uuid": UUID_A}},
            "cues": [
                {"kind": "speech", "speaker": "a", "text": "hi"},
                {"kind": "pause", "duration": 2},
            ],
        }
        script, _ = parse_json(data)
        assert script.cues[1].duration == 2.0


# ===================================================================
# JSON parser – warnings
# ===================================================================


class TestParseJsonWarnings:
    def test_unused_speaker(self):
        data = {
            "version": 1,
            "speakers": {
                "a": {"type": "lora", "uuid": UUID_A},
                "b": {"type": "lora", "uuid": UUID_B},
            },
            "cues": [{"kind": "speech", "speaker": "a", "text": "hello"}],
        }
        _, warnings = parse_json(data)
        assert any("unused" in w.message and "'b'" in w.message for w in warnings)

    def test_consecutive_pause(self):
        data = {
            "version": 1,
            "speakers": {"a": {"type": "lora", "uuid": UUID_A}},
            "cues": [
                {"kind": "speech", "speaker": "a", "text": "hello"},
                {"kind": "pause", "duration": 1.0},
                {"kind": "pause", "duration": 0.5},
            ],
        }
        _, warnings = parse_json(data)
        assert any("consecutive" in w.message for w in warnings)


# ===================================================================
# JSON parser – errors
# ===================================================================


class TestParseJsonErrors:
    def test_missing_version(self):
        with pytest.raises(ParseError, match="version.*required"):
            parse_json({"speakers": {}, "cues": []})

    def test_unsupported_version(self):
        with pytest.raises(ParseError, match="unsupported version"):
            parse_json({"version": 2, "speakers": {}, "cues": []})

    def test_undefined_speaker_in_cue(self):
        data = {
            "version": 1,
            "speakers": {},
            "cues": [{"kind": "speech", "speaker": "unknown", "text": "hi"}],
        }
        with pytest.raises(ParseError, match="undefined speaker"):
            parse_json(data)

    def test_empty_text(self):
        data = {
            "version": 1,
            "speakers": {"a": {"type": "lora", "uuid": UUID_A}},
            "cues": [{"kind": "speech", "speaker": "a", "text": ""}],
        }
        with pytest.raises(ParseError, match="non-empty string"):
            parse_json(data)

    def test_invalid_speaker_type(self):
        data = {
            "version": 1,
            "speakers": {"a": {"type": "bad", "uuid": UUID_A}},
            "cues": [],
        }
        with pytest.raises(ParseError, match="\"lora\" or \"caption\""):
            parse_json(data)

    def test_missing_speaker_type(self):
        data = {
            "version": 1,
            "speakers": {"a": {"uuid": UUID_A}},
            "cues": [],
        }
        with pytest.raises(ParseError, match="\"lora\" or \"caption\""):
            parse_json(data)

    def test_extra_fields_in_cue(self):
        data = {
            "version": 1,
            "speakers": {"a": {"type": "lora", "uuid": UUID_A}},
            "cues": [{"kind": "speech", "speaker": "a", "text": "hi", "extra": 1}],
        }
        with pytest.raises(ParseError, match="unexpected fields"):
            parse_json(data)

    def test_extra_fields_in_speaker(self):
        data = {
            "version": 1,
            "speakers": {"a": {"type": "lora", "uuid": UUID_A, "name": "Alice"}},
            "cues": [],
        }
        with pytest.raises(ParseError, match="unexpected fields"):
            parse_json(data)

    def test_unknown_cue_kind(self):
        data = {
            "version": 1,
            "speakers": {},
            "cues": [{"kind": "bgm", "file": "music.mp3"}],
        }
        with pytest.raises(ParseError, match="\"speech\", \"pause\", or \"scene\""):
            parse_json(data)

    def test_pause_zero_duration(self):
        data = {
            "version": 1,
            "speakers": {"a": {"type": "lora", "uuid": UUID_A}},
            "cues": [{"kind": "pause", "duration": 0}],
        }
        with pytest.raises(ParseError, match="positive number"):
            parse_json(data)

    def test_invalid_json_string(self):
        with pytest.raises(ParseError, match="invalid JSON"):
            parse_json("{bad json")

    def test_invalid_alias(self):
        data = {
            "version": 1,
            "speakers": {"123bad": {"type": "lora", "uuid": UUID_A}},
            "cues": [],
        }
        with pytest.raises(ParseError, match="invalid speaker alias"):
            parse_json(data)

    def test_unknown_option_key_in_cue(self):
        data = {
            "version": 1,
            "speakers": {"a": {"type": "lora", "uuid": UUID_A}},
            "cues": [
                {
                    "kind": "speech",
                    "speaker": "a",
                    "text": "hi",
                    "options": {"badkey": 1},
                }
            ],
        }
        with pytest.raises(ParseError, match="unknown option key"):
            parse_json(data)


# ===================================================================
# Text ↔ JSON equivalence
# ===================================================================


class TestEquivalence:
    def test_full_example_equivalence(self):
        text_script, _ = parse_text(FULL_VDS)
        json_script, _ = parse_json(FULL_JSON)

        assert text_script.version == json_script.version
        assert text_script.title == json_script.title
        assert text_script.defaults == json_script.defaults
        assert text_script.speakers == json_script.speakers
        assert text_script.cues == json_script.cues


class TestToDict:
    def test_roundtrip_full(self):
        text_script, _ = parse_text(FULL_VDS)
        d = text_script.to_dict()
        roundtrip_script, _ = parse_json(d)

        assert roundtrip_script.version == text_script.version
        assert roundtrip_script.title == text_script.title
        assert roundtrip_script.defaults == text_script.defaults
        assert roundtrip_script.speakers == text_script.speakers
        assert roundtrip_script.cues == text_script.cues

    def test_roundtrip_minimal(self):
        text_script, _ = parse_text(MINIMAL_VDS)
        d = text_script.to_dict()
        roundtrip_script, _ = parse_json(d)

        assert roundtrip_script.speakers == text_script.speakers
        assert roundtrip_script.cues == text_script.cues

    def test_speaker_order_preserved(self):
        source = f"""\
@version: 1
@speaker z_last = {UUID_A}
@speaker a_first = {UUID_B}

z_last: hello
a_first: world
"""
        script, _ = parse_text(source)
        d = script.to_dict()
        aliases = list(d["speakers"].keys())
        assert aliases == ["z_last", "a_first"]

    def test_caption_speaker_roundtrip(self):
        source = """\
@version: 1
@speaker narrator = caption "落ち着いた男性の声"

narrator: test
"""
        script, _ = parse_text(source)
        d = script.to_dict()
        assert d["speakers"]["narrator"] == {"type": "caption", "caption": "落ち着いた男性の声"}
        roundtrip, _ = parse_json(d)
        assert roundtrip.speakers == script.speakers

    def test_to_dict_matches_handwritten_json(self):
        text_script, _ = parse_text(FULL_VDS)
        d = text_script.to_dict()
        assert d == FULL_JSON


# ===================================================================
# Shortcode expansion
# ===================================================================


class TestExpandShortcodes:
    def test_single_shortcode(self):
        assert expand_shortcodes("{whisper}ねえ") == "👂ねえ"

    def test_multiple_shortcodes(self):
        result = expand_shortcodes("{whisper}ねえ、聞こえる？{gasp}え、何それ？")
        assert result == "👂ねえ、聞こえる？😮え、何それ？"

    def test_unknown_shortcode_left_as_is(self):
        assert expand_shortcodes("{unknown}テスト") == "{unknown}テスト"

    def test_no_shortcodes(self):
        assert expand_shortcodes("普通のテキスト") == "普通のテキスト"

    def test_all_39_shortcodes_exist(self):
        from irodori_tts.vds.shortcodes import SHORTCODE_MAP

        assert len(SHORTCODE_MAP) == 39


class TestShortcodeInTextParser:
    def test_cue_text_expanded(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a: {{whisper}}ねえ、聞こえる？
"""
        script, _ = parse_text(source)
        assert script.cues[0].text == "👂ねえ、聞こえる？"

    def test_multiple_shortcodes_in_cue(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a: {{cheerful}}おはよう！{{gasp}}あ、忘れてた！
"""
        script, _ = parse_text(source)
        assert script.cues[0].text == "😊おはよう！😮あ、忘れてた！"

    def test_unknown_shortcode_preserved(self):
        source = f"""\
@version: 1
@speaker a = {UUID_A}

a: {{notreal}}テスト
"""
        script, _ = parse_text(source)
        assert script.cues[0].text == "{notreal}テスト"


class TestShortcodeInJsonParser:
    def test_cue_text_expanded(self):
        data = {
            "version": 1,
            "speakers": {"a": {"type": "lora", "uuid": UUID_A}},
            "cues": [
                {"kind": "speech", "speaker": "a", "text": "{whisper}ねえ"},
            ],
        }
        script, _ = parse_json(data)
        assert script.cues[0].text == "👂ねえ"

    def test_multiple_shortcodes_in_cue(self):
        data = {
            "version": 1,
            "speakers": {"a": {"type": "lora", "uuid": UUID_A}},
            "cues": [
                {"kind": "speech", "speaker": "a", "text": "{joyful}わーい！{cry}うう…"},
            ],
        }
        script, _ = parse_json(data)
        assert script.cues[0].text == "😆わーい！😭うう…"
