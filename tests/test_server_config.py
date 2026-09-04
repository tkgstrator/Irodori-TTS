"""Characterization tests for server.py.

These pin down the behavior of the config loader, the request schemas, the
audio fade helper and the four HTTP routes exactly as they behave today, so a
later refactor into ``irodori_tts/server/`` can be proven behavior-preserving.
No GPU, no checkpoints and no network: every test either stays below the model
boundary or builds the app with ``eager_load=False``.
"""

from __future__ import annotations

import json
import uuid as uuid_lib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError
from safetensors.torch import save_file

from irodori_tts.server import registry as registry_module
from irodori_tts.server.audio import _apply_fade
from irodori_tts.server.config import (
    _LORA_UUID_NAMESPACE,
    SpeakerSpec,
    _discover_lora_dir,
    _resolve_checkpoint,
    _resolve_lora_display_name,
    load_config,
    resolve_base_checkpoint,
)
from irodori_tts.server.registry import RuntimeRegistry
from irodori_tts.server.schemas import (
    SynthRequest,
    VdsDefaults,
    VdsScriptBody,
    _merge_defaults,
)
from irodori_tts.server.synthesis import _synth_single
from server import build_app

UUID_A = "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb"
UUID_B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def write_config(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def write_lora(path: Path, metadata: dict[str, str] | None = None) -> Path:
    meta = {"adapter_config": "{}"}
    if metadata:
        meta.update(metadata)
    save_file({"lora_A.weight": torch.zeros(2, 2)}, str(path), metadata=meta)
    return path


def speaker_entry(**overrides: Any) -> dict[str, Any]:
    entry = {"uuid": UUID_A, "name": "Alice", "adapter": "/models/alice.safetensors"}
    entry.update(overrides)
    return entry


class FakeRuntime:
    """Stand-in for ``InferenceRuntime``: only what the registry and caption path touch."""

    def __init__(self, checkpoint: str, *, use_caption_condition: bool) -> None:
        self.checkpoint = checkpoint
        self.model_cfg = SimpleNamespace(use_caption_condition=use_caption_condition)
        self.codec = SimpleNamespace(sample_rate=48000)

    def set_active_adapter(self, name: str) -> None:
        self.active_adapter = name

    def synthesize(self, _req: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(audio=torch.zeros(1, 4800), sample_rate=48000, used_seed=7)


def install_fake_runtimes(
    monkeypatch: pytest.MonkeyPatch, *, base_caption: bool
) -> dict[str, list[str]]:
    """Replace both runtime loaders with fakes and record the checkpoints they were asked for."""
    calls: dict[str, list[str]] = {"base": [], "caption": []}

    def from_base_with_adapters(*, key: Any, adapters: Any, default_adapter: Any) -> FakeRuntime:
        del adapters, default_adapter
        calls["base"].append(key.checkpoint)
        return FakeRuntime(key.checkpoint, use_caption_condition=base_caption)

    def from_key(key: Any) -> FakeRuntime:
        calls["caption"].append(key.checkpoint)
        return FakeRuntime(key.checkpoint, use_caption_condition=True)

    monkeypatch.setattr(
        registry_module.InferenceRuntime, "from_base_with_adapters", from_base_with_adapters
    )
    monkeypatch.setattr(registry_module.InferenceRuntime, "from_key", from_key)
    return calls


def caption_test_config(tmp_path: Path, **extra: Any) -> Path:
    """Config with one discoverable LoRA and an existing (dummy) base checkpoint file."""
    ckpt = tmp_path / "base.safetensors"
    ckpt.write_text("x", encoding="utf-8")
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    write_lora(lora_dir / "alice.safetensors", {"name": "Alice", "uuid": UUID_A})
    data: dict[str, Any] = {"base_checkpoint": str(ckpt), "lora_dir": str(lora_dir)}
    data.update(extra)
    return write_config(tmp_path / "c.yaml", data)


# ===================================================================
# load_config
# ===================================================================


class TestLoadConfigDefaults:
    def test_empty_mapping_yields_all_defaults(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {}))
        assert cfg.base_checkpoint is None
        assert cfg.base_hf_repo is None
        assert cfg.base_hf_filename == "model.safetensors"
        assert cfg.model_device == "cuda"
        assert cfg.codec_device == "cuda"
        assert cfg.model_precision == "bf16"
        assert cfg.codec_precision == "fp32"
        assert cfg.codec_repo == "Aratako/Semantic-DACVAE-Japanese-32dim"
        assert cfg.codec_deterministic_encode is True
        assert cfg.codec_deterministic_decode is True
        assert cfg.caption_checkpoint is None
        assert cfg.caption_hf_repo is None
        assert cfg.caption_hf_filename == "model.safetensors"
        assert cfg.tail_window_size == 20
        assert cfg.tail_std_threshold == 0.05
        assert cfg.tail_mean_threshold == 0.1
        assert cfg.show_timings is True
        assert cfg.speakers == []

    def test_scalars_are_coerced_to_declared_types(self, tmp_path: Path):
        cfg = load_config(
            write_config(
                tmp_path / "c.yaml",
                {
                    "tail_window_size": "30",
                    "tail_std_threshold": "0.25",
                    "tail_mean_threshold": 1,
                    "show_timings": 0,
                    "codec_deterministic_encode": "",
                    "base_checkpoint": 123,
                },
            )
        )
        assert cfg.tail_window_size == 30
        assert cfg.tail_std_threshold == 0.25
        assert cfg.tail_mean_threshold == 1.0
        assert cfg.show_timings is False
        assert cfg.codec_deterministic_encode is False
        assert cfg.base_checkpoint == "123"

    def test_falsy_checkpoint_becomes_none(self, tmp_path: Path):
        cfg = load_config(
            write_config(tmp_path / "c.yaml", {"base_checkpoint": "", "caption_hf_repo": ""})
        )
        assert cfg.base_checkpoint is None
        assert cfg.caption_hf_repo is None


class TestLoadConfigSpeakers:
    def test_inline_speaker_fields(self, tmp_path: Path):
        cfg = load_config(
            write_config(
                tmp_path / "c.yaml",
                {
                    "speakers": [
                        speaker_entry(
                            defaults={"num_steps": 30},
                            category_id=" cat ",
                            category_label=" Cat ",
                        )
                    ]
                },
            )
        )
        (spec,) = cfg.speakers
        assert spec == SpeakerSpec(
            uuid=UUID_A,
            name="Alice",
            adapter="/models/alice.safetensors",
            defaults={"num_steps": 30},
            category_id="cat",
            category_label="Cat",
            cv=None,
        )

    def test_optional_speaker_fields_default_to_none(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {"speakers": [speaker_entry()]}))
        (spec,) = cfg.speakers
        assert spec.defaults == {}
        assert spec.category_id is None
        assert spec.category_label is None
        assert spec.cv is None

    def test_whitespace_only_category_collapses_to_none(self, tmp_path: Path):
        cfg = load_config(
            write_config(
                tmp_path / "c.yaml",
                {"speakers": [speaker_entry(category_id="   ", category_label="  ")]},
            )
        )
        (spec,) = cfg.speakers
        assert spec.category_id is None
        assert spec.category_label is None

    def test_cv_is_not_readable_from_yaml(self, tmp_path: Path):
        """``cv`` is only populated from LoRA metadata; the YAML key is ignored."""
        cfg = load_config(
            write_config(tmp_path / "c.yaml", {"speakers": [speaker_entry(cv="CV Name")]})
        )
        assert cfg.speakers[0].cv is None

    @pytest.mark.parametrize("missing", ["uuid", "name", "adapter"])
    def test_missing_required_speaker_key_raises(self, tmp_path: Path, missing: str):
        entry = speaker_entry()
        del entry[missing]
        path = write_config(tmp_path / "c.yaml", {"speakers": [entry]})
        with pytest.raises(KeyError, match=missing):
            load_config(path)

    def test_null_speakers_list_is_empty(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {"speakers": None}))
        assert cfg.speakers == []


class TestLoadConfigErrors:
    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_empty_file_is_reported_as_a_bad_config(self, tmp_path: Path):
        """An empty YAML parses to None, which is not a usable config root."""
        path = tmp_path / "c.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Config root must be a mapping"):
            load_config(path)

    def test_non_mapping_root_is_reported_as_a_bad_config(self, tmp_path: Path):
        path = tmp_path / "c.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Config root must be a mapping"):
            load_config(path)

    def test_malformed_yaml(self, tmp_path: Path):
        path = tmp_path / "c.yaml"
        path.write_text("speakers: [\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_config(path)


# ===================================================================
# LoRA discovery
# ===================================================================


class TestResolveLoraDisplayName:
    @pytest.mark.parametrize(
        ("meta", "expected"),
        [
            ({"speaker.label": "Label", "name": "Name", "speaker": "Speaker"}, "Label"),
            ({"name": "Name", "speaker": "Speaker"}, "Name"),
            ({"speaker": "Speaker"}, "Speaker"),
            ({"speaker.label": "  ", "name": "Name"}, "Name"),
            ({"speaker.label": "", "name": "", "speaker": ""}, "fallback"),
            ({}, "fallback"),
        ],
    )
    def test_precedence(self, meta: dict[str, str], expected: str):
        assert _resolve_lora_display_name(meta, "fallback") == expected

    def test_values_are_stripped(self):
        assert _resolve_lora_display_name({"name": "  Padded  "}, "fallback") == "Padded"


class TestDiscoverLoraDir:
    def test_missing_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="lora_dir does not exist"):
            _discover_lora_dir(tmp_path / "absent")

    def test_file_instead_of_dir_raises(self, tmp_path: Path):
        path = tmp_path / "not_a_dir"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="lora_dir does not exist"):
            _discover_lora_dir(path)

    def test_empty_dir(self, tmp_path: Path):
        assert _discover_lora_dir(tmp_path) == []

    def test_full_metadata(self, tmp_path: Path):
        entry = write_lora(
            tmp_path / "alice.safetensors",
            {
                "name": "Alice",
                "uuid": UUID_A,
                "defaults": json.dumps({"num_steps": 30}),
                "category.id": " cat ",
                "category.label": " Cat ",
                "speaker.cv": " CV ",
            },
        )
        (spec,) = _discover_lora_dir(tmp_path)
        assert spec == SpeakerSpec(
            uuid=UUID_A,
            name="Alice",
            adapter=str(entry),
            defaults={"num_steps": 30},
            category_id="cat",
            category_label="Cat",
            cv="CV",
        )

    def test_uuid_derived_from_stem_when_absent(self, tmp_path: Path):
        write_lora(tmp_path / "alice.safetensors")
        (spec,) = _discover_lora_dir(tmp_path)
        assert spec.uuid == str(uuid_lib.uuid5(_LORA_UUID_NAMESPACE, "alice"))
        assert spec.name == "alice"

    def test_derived_uuid_is_stable(self, tmp_path: Path):
        write_lora(tmp_path / "alice.safetensors")
        first = _discover_lora_dir(tmp_path)[0].uuid
        assert first == _discover_lora_dir(tmp_path)[0].uuid

    def test_results_sorted_by_filename(self, tmp_path: Path):
        for stem in ("charlie", "alice", "bravo"):
            write_lora(tmp_path / f"{stem}.safetensors")
        assert [s.name for s in _discover_lora_dir(tmp_path)] == ["alice", "bravo", "charlie"]

    def test_adapters_in_subdirectories_are_discovered(self, tmp_path: Path):
        (tmp_path / "genshin_impact").mkdir()
        (tmp_path / "wuthering_waves").mkdir()
        write_lora(tmp_path / "genshin_impact" / "gi_paimon.safetensors")
        write_lora(tmp_path / "wuthering_waves" / "wuwa_yangyang.safetensors")
        write_lora(tmp_path / "loose.safetensors")
        assert [s.name for s in _discover_lora_dir(tmp_path)] == [
            "gi_paimon",
            "loose",
            "wuwa_yangyang",
        ]

    def test_non_lora_safetensors_skipped(self, tmp_path: Path):
        save_file({"w": torch.zeros(2)}, str(tmp_path / "plain.safetensors"))
        write_lora(tmp_path / "alice.safetensors")
        assert [s.name for s in _discover_lora_dir(tmp_path)] == ["alice"]

    def test_non_safetensors_files_ignored(self, tmp_path: Path):
        (tmp_path / "readme.txt").write_text("x", encoding="utf-8")
        (tmp_path / "alice.pt").write_text("x", encoding="utf-8")
        assert _discover_lora_dir(tmp_path) == []

    def test_malformed_defaults_json_is_dropped(self, tmp_path: Path):
        write_lora(tmp_path / "alice.safetensors", {"defaults": "not json"})
        (spec,) = _discover_lora_dir(tmp_path)
        assert spec.defaults == {}

    def test_non_dict_defaults_json_is_dropped(self, tmp_path: Path):
        write_lora(tmp_path / "alice.safetensors", {"defaults": "[1, 2]"})
        (spec,) = _discover_lora_dir(tmp_path)
        assert spec.defaults == {}

    def test_empty_uuid_metadata_falls_back_to_derived(self, tmp_path: Path):
        write_lora(tmp_path / "alice.safetensors", {"uuid": ""})
        (spec,) = _discover_lora_dir(tmp_path)
        assert spec.uuid == str(uuid_lib.uuid5(_LORA_UUID_NAMESPACE, "alice"))


class TestLoadConfigLoraDir:
    def test_absolute_lora_dir(self, tmp_path: Path):
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir()
        write_lora(lora_dir / "alice.safetensors", {"name": "Alice", "uuid": UUID_A})
        cfg = load_config(write_config(tmp_path / "c.yaml", {"lora_dir": str(lora_dir)}))
        assert [s.name for s in cfg.speakers] == ["Alice"]

    def test_relative_lora_dir_resolves_against_config_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        lora_dir = conf_dir / "loras"
        lora_dir.mkdir()
        write_lora(lora_dir / "alice.safetensors", {"name": "Alice", "uuid": UUID_A})
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        cfg = load_config(write_config(conf_dir / "c.yaml", {"lora_dir": "loras"}))
        assert [s.name for s in cfg.speakers] == ["Alice"]

    def test_relative_lora_dir_prefers_cwd_when_it_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """CWD-relative resolution wins over the config dir, and that is intentional.

        ``configs/runtime.yaml`` ships ``lora_dir: models/LoRA`` while living in
        ``configs/``: resolving against the config file would look in
        ``configs/models/LoRA`` and find nothing. In the runtime container the
        config is mounted at ``/app/config.yaml`` with CWD ``/app``, so both
        readings coincide there. Do not "fix" this into config-relative.
        """
        conf_dir = tmp_path / "conf"
        (conf_dir / "loras").mkdir(parents=True)
        write_lora(conf_dir / "loras" / "from_config.safetensors")
        cwd = tmp_path / "cwd"
        (cwd / "loras").mkdir(parents=True)
        write_lora(cwd / "loras" / "from_cwd.safetensors")
        monkeypatch.chdir(cwd)
        cfg = load_config(write_config(conf_dir / "c.yaml", {"lora_dir": "loras"}))
        assert [s.name for s in cfg.speakers] == ["from_cwd"]

    def test_discovered_speakers_come_before_inline_speakers(self, tmp_path: Path):
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir()
        write_lora(lora_dir / "zzz.safetensors", {"name": "Discovered", "uuid": UUID_B})
        cfg = load_config(
            write_config(
                tmp_path / "c.yaml",
                {"lora_dir": str(lora_dir), "speakers": [speaker_entry()]},
            )
        )
        assert [s.name for s in cfg.speakers] == ["Discovered", "Alice"]

    def test_falsy_lora_dir_is_skipped(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {"lora_dir": ""}))
        assert cfg.speakers == []

    def test_missing_lora_dir_propagates(self, tmp_path: Path):
        path = write_config(tmp_path / "c.yaml", {"lora_dir": str(tmp_path / "absent")})
        with pytest.raises(FileNotFoundError, match="lora_dir does not exist"):
            load_config(path)


# ===================================================================
# Checkpoint resolution
# ===================================================================


class TestResolveCheckpoint:
    def test_existing_local_path_wins(self, tmp_path: Path):
        ckpt = tmp_path / "model.safetensors"
        ckpt.write_text("x", encoding="utf-8")
        assert _resolve_checkpoint(str(ckpt), "some/repo", "model.safetensors", "base") == ckpt

    def test_existing_local_dir_is_accepted(self, tmp_path: Path):
        assert _resolve_checkpoint(str(tmp_path), None, "model.safetensors", "base") == tmp_path

    def test_missing_local_and_no_repo_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="base checkpoint not found"):
            _resolve_checkpoint(str(tmp_path / "absent"), None, "model.safetensors", "base")

    def test_no_local_and_no_repo_raises(self):
        with pytest.raises(FileNotFoundError, match="caption checkpoint not found"):
            _resolve_checkpoint(None, None, "model.safetensors", "caption")

    def test_resolve_base_checkpoint_reads_config(self, tmp_path: Path):
        ckpt = tmp_path / "base.safetensors"
        ckpt.write_text("x", encoding="utf-8")
        cfg = load_config(write_config(tmp_path / "c.yaml", {"base_checkpoint": str(ckpt)}))
        assert resolve_base_checkpoint(cfg) == ckpt

    def test_resolve_base_checkpoint_without_anything_raises(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {}))
        with pytest.raises(FileNotFoundError, match="base checkpoint not found"):
            resolve_base_checkpoint(cfg)


# ===================================================================
# Caption runtime selection
# ===================================================================


class TestCaptionRuntimeSelection:
    def test_capable_base_serves_captions_without_a_second_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        calls = install_fake_runtimes(monkeypatch, base_caption=True)
        registry = RuntimeRegistry(load_config(caption_test_config(tmp_path)))
        registry.load()
        base, _ = registry.acquire(UUID_A)
        assert registry.caption_available is True
        assert registry.acquire_caption() is base
        assert len(calls["base"]) == 1
        assert calls["caption"] == []

    def test_explicit_caption_checkpoint_wins_over_a_capable_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        calls = install_fake_runtimes(monkeypatch, base_caption=True)
        caption_ckpt = tmp_path / "voicedesign.safetensors"
        caption_ckpt.write_text("x", encoding="utf-8")
        path = caption_test_config(tmp_path, caption_checkpoint=str(caption_ckpt))
        registry = RuntimeRegistry(load_config(path))
        registry.load()
        base, _ = registry.acquire(UUID_A)
        assert registry.acquire_caption() is not base
        assert calls["caption"] == [str(caption_ckpt)]

    def test_legacy_sidecar_still_serves_captions_for_an_incapable_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        calls = install_fake_runtimes(monkeypatch, base_caption=False)
        path = caption_test_config(
            tmp_path, caption_hf_repo="Aratako/Irodori-TTS-500M-v2-VoiceDesign"
        )
        registry = RuntimeRegistry(load_config(path))
        registry.load()
        base, _ = registry.acquire(UUID_A)
        assert registry.caption_available is True
        assert registry.acquire_caption() is not base
        assert len(calls["caption"]) == 1

    def test_incapable_base_without_a_caption_checkpoint_has_no_caption(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        install_fake_runtimes(monkeypatch, base_caption=False)
        registry = RuntimeRegistry(load_config(caption_test_config(tmp_path)))
        registry.load()
        assert registry.caption_available is False
        with pytest.raises(RuntimeError, match="Caption runtime not configured"):
            registry.acquire_caption()

    def test_unloaded_registry_has_no_caption(self, tmp_path: Path):
        registry = RuntimeRegistry(load_config(caption_test_config(tmp_path)))
        assert registry.caption_available is False
        with pytest.raises(RuntimeError, match="Caption runtime not configured"):
            registry.acquire_caption()


# ===================================================================
# SynthRequest schema
# ===================================================================


class TestSynthRequestSchema:
    def test_all_fields_optional(self):
        req = SynthRequest()
        assert req.speaker_id is None
        assert req.text is None
        assert req.seed is None
        assert req.script is None
        assert req.seconds is None
        assert req.min_seconds is None
        assert req.max_seconds is None
        assert req.duration_scale is None

    def test_negative_seed_and_scales_are_accepted(self):
        req = SynthRequest(text="hi", seed=-1, num_steps=-5, cfg_scale_text=-1.0)
        assert req.seed == -1
        assert req.num_steps == -5

    @pytest.mark.parametrize(
        "payload",
        [
            {"text": ""},
            {"text": "hi", "seconds": 0},
            {"text": "hi", "seconds": -1},
            {"text": "hi", "min_seconds": 0},
            {"text": "hi", "max_seconds": 0},
            {"text": "hi", "duration_scale": 0},
            {"text": "hi", "min_seconds": 5.0, "max_seconds": 1.0},
            {"text": "hi", "seed": "abc"},
        ],
    )
    def test_rejected_payloads(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            SynthRequest(**payload)

    def test_equal_duration_bounds_allowed(self):
        assert SynthRequest(text="hi", min_seconds=2.0, max_seconds=2.0).min_seconds == 2.0

    def test_unknown_fields_are_ignored(self):
        assert not hasattr(SynthRequest(text="hi", bogus=1), "bogus")


class TestVdsSchemaModels:
    def test_minimal_script_body(self):
        body = VdsScriptBody(
            version=1,
            speakers={"a": {"type": "lora", "uuid": UUID_A}},
            cues=[{"kind": "speech", "speaker": "a", "text": "hi"}],
        )
        assert body.title is None
        assert body.defaults is None
        assert body.cues[0].options is None

    def test_defaults_gap_is_one(self):
        defaults = VdsDefaults()
        assert defaults.gap == 1.0
        assert defaults.num_steps is None
        assert defaults.seed is None

    def test_model_dump_exclude_none_round_trips_to_parser_shape(self):
        body = VdsScriptBody(
            version=1,
            speakers={"a": {"type": "lora", "uuid": UUID_A}},
            cues=[{"kind": "speech", "speaker": "a", "text": "hi"}],
        )
        assert body.model_dump(exclude_none=True) == {
            "version": 1,
            "speakers": {"a": {"type": "lora", "uuid": UUID_A}},
            "cues": [{"kind": "speech", "speaker": "a", "text": "hi"}],
        }

    @pytest.mark.parametrize(
        ("payload", "loc"),
        [
            ({"version": 2, "speakers": {}, "cues": []}, ("version",)),
            (
                {"version": 1, "speakers": {"a": {"type": "lora", "uuid": "nope"}}, "cues": []},
                ("speakers", "a", "lora", "uuid"),
            ),
            (
                {"version": 1, "speakers": {"a": {"type": "bogus"}}, "cues": []},
                ("speakers", "a"),
            ),
            (
                {"version": 1, "speakers": {"a": {"type": "caption", "caption": ""}}, "cues": []},
                ("speakers", "a", "caption", "caption"),
            ),
            ({"version": 1, "speakers": {}, "cues": [{"kind": "bgm"}]}, ("cues", 0)),
            (
                {"version": 1, "speakers": {}, "cues": [{"kind": "pause", "duration": 0}]},
                ("cues", 0, "pause", "duration"),
            ),
            (
                {"version": 1, "speakers": {}, "cues": [{"kind": "scene", "name": ""}]},
                ("cues", 0, "scene", "name"),
            ),
            ({"version": 1, "cues": []}, ("speakers",)),
            ({"version": 1, "speakers": {}}, ("cues",)),
        ],
    )
    def test_rejected_script_bodies(self, payload: dict[str, Any], loc: tuple[Any, ...]):
        with pytest.raises(ValidationError) as excinfo:
            VdsScriptBody(**payload)
        assert any(err["loc"] == loc for err in excinfo.value.errors())

    def test_negative_gap_rejected(self):
        with pytest.raises(ValidationError):
            VdsDefaults(gap=-1)

    def test_zero_gap_allowed(self):
        assert VdsDefaults(gap=0).gap == 0.0


# ===================================================================
# _merge_defaults
# ===================================================================

BASE_RESOLVED = {
    "num_steps": 40,
    "cfg_scale_text": 3.0,
    "cfg_scale_speaker": 5.0,
    "speaker_kv_scale": None,
    "truncation_factor": None,
    "seconds": None,
    "min_seconds": 0.5,
    "max_seconds": 30.0,
    "duration_scale": 1.0,
    "seed": None,
}


class TestMergeDefaults:
    def test_bare_request_gets_hardcoded_defaults(self):
        assert _merge_defaults(SynthRequest(text="hi"), {}) == BASE_RESOLVED

    def test_speaker_defaults_override_hardcoded(self):
        params = _merge_defaults(SynthRequest(text="hi"), {"num_steps": 10, "max_seconds": 12.0})
        assert params["num_steps"] == 10
        assert params["max_seconds"] == 12.0

    def test_request_overrides_speaker_defaults(self):
        params = _merge_defaults(SynthRequest(text="hi", num_steps=5), {"num_steps": 10})
        assert params["num_steps"] == 5

    def test_unknown_speaker_default_keys_ignored(self):
        params = _merge_defaults(SynthRequest(text="hi"), {"bogus": 1})
        assert "bogus" not in params

    def test_speaker_default_seed_is_used(self):
        params = _merge_defaults(SynthRequest(text="hi"), {"seed": 99})
        assert params["seed"] == 99

    def test_request_seed_overrides_speaker_default_seed(self):
        params = _merge_defaults(SynthRequest(text="hi", seed=7), {"seed": 99})
        assert params["seed"] == 7

    def test_negative_request_seed_means_random_despite_speaker_default(self):
        params = _merge_defaults(SynthRequest(text="hi", seed=-1), {"seed": 99})
        assert params["seed"] is None

    def test_negative_speaker_default_seed_means_random(self):
        params = _merge_defaults(SynthRequest(text="hi"), {"seed": -1})
        assert params["seed"] is None

    @pytest.mark.parametrize(
        "field",
        [
            "num_steps",
            "cfg_scale_text",
            "cfg_scale_speaker",
            "speaker_kv_scale",
            "truncation_factor",
        ],
    )
    def test_non_positive_override_falls_back_to_speaker_default(self, field: str):
        req = SynthRequest(text="hi", **{field: -1})
        params = _merge_defaults(req, {field: 7.5})
        assert params[field] == 7.5

    def test_zero_override_is_also_ignored(self):
        params = _merge_defaults(SynthRequest(text="hi", cfg_scale_text=0.0), {})
        assert params["cfg_scale_text"] == 3.0

    @pytest.mark.parametrize("bad", [-2, 0])
    def test_non_positive_duration_scale_from_defaults_raises_http_422(self, bad: float):
        with pytest.raises(HTTPException) as excinfo:
            _merge_defaults(SynthRequest(text="hi"), {"duration_scale": bad})
        assert excinfo.value.status_code == 422
        assert "duration_scale" in excinfo.value.detail

    def test_positive_duration_scale_from_defaults_is_kept(self):
        params = _merge_defaults(SynthRequest(text="hi"), {"duration_scale": 1.5})
        assert params["duration_scale"] == 1.5

    @pytest.mark.parametrize(("seed", "expected"), [(None, None), (-1, None), (0, 0), (42, 42)])
    def test_seed_normalization(self, seed: int | None, expected: int | None):
        assert _merge_defaults(SynthRequest(text="hi", seed=seed), {})["seed"] == expected

    def test_merged_bounds_inversion_raises_http_422(self):
        with pytest.raises(HTTPException) as excinfo:
            _merge_defaults(SynthRequest(text="hi"), {"min_seconds": 10.0, "max_seconds": 2.0})
        assert excinfo.value.status_code == 422
        assert "after merging speaker defaults" in excinfo.value.detail

    def test_request_bound_can_rescue_speaker_default_inversion(self):
        params = _merge_defaults(
            SynthRequest(text="hi", max_seconds=20.0),
            {"min_seconds": 10.0, "max_seconds": 2.0},
        )
        assert params["min_seconds"] == 10.0
        assert params["max_seconds"] == 20.0


# ===================================================================
# _apply_fade
# ===================================================================


class TestApplyFade:
    def test_length_preserved(self):
        audio = np.ones(1000, dtype=np.float32)
        assert len(_apply_fade(audio, 16000)) == 1000

    def test_ramps_in_and_out(self):
        audio = np.ones(1000, dtype=np.float32)
        faded = _apply_fade(audio, 1000)
        assert faded[0] == 0.0
        assert faded[-1] == 0.0
        assert faded[49] == pytest.approx(1.0)
        assert faded[500] == 1.0
        assert np.all(np.diff(faded[:50]) > 0)
        assert np.all(np.diff(faded[-50:]) < 0)

    def test_input_is_not_mutated(self):
        audio = np.ones(1000, dtype=np.float32)
        faded = _apply_fade(audio, 1000)
        assert faded is not audio
        assert audio[0] == 1.0

    def test_zero_length_fade_returns_input_unchanged(self):
        audio = np.ones(1000, dtype=np.float32)
        assert _apply_fade(audio, 0) is audio

    def test_sample_rate_too_low_for_one_fade_sample(self):
        """int(19 * 50 / 1000) == 0, so no fade is applied."""
        audio = np.ones(1000, dtype=np.float32)
        assert _apply_fade(audio, 19) is audio

    def test_audio_shorter_than_two_fades_returned_unchanged(self):
        audio = np.ones(99, dtype=np.float32)
        assert _apply_fade(audio, 1000) is audio

    def test_audio_exactly_two_fades_is_faded(self):
        audio = np.ones(100, dtype=np.float32)
        faded = _apply_fade(audio, 1000)
        assert faded is not audio
        assert faded[0] == 0.0
        assert faded[-1] == 0.0

    def test_empty_audio_returned_unchanged(self):
        audio = np.zeros(0, dtype=np.float32)
        assert _apply_fade(audio, 1000) is audio

    def test_float64_preserves_dtype(self):
        audio = np.ones(1000, dtype=np.float64)
        assert _apply_fade(audio, 1000).dtype == np.float64

    def test_short_torch_tensor_takes_the_early_return(self):
        audio = torch.ones(99)
        assert _apply_fade(audio, 1000) is audio

    def test_torch_tensor_is_unsupported(self):
        """Numpy only, by design: every call site converts with ``.numpy()`` first.

        The helper calls ``ndarray.copy()``, which ``torch.Tensor`` does not
        provide. No reachable path hands it a tensor, so this is documented
        rather than supported.
        """
        with pytest.raises(AttributeError):
            _apply_fade(torch.ones(1000), 1000)


# ===================================================================
# _synth_single
# ===================================================================


class TestSynthSingle:
    def test_shortcode_expansion_does_not_mutate_the_request(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {}))
        req = SynthRequest(text="ねえ{cheerful}", speaker_id=UUID_A)
        with pytest.raises(HTTPException) as excinfo:
            _synth_single(
                RuntimeRegistry(cfg),
                cfg,
                req,
                Request({"type": "http", "headers": []}),
            )
        assert excinfo.value.status_code == 404
        assert req.text == "ねえ{cheerful}"


# ===================================================================
# HTTP surface
# ===================================================================


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    path = write_config(
        tmp_path / "c.yaml",
        {
            "speakers": [
                speaker_entry(
                    defaults={"num_steps": 30},
                    category_id="female",
                    category_label="女性",
                )
            ]
        },
    )
    return TestClient(build_app(path, eager_load=False))


@pytest.fixture
def empty_client(tmp_path: Path) -> TestClient:
    return TestClient(build_app(write_config(tmp_path / "c.yaml", {}), eager_load=False))


def vds_upload(source: str | bytes) -> dict[str, Any]:
    data = source.encode("utf-8") if isinstance(source, str) else source
    return {"file": ("script.vds", data, "text/plain")}


class TestHealth:
    def test_reports_speaker_count(self, client: TestClient):
        assert client.get("/health").json() == {"status": "ok", "speakers": 1, "caption": False}

    def test_empty_config(self, empty_client: TestClient):
        assert empty_client.get("/health").json() == {
            "status": "ok",
            "speakers": 0,
            "caption": False,
        }


class TestSpeakers:
    def test_payload_shape(self, client: TestClient):
        assert client.get("/speakers").json() == {
            "speakers": [
                {
                    "uuid": UUID_A,
                    "name": "Alice",
                    "cv": None,
                    "defaults": {"num_steps": 30},
                    "category": {"id": "female", "label": "女性"},
                }
            ]
        }

    def test_empty_config(self, empty_client: TestClient):
        assert empty_client.get("/speakers").json() == {"speakers": []}


class TestSynthValidation:
    def test_missing_text(self, client: TestClient):
        response = client.post("/synth", json={"speaker_id": UUID_A})
        assert response.status_code == 422
        assert response.json()["detail"] == "'text' is required"

    def test_empty_text_rejected_by_schema(self, client: TestClient):
        response = client.post("/synth", json={"speaker_id": UUID_A, "text": ""})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "text"]

    def test_neither_speaker_nor_caption(self, client: TestClient):
        response = client.post("/synth", json={"text": "hi"})
        assert response.status_code == 422
        assert response.json()["detail"] == "either 'speaker_id' or 'caption' is required"

    def test_speaker_and_caption_mutually_exclusive(self, client: TestClient):
        response = client.post(
            "/synth", json={"text": "hi", "speaker_id": UUID_A, "caption": "やわらかい声"}
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "'speaker_id' and 'caption' are mutually exclusive"

    def test_inverted_duration_bounds(self, client: TestClient):
        response = client.post(
            "/synth",
            json={"text": "hi", "speaker_id": UUID_A, "min_seconds": 5, "max_seconds": 1},
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "value_error"

    def test_non_positive_seconds(self, client: TestClient):
        response = client.post("/synth", json={"text": "hi", "speaker_id": UUID_A, "seconds": 0})
        assert response.status_code == 422
        assert response.json()["detail"][0]["ctx"] == {"gt": 0.0}

    def test_unknown_speaker_id_is_404(self, client: TestClient):
        response = client.post("/synth", json={"text": "hi", "speaker_id": UUID_B})
        assert response.status_code == 404
        assert response.json()["detail"] == f"unknown speaker_id: {UUID_B}"

    def test_caption_without_any_caption_capable_runtime_is_501(self, client: TestClient):
        """No runtime is loaded here, so nothing can serve captions."""
        response = client.post("/synth", json={"text": "hi", "caption": "やわらかい声"})
        assert response.status_code == 501
        assert response.json()["detail"] == "caption runtime not configured"

    def test_malformed_script_is_422(self, client: TestClient):
        response = client.post(
            "/synth",
            json={"script": {"version": 2, "speakers": {}, "cues": []}},
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "script", "version"]

    def test_script_without_speech_cues_is_422(self, client: TestClient):
        response = client.post(
            "/synth",
            json={
                "script": {
                    "version": 1,
                    "speakers": {"a": {"type": "lora", "uuid": UUID_A}},
                    "cues": [{"kind": "scene", "name": "幕"}],
                }
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "no speech cues in script"

    def test_script_with_unknown_uuid_is_404(self, client: TestClient):
        response = client.post(
            "/synth",
            json={
                "script": {
                    "version": 1,
                    "speakers": {"a": {"type": "lora", "uuid": UUID_B}},
                    "cues": [{"kind": "speech", "speaker": "a", "text": "hi"}],
                }
            },
        )
        assert response.status_code == 404
        assert UUID_B in response.json()["detail"]

    def test_script_takes_precedence_over_missing_text(self, client: TestClient):
        """With a script present, the single-cue 'text is required' check is skipped."""
        response = client.post(
            "/synth",
            json={
                "script": {
                    "version": 1,
                    "speakers": {"a": {"type": "caption", "caption": "やわらかい声"}},
                    "cues": [{"kind": "speech", "speaker": "a", "text": "hi"}],
                }
            },
        )
        assert response.status_code == 501


class TestSynthVdsValidation:
    def test_missing_file_is_422(self, client: TestClient):
        assert client.post("/synth/vds").status_code == 422

    def test_non_utf8_body_is_422(self, client: TestClient):
        response = client.post("/synth/vds", files=vds_upload(b"\xff\xfe\x00bad"))
        assert response.status_code == 422
        assert response.json()["detail"] == "file must be UTF-8 encoded"

    def test_parse_error_is_422(self, client: TestClient):
        response = client.post("/synth/vds", files=vds_upload("@version: 2\n"))
        assert response.status_code == 422
        assert "unsupported version" in response.json()["detail"]

    def test_no_speech_cues_is_422(self, client: TestClient):
        response = client.post(
            "/synth/vds", files=vds_upload(f"@version: 1\n@speaker a = {UUID_A}\n")
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "no speech cues in script"

    def test_unknown_uuid_is_404(self, client: TestClient):
        response = client.post(
            "/synth/vds", files=vds_upload(f"@version: 1\n@speaker a = {UUID_B}\n\na: hi\n")
        )
        assert response.status_code == 404
        assert UUID_B in response.json()["detail"]

    def test_caption_speaker_without_any_caption_capable_runtime_is_501(self, client: TestClient):
        response = client.post(
            "/synth/vds",
            files=vds_upload('@version: 1\n@speaker a = caption "やわらかい声"\n\na: hi\n'),
        )
        assert response.status_code == 501
        assert "caption runtime not configured" in response.json()["detail"]

    def test_utf8_bom_is_stripped(self, client: TestClient):
        source = f"@version: 1\n@speaker a = {UUID_B}\n\na: hi\n"
        response = client.post(
            "/synth/vds", files=vds_upload(b"\xef\xbb\xbf" + source.encode("utf-8"))
        )
        assert response.status_code == 404


class TestCaptionCapableBaseRoutes:
    """With a caption-capable base and no caption checkpoint, caption requests stop being 501."""

    @pytest.fixture
    def caption_client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        install_fake_runtimes(monkeypatch, base_caption=True)
        return TestClient(build_app(caption_test_config(tmp_path), eager_load=True))

    def test_health_reports_caption(self, caption_client: TestClient):
        assert caption_client.get("/health").json() == {
            "status": "ok",
            "speakers": 1,
            "caption": True,
        }

    def test_single_cue_caption_synthesis(self, caption_client: TestClient):
        response = caption_client.post("/synth", json={"text": "hi", "caption": "やわらかい声"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"
        assert response.headers["X-TTS-Sample-Rate"] == "48000"

    def test_vds_caption_speaker(self, caption_client: TestClient):
        response = caption_client.post(
            "/synth/vds",
            files=vds_upload('@version: 1\n@speaker a = caption "やわらかい声"\n\na: hi\n'),
        )
        assert response.status_code == 200
        assert response.headers["X-TTS-Cue-Count"] == "1"


class TestOpenApiContract:
    def test_routes_are_registered(self, client: TestClient):
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths) == {"/health", "/speakers", "/synth", "/synth/vds"}
        assert set(paths["/synth"]) == {"post"}
        assert set(paths["/synth/vds"]) == {"post"}
        assert set(paths["/health"]) == {"get"}
        assert set(paths["/speakers"]) == {"get"}

    def test_audio_media_types_documented(self, client: TestClient):
        paths = client.get("/openapi.json").json()["paths"]
        for route in ("/synth", "/synth/vds"):
            content = paths[route]["post"]["responses"]["200"]["content"]
            assert set(content) >= {"audio/wav", "audio/pcm"}
