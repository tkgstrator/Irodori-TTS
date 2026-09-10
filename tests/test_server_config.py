"""Tests for everything below the HTTP boundary: config loading, LoRA discovery,
the runtime registry, the request schema, defaults merging and the fade helper.

No GPU, no checkpoints and no network. The HTTP surface lives in
``test_openai_api.py``.
"""

from __future__ import annotations

import json
import uuid as uuid_lib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError
from safetensors.torch import save_file

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
from irodori_tts.server.errors import ApiError
from irodori_tts.server.registry import RuntimeRegistry
from irodori_tts.server.schemas import SpeechRequest, _merge_defaults
from tests.helpers import (
    UUID_A,
    UUID_B,
    install_fake_runtime,
    lora_test_config,
    speaker_entry,
    write_config,
    write_lora,
)


def speech(**overrides: Any) -> SpeechRequest:
    payload: dict[str, Any] = {"model": "m", "input": "hi", "voice": UUID_A}
    payload.update(overrides)
    return SpeechRequest(**payload)


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
        assert cfg.model_id == "irodori-tts"
        assert cfg.tail_window_size == 20
        assert cfg.tail_std_threshold == 0.05
        assert cfg.tail_mean_threshold == 0.1
        assert cfg.show_timings is True
        assert cfg.speakers == []

    def test_base_version_resolves_to_upstream_repo(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {"base_version": "v4.1-small"}))
        assert cfg.base_hf_repo == "Aratako/Irodori-TTS-v4.1-Small"

    def test_base_version_is_stripped(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {"base_version": "  v3  "}))
        assert cfg.base_hf_repo == "Aratako/Irodori-TTS-500M-v3"

    def test_unknown_base_version_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown base_version: 'v5'"):
            load_config(write_config(tmp_path / "c.yaml", {"base_version": "v5"}))

    def test_base_version_with_explicit_repo_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not both"):
            load_config(
                write_config(
                    tmp_path / "c.yaml",
                    {"base_version": "v3", "base_hf_repo": "someone/fork"},
                )
            )

    def test_explicit_repo_alone_is_kept(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {"base_hf_repo": "someone/fork"}))
        assert cfg.base_hf_repo == "someone/fork"

    def test_empty_base_version_falls_back_to_explicit_repo(self, tmp_path: Path):
        cfg = load_config(
            write_config(
                tmp_path / "c.yaml",
                {"base_version": "", "base_hf_repo": "someone/fork"},
            )
        )
        assert cfg.base_hf_repo == "someone/fork"

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
        cfg = load_config(write_config(tmp_path / "c.yaml", {"base_checkpoint": ""}))
        assert cfg.base_checkpoint is None


class TestModelId:
    def test_derived_from_base_version(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {"base_version": "v4.1-small"}))
        assert cfg.model_id == "irodori-tts-v4.1-small"

    def test_derived_from_an_explicit_repo(self, tmp_path: Path):
        cfg = load_config(write_config(tmp_path / "c.yaml", {"base_hf_repo": "Someone/My-Fork"}))
        assert cfg.model_id == "my-fork"

    def test_explicit_model_id_wins(self, tmp_path: Path):
        cfg = load_config(
            write_config(tmp_path / "c.yaml", {"base_version": "v3", "model_id": "custom"})
        )
        assert cfg.model_id == "custom"


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
    def test_missing_dir_is_empty(self, tmp_path: Path):
        assert _discover_lora_dir(tmp_path / "absent") == []

    def test_file_instead_of_dir_is_empty(self, tmp_path: Path):
        path = tmp_path / "not_a_dir"
        path.write_text("x", encoding="utf-8")
        assert _discover_lora_dir(path) == []

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

    def test_missing_lora_dir_yields_no_speakers(self, tmp_path: Path):
        path = write_config(tmp_path / "c.yaml", {"lora_dir": str(tmp_path / "absent")})
        assert load_config(path).speakers == []


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
        with pytest.raises(FileNotFoundError, match="base checkpoint not found"):
            _resolve_checkpoint(None, None, "model.safetensors", "base")

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
# Runtime loading
# ===================================================================


class TestRuntimeLoad:
    def test_base_and_adapters_are_loaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        calls = install_fake_runtime(monkeypatch)
        registry = RuntimeRegistry(load_config(lora_test_config(tmp_path)))
        registry.load()
        assert len(calls["base"]) == 1
        assert registry.acquire(UUID_A)[1].name == "Alice"

    def test_no_speakers_loads_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Every voice is a LoRA, so a base with none attached has nothing to say."""
        calls = install_fake_runtime(monkeypatch)
        ckpt = tmp_path / "base.safetensors"
        ckpt.write_text("x", encoding="utf-8")
        path = write_config(tmp_path / "c.yaml", {"base_checkpoint": str(ckpt)})
        registry = RuntimeRegistry(load_config(path))
        registry.load()
        assert calls["base"] == []
        with pytest.raises(KeyError):
            registry.acquire(UUID_A)

    def test_watermarking_is_left_on_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        install_fake_runtime(monkeypatch)
        registry = RuntimeRegistry(load_config(lora_test_config(tmp_path)))
        registry.load()
        base, _ = registry.acquire(UUID_A)
        assert base.watermarker.model is not None

    def test_disabling_the_watermark_drops_the_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        install_fake_runtime(monkeypatch)
        path = lora_test_config(tmp_path, enable_watermark=False)
        registry = RuntimeRegistry(load_config(path))
        registry.load()
        base, _ = registry.acquire(UUID_A)
        assert base.watermarker.model is None

    def test_unloaded_registry_refuses_to_acquire(self, tmp_path: Path):
        registry = RuntimeRegistry(load_config(lora_test_config(tmp_path)))
        with pytest.raises(RuntimeError, match="not loaded"):
            registry.acquire(UUID_A)


# ===================================================================
# SpeechRequest schema
# ===================================================================


class TestSpeechRequestSchema:
    def test_defaults(self):
        req = speech()
        assert req.response_format == "mp3"
        assert req.speed == 1.0
        assert req.stream_format == "audio"
        assert req.instructions is None
        assert req.seed is None

    def test_voice_accepts_the_object_form(self):
        assert speech(voice={"id": UUID_A}).voice_id == UUID_A

    def test_voice_accepts_the_string_form(self):
        assert speech(voice=UUID_A).voice_id == UUID_A

    def test_negative_seed_and_scales_are_accepted(self):
        req = speech(seed=-1, num_steps=-5, cfg_scale_text=-1.0)
        assert req.seed == -1
        assert req.num_steps == -5

    @pytest.mark.parametrize(
        "payload",
        [
            {"input": ""},
            {"input": "x" * 4097},
            {"voice": {"nope": "x"}},
            {"response_format": "ogg"},
            {"stream_format": "chunked"},
            {"speed": 0.2},
            {"speed": 4.1},
            {"seconds": 0},
            {"min_seconds": 0},
            {"min_seconds": 5.0, "max_seconds": 1.0},
            {"seed": "abc"},
        ],
    )
    def test_rejected_payloads(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            speech(**payload)

    @pytest.mark.parametrize("missing", ["model", "input", "voice"])
    def test_required_fields(self, missing: str):
        payload = {"model": "m", "input": "hi", "voice": UUID_A}
        del payload[missing]
        with pytest.raises(ValidationError):
            SpeechRequest(**payload)

    def test_speed_bounds_are_inclusive(self):
        assert speech(speed=0.25).speed == 0.25
        assert speech(speed=4.0).speed == 4.0

    def test_equal_duration_bounds_allowed(self):
        assert speech(min_seconds=2.0, max_seconds=2.0).min_seconds == 2.0

    def test_unknown_fields_are_ignored(self):
        assert not hasattr(speech(bogus=1), "bogus")


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
        assert _merge_defaults(speech(), {}) == BASE_RESOLVED

    def test_speaker_defaults_override_hardcoded(self):
        params = _merge_defaults(speech(), {"num_steps": 10, "max_seconds": 12.0})
        assert params["num_steps"] == 10
        assert params["max_seconds"] == 12.0

    def test_request_overrides_speaker_defaults(self):
        params = _merge_defaults(speech(num_steps=5), {"num_steps": 10})
        assert params["num_steps"] == 5

    def test_unknown_speaker_default_keys_ignored(self):
        params = _merge_defaults(speech(), {"bogus": 1})
        assert "bogus" not in params

    def test_speaker_default_seed_is_used(self):
        params = _merge_defaults(speech(), {"seed": 99})
        assert params["seed"] == 99

    def test_request_seed_overrides_speaker_default_seed(self):
        params = _merge_defaults(speech(seed=7), {"seed": 99})
        assert params["seed"] == 7

    def test_negative_request_seed_means_random_despite_speaker_default(self):
        params = _merge_defaults(speech(seed=-1), {"seed": 99})
        assert params["seed"] is None

    def test_negative_speaker_default_seed_means_random(self):
        params = _merge_defaults(speech(), {"seed": -1})
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
        params = _merge_defaults(speech(**{field: -1}), {field: 7.5})
        assert params[field] == 7.5

    def test_zero_override_is_also_ignored(self):
        params = _merge_defaults(speech(cfg_scale_text=0.0), {})
        assert params["cfg_scale_text"] == 3.0

    def test_speed_does_not_touch_the_merged_duration_scale(self):
        """``speed`` is applied on top of the merged value, not merged into it."""
        params = _merge_defaults(speech(speed=2.0), {"duration_scale": 1.5})
        assert params["duration_scale"] == 1.5

    @pytest.mark.parametrize("bad", [-2, 0])
    def test_non_positive_duration_scale_from_defaults_is_rejected(self, bad: float):
        with pytest.raises(ApiError) as excinfo:
            _merge_defaults(speech(), {"duration_scale": bad})
        assert excinfo.value.status_code == 400
        assert "duration_scale" in excinfo.value.message

    def test_positive_duration_scale_from_defaults_is_kept(self):
        params = _merge_defaults(speech(), {"duration_scale": 1.5})
        assert params["duration_scale"] == 1.5

    @pytest.mark.parametrize(("seed", "expected"), [(None, None), (-1, None), (0, 0), (42, 42)])
    def test_seed_normalization(self, seed: int | None, expected: int | None):
        assert _merge_defaults(speech(seed=seed), {})["seed"] == expected

    def test_merged_bounds_inversion_is_rejected(self):
        with pytest.raises(ApiError) as excinfo:
            _merge_defaults(speech(), {"min_seconds": 10.0, "max_seconds": 2.0})
        assert excinfo.value.status_code == 400
        assert "after merging speaker defaults" in excinfo.value.message

    def test_request_bound_can_rescue_speaker_default_inversion(self):
        params = _merge_defaults(
            speech(max_seconds=20.0),
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
