"""Characterization tests for the pure helpers in irodori_tts.inference_runtime.

These pin down the behavior of the module as it exists today, before its
contents are split into smaller modules. Nothing here loads a checkpoint,
a codec, or a model: only CPU-testable helpers and the import surface that
server.py / infer.py / the gradio apps rely on.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import pathlib

import pytest
import torch
import torchaudio

from irodori_tts import inference_runtime
from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    _coerce_latent_shape,
    default_runtime_device,
    find_flattening_point,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    resolve_cfg_scales,
    resolve_runtime_device,
    resolve_runtime_dtype,
    save_wav,
)

# Names that server.py, infer.py, gradio_app.py, gradio_app_voicedesign.py,
# irodori_tts/training_samples.py and scripts/** import by name.
PUBLIC_API_NAMES = (
    "InferenceRuntime",
    "RuntimeKey",
    "SamplingRequest",
    "SamplingResult",
    "clear_cached_runtime",
    "default_runtime_device",
    "download_hf_checkpoint",
    "find_flattening_point",
    "get_cached_runtime",
    "list_available_runtime_devices",
    "list_available_runtime_precisions",
    "resolve_cfg_scales",
    "resolve_runtime_device",
    "resolve_runtime_dtype",
    "save_wav",
)

# Underscored, but imported across module boundaries anyway, so a move that
# drops them breaks convert_checkpoint_to_safetensors.py / quantize_checkpoint.py.
CROSS_MODULE_PRIVATE_NAMES = ("_coerce_latent_shape", "_load_checkpoint_for_inference")


# ===================================================================
# resolve_cfg_scales
# ===================================================================


class TestResolveCfgScales:
    def test_passthrough_independent(self):
        text, caption, speaker, messages = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=2.0,
            cfg_scale_speaker=5.0,
            cfg_scale=None,
        )
        assert text == pytest.approx(3.0)
        assert caption == pytest.approx(2.0)
        assert speaker == pytest.approx(5.0)
        assert messages == []

    def test_cfg_scale_overrides_all_three(self):
        text, caption, speaker, messages = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=2.0,
            cfg_scale_speaker=5.0,
            cfg_scale=1.5,
        )
        assert (text, caption, speaker) == pytest.approx((1.5, 1.5, 1.5))
        assert messages == []

    def test_values_are_coerced_to_float(self):
        text, caption, speaker, _ = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=3,
            cfg_scale_caption=2,
            cfg_scale_speaker=1,
            cfg_scale=None,
        )
        assert isinstance(text, float)
        assert isinstance(caption, float)
        assert isinstance(speaker, float)

    def test_joint_accepts_equal_scales(self):
        text, caption, speaker, messages = resolve_cfg_scales(
            cfg_guidance_mode="joint",
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=3.0,
            cfg_scale=None,
        )
        assert (text, caption, speaker) == pytest.approx((3.0, 3.0, 3.0))
        assert messages == []

    def test_joint_rejects_mismatched_scales(self):
        with pytest.raises(ValueError, match="cfg_guidance_mode='joint' requires equal"):
            resolve_cfg_scales(
                cfg_guidance_mode="joint",
                cfg_scale_text=3.0,
                cfg_scale_caption=2.0,
                cfg_scale_speaker=3.0,
                cfg_scale=None,
            )

    def test_joint_tolerates_tiny_difference(self):
        text, caption, speaker, _ = resolve_cfg_scales(
            cfg_guidance_mode="joint",
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0000001,
            cfg_scale_speaker=3.0,
            cfg_scale=None,
        )
        assert (text, speaker) == pytest.approx((3.0, 3.0))
        assert caption == pytest.approx(3.0)

    def test_joint_with_explicit_cfg_scale_bypasses_mismatch(self):
        text, caption, speaker, _ = resolve_cfg_scales(
            cfg_guidance_mode="joint",
            cfg_scale_text=3.0,
            cfg_scale_caption=2.0,
            cfg_scale_speaker=5.0,
            cfg_scale=4.0,
        )
        assert (text, caption, speaker) == pytest.approx((4.0, 4.0, 4.0))

    def test_mode_is_case_and_whitespace_insensitive(self):
        with pytest.raises(ValueError, match="requires equal"):
            resolve_cfg_scales(
                cfg_guidance_mode="  JOINT ",
                cfg_scale_text=3.0,
                cfg_scale_caption=2.0,
                cfg_scale_speaker=3.0,
                cfg_scale=None,
            )

    def test_unknown_mode_is_not_validated_here(self):
        # cfg_guidance_mode is validated in synthesize(); this helper only
        # special-cases "joint" and lets anything else through untouched.
        text, caption, speaker, messages = resolve_cfg_scales(
            cfg_guidance_mode="banana",
            cfg_scale_text=3.0,
            cfg_scale_caption=2.0,
            cfg_scale_speaker=5.0,
            cfg_scale=None,
        )
        assert (text, caption, speaker) == pytest.approx((3.0, 2.0, 5.0))
        assert messages == []

    def test_speaker_disabled_zeroes_scale_and_warns(self):
        text, caption, speaker, messages = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=2.0,
            cfg_scale_speaker=5.0,
            cfg_scale=None,
            use_speaker_condition=False,
        )
        assert (text, caption) == pytest.approx((3.0, 2.0))
        assert speaker == pytest.approx(0.0)
        assert len(messages) == 1
        assert "ignoring cfg_scale_speaker" in messages[0]

    def test_speaker_disabled_stays_quiet_when_scale_already_zero(self):
        _, _, speaker, messages = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=2.0,
            cfg_scale_speaker=0.0,
            cfg_scale=None,
            use_speaker_condition=False,
        )
        assert speaker == pytest.approx(0.0)
        assert messages == []

    def test_joint_ignores_disabled_speaker_when_checking_equality(self):
        _, _, speaker, messages = resolve_cfg_scales(
            cfg_guidance_mode="joint",
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=9.0,
            cfg_scale=None,
            use_speaker_condition=False,
        )
        assert speaker == pytest.approx(0.0)
        assert len(messages) == 1

    def test_disabled_caption_scale_is_returned_unchanged(self):
        # Unlike the speaker scale, a caption scale is not zeroed when the
        # caption condition is off: it is only skipped by the joint check.
        _, caption, _, messages = resolve_cfg_scales(
            cfg_guidance_mode="joint",
            cfg_scale_text=3.0,
            cfg_scale_caption=99.0,
            cfg_scale_speaker=3.0,
            cfg_scale=None,
            use_caption_condition=False,
        )
        assert caption == pytest.approx(99.0)
        assert messages == []

    def test_non_positive_scales_are_excluded_from_joint_check(self):
        text, caption, speaker, _ = resolve_cfg_scales(
            cfg_guidance_mode="joint",
            cfg_scale_text=-1.0,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=3.0,
            cfg_scale=None,
        )
        assert (text, caption, speaker) == pytest.approx((-1.0, 3.0, 3.0))

    def test_all_scales_zero_is_accepted_in_joint_mode(self):
        text, caption, speaker, messages = resolve_cfg_scales(
            cfg_guidance_mode="joint",
            cfg_scale_text=0.0,
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            cfg_scale=None,
        )
        assert (text, caption, speaker) == pytest.approx((0.0, 0.0, 0.0))
        assert messages == []


# ===================================================================
# find_flattening_point
# ===================================================================


class TestFindFlatteningPoint:
    def test_loud_then_silent_returns_silence_onset(self):
        latent = torch.cat([torch.full((30, 4), 3.0), torch.zeros(40, 4)], dim=0)
        assert find_flattening_point(latent) == 30

    def test_never_flattens_returns_total_steps(self):
        latent = torch.full((50, 4), 10.0)
        assert find_flattening_point(latent) == 50

    def test_flat_but_not_near_zero_is_not_a_flattening_point(self):
        latent = torch.full((60, 4), 5.0)
        assert find_flattening_point(latent) == 60

    def test_near_zero_but_noisy_is_not_a_flattening_point(self):
        tail = torch.tensor([[1.0, -1.0, 1.0, -1.0]]).repeat(40, 1)
        latent = torch.cat([torch.full((30, 4), 3.0), tail], dim=0)
        assert find_flattening_point(latent) == 70

    def test_already_flat_from_the_start_returns_zero(self):
        latent = torch.full((25, 4), 0.01)
        assert find_flattening_point(latent) == 0

    def test_zero_length_latent_returns_zero(self):
        assert find_flattening_point(torch.zeros(0, 4)) == 0

    def test_shorter_than_window_still_scans_the_zero_padding(self):
        assert find_flattening_point(torch.zeros(5, 4)) == 0

    def test_non_positive_window_size_short_circuits(self):
        latent = torch.zeros(10, 3)
        assert find_flattening_point(latent, window_size=0) == 10
        assert find_flattening_point(latent, window_size=-3) == 10

    def test_zero_padding_can_flatten_a_quiet_signal_early(self):
        # The trailing window is zero-padded, so a low-amplitude signal that
        # never actually goes silent can still report a point before its end.
        latent = torch.full((25, 4), 0.15)
        assert find_flattening_point(latent) == 23

    def test_thresholds_are_honored(self):
        latent = torch.full((60, 4), 0.5)
        assert find_flattening_point(latent, mean_threshold=1.0) == 0
        assert find_flattening_point(latent, target_value=0.5) == 0

    def test_wrong_dimensionality_raises(self):
        with pytest.raises(ValueError, match=r"Expected latent shape \(T, D\)"):
            find_flattening_point(torch.zeros(2, 3, 4))
        with pytest.raises(ValueError, match=r"Expected latent shape \(T, D\)"):
            find_flattening_point(torch.zeros(7))


# ===================================================================
# _coerce_latent_shape
# ===================================================================


class TestCoerceLatentShape:
    def test_time_major_is_returned_as_is(self):
        latent = torch.arange(12.0).reshape(3, 4)
        out = _coerce_latent_shape(latent, latent_dim=4)
        assert out.shape == (3, 4)
        assert torch.equal(out, latent)

    def test_dim_major_is_transposed_and_contiguous(self):
        latent = torch.arange(12.0).reshape(3, 4)
        out = _coerce_latent_shape(latent, latent_dim=3)
        assert out.shape == (4, 3)
        assert out.is_contiguous()
        assert torch.equal(out, latent.transpose(0, 1))

    def test_leading_batch_of_one_is_squeezed(self):
        latent = torch.arange(12.0).reshape(1, 3, 4)
        assert _coerce_latent_shape(latent, latent_dim=4).shape == (3, 4)

    def test_square_latent_prefers_time_major(self):
        latent = torch.arange(16.0).reshape(4, 4)
        assert torch.equal(_coerce_latent_shape(latent, latent_dim=4), latent)

    def test_batched_latent_raises(self):
        with pytest.raises(ValueError, match="Unsupported latent shape"):
            _coerce_latent_shape(torch.zeros(2, 3, 4), latent_dim=4)

    def test_one_dimensional_latent_raises(self):
        with pytest.raises(ValueError, match="Unsupported latent shape"):
            _coerce_latent_shape(torch.zeros(5), latent_dim=5)

    def test_latent_dim_matching_neither_axis_raises(self):
        with pytest.raises(ValueError, match="Could not infer latent layout"):
            _coerce_latent_shape(torch.zeros(5, 7), latent_dim=4)


# ===================================================================
# RuntimeKey
# ===================================================================

RUNTIME_KEY_BASE = RuntimeKey(checkpoint="/ckpt/model.safetensors", model_device="cpu")

RUNTIME_KEY_ALTERNATIVES = {
    "checkpoint": "/ckpt/other.safetensors",
    "model_device": "cuda",
    "codec_repo": "SomeoneElse/Other-Codec",
    "model_precision": "bf16",
    "codec_device": "cuda",
    "codec_precision": "fp16",
    "codec_deterministic_encode": False,
    "codec_deterministic_decode": False,
    "compile_model": True,
    "compile_dynamic": True,
}


class TestRuntimeKey:
    def test_defaults(self):
        key = RuntimeKey(checkpoint="/ckpt/model.safetensors", model_device="cpu")
        assert key.codec_repo == "Aratako/Semantic-DACVAE-Japanese-32dim"
        assert key.model_precision == "fp32"
        assert key.codec_device == "cpu"
        assert key.codec_precision == "fp32"
        assert key.codec_deterministic_encode is True
        assert key.codec_deterministic_decode is True
        assert key.compile_model is False
        assert key.compile_dynamic is False

    def test_is_frozen(self):
        key = RuntimeKey(checkpoint="a", model_device="cpu")
        with pytest.raises(dataclasses.FrozenInstanceError):
            key.checkpoint = "b"

    def test_equal_keys_are_interchangeable_cache_entries(self):
        a = RuntimeKey(checkpoint="a", model_device="cpu")
        b = RuntimeKey(checkpoint="a", model_device="cpu")
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1
        assert {a: 1}[b] == 1

    def test_all_fields_are_covered_by_the_distinctness_check(self):
        names = {field.name for field in dataclasses.fields(RuntimeKey)}
        assert names == set(RUNTIME_KEY_ALTERNATIVES)

    @pytest.mark.parametrize("field_name", sorted(RUNTIME_KEY_ALTERNATIVES))
    def test_every_field_distinguishes_two_keys(self, field_name: str):
        # A collision here would make get_cached_runtime hand back a runtime
        # built for different weights, devices, or precision.
        other = dataclasses.replace(
            RUNTIME_KEY_BASE, **{field_name: RUNTIME_KEY_ALTERNATIVES[field_name]}
        )
        assert other != RUNTIME_KEY_BASE
        assert len({RUNTIME_KEY_BASE, other}) == 2


# ===================================================================
# SamplingRequest
# ===================================================================


class TestSamplingRequest:
    def test_only_text_is_required(self):
        req = SamplingRequest(text="こんにちは")
        assert req.text == "こんにちは"
        assert req.caption is None
        assert req.ref_wav is None
        assert req.no_ref is False
        assert req.lora_adapter is None

    def test_documented_defaults(self):
        req = SamplingRequest(text="x")
        assert req.num_candidates == 1
        assert req.decode_mode == "sequential"
        assert req.seconds is None
        assert req.duration_scale == pytest.approx(1.0)
        assert req.min_seconds == pytest.approx(0.5)
        assert req.max_seconds == pytest.approx(30.0)
        assert req.max_ref_seconds is None
        assert req.ref_normalize_db == pytest.approx(-16.0)
        assert req.ref_ensure_max is True
        assert req.num_steps == 40
        assert req.cfg_scale_text == pytest.approx(3.0)
        assert req.cfg_scale_caption == pytest.approx(3.0)
        assert req.cfg_scale_speaker == pytest.approx(5.0)
        assert req.cfg_guidance_mode == "independent"
        assert req.cfg_scale is None
        assert req.cfg_min_t == pytest.approx(0.5)
        assert req.cfg_max_t == pytest.approx(1.0)
        assert req.speaker_uncond_mode == "mask"
        assert req.seed is None
        assert req.t_schedule_mode == "linear"
        assert req.sway_coeff == pytest.approx(-1.0)
        assert req.context_kv_cache is True

    def test_tail_trimming_defaults_match_find_flattening_point(self):
        req = SamplingRequest(text="x")
        assert req.trim_tail is True
        assert req.tail_window_size == 20
        assert req.tail_std_threshold == pytest.approx(0.05)
        assert req.tail_mean_threshold == pytest.approx(0.1)

    def test_is_mutable_and_compares_by_value(self):
        req = SamplingRequest(text="x")
        req.seed = 42
        assert req.seed == 42
        assert SamplingRequest(text="x", seed=42) == req

    def test_no_validation_at_construction(self):
        # Every constraint is enforced inside synthesize(), not on the dataclass.
        req = SamplingRequest(text="", num_candidates=-5, decode_mode="nonsense")
        assert req.num_candidates == -5


# ===================================================================
# Device / precision resolution
# ===================================================================


class TestDeviceAndPrecision:
    def test_cpu_resolves(self):
        assert resolve_runtime_device("cpu") == torch.device("cpu")
        assert resolve_runtime_device(torch.device("cpu")).type == "cpu"

    def test_known_but_unsupported_device_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported inference device"):
            resolve_runtime_device("meta")

    def test_unparseable_device_raises_runtime_error_from_torch(self):
        # torch.device() rejects the string before the helper's own check runs,
        # so callers see a RuntimeError rather than the friendlier ValueError.
        with pytest.raises(RuntimeError, match="device type at start of device string"):
            resolve_runtime_device("tpu")

    def test_cpu_only_supports_fp32(self):
        assert list_available_runtime_precisions("cpu") == ["fp32"]

    def test_cpu_is_always_offered_last(self):
        devices = list_available_runtime_devices()
        assert devices[-1] == "cpu"
        assert default_runtime_device() == devices[0]

    def test_fp32_dtype(self):
        assert resolve_runtime_dtype(precision="FP32", device=torch.device("cpu")) is torch.float32

    def test_reduced_precision_needs_an_accelerator(self):
        cpu = torch.device("cpu")
        with pytest.raises(ValueError, match="bf16"):
            resolve_runtime_dtype(precision="bf16", device=cpu)
        with pytest.raises(ValueError, match="fp16"):
            resolve_runtime_dtype(precision="fp16", device=cpu)

    def test_unknown_precision_raises(self):
        with pytest.raises(ValueError, match="Unsupported precision"):
            resolve_runtime_dtype(precision="int4", device=torch.device("cpu"))


# ===================================================================
# save_wav
# ===================================================================


class TestSaveWav:
    def test_writes_a_readable_mono_wav(self, tmp_path):
        audio = torch.full((1, 480), 0.5)
        out = save_wav(tmp_path / "out.wav", audio, 24000)
        assert out == tmp_path / "out.wav"
        assert out.is_file()

        loaded, sample_rate = torchaudio.load(str(out))
        assert sample_rate == 24000
        assert loaded.shape == (1, 480)
        assert loaded.max().item() == pytest.approx(0.5, abs=1e-4)

    def test_creates_missing_parent_directories(self, tmp_path):
        out = save_wav(tmp_path / "a" / "b" / "out.wav", torch.zeros(1, 64), 16000)
        assert out.is_file()

    def test_preserves_channel_count_and_sample_rate(self, tmp_path):
        out = save_wav(tmp_path / "stereo.wav", torch.zeros(2, 100), 16000)
        loaded, sample_rate = torchaudio.load(str(out))
        assert loaded.shape == (2, 100)
        assert sample_rate == 16000

    def test_accepts_a_string_path_and_non_float32_input(self, tmp_path):
        audio = torch.full((1, 240), 0.25, dtype=torch.float64)
        out = save_wav(str(tmp_path / "f64.wav"), audio, 8000)
        loaded, sample_rate = torchaudio.load(str(out))
        assert sample_rate == 8000
        assert loaded.dtype is torch.float32
        assert loaded[0, 0].item() == pytest.approx(0.25, abs=1e-4)

    def test_detaches_from_autograd(self, tmp_path):
        audio = torch.zeros(1, 64, requires_grad=True) + 0.1
        out = save_wav(tmp_path / "grad.wav", audio, 8000)
        assert out.is_file()


# ===================================================================
# Import surface
# ===================================================================


class TestCheckpointLoaderCallSites:
    """`_load_checkpoint_for_inference` is unpacked, never inspected.

    Both runtimes destructure its return value positionally, so growing the
    tuple silently breaks whichever call site was not updated — and the one
    the server uses is only reachable with a real checkpoint, so no unit test
    covers it. Comparing the annotated arity against every call site catches
    that at import time instead of at model load.
    """

    def test_every_call_site_unpacks_the_full_tuple(self):
        source = pathlib.Path(inference_runtime.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        loader = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_load_checkpoint_for_inference"
        )
        # -> tuple[A, B, C, D]
        arity = len(loader.returns.slice.elts)

        widths = [
            len(node.targets[0].elts)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Tuple)
            and "_load_checkpoint_for_inference" in ast.dump(node.value)
        ]
        assert widths, "no call site found; did the helper get renamed?"
        assert widths == [arity] * len(widths)


class TestPublicApi:
    def test_public_names_are_importable(self):
        module = importlib.import_module("irodori_tts.inference_runtime")
        missing = [name for name in PUBLIC_API_NAMES if not hasattr(module, name)]
        assert missing == []

    def test_cross_module_private_names_are_importable(self):
        module = importlib.import_module("irodori_tts.inference_runtime")
        missing = [name for name in CROSS_MODULE_PRIVATE_NAMES if not hasattr(module, name)]
        assert missing == []

    def test_public_callables_are_callable(self):
        module = importlib.import_module("irodori_tts.inference_runtime")
        not_callable = [name for name in PUBLIC_API_NAMES if not callable(getattr(module, name))]
        assert not_callable == []
