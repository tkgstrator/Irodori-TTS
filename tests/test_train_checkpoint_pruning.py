"""Characterization tests for the checkpoint file management helpers in train.py.

These pin down the current filename conventions and retention logic so that a
verbatim move into a training subpackage can be shown to be behavior-preserving.
Everything here runs on CPU against tmp_path: no GPU, no real model, no network.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from irodori_tts.config import ModelConfig, TrainConfig
from train import (
    _best_checkpoint_path,
    _final_checkpoint_path,
    _periodic_checkpoint_path,
    _runtime_state_for_checkpoint,
    _safe_unlink,
    enforce_periodic_checkpoint_limit,
    list_best_val_loss_checkpoints,
    list_periodic_checkpoints,
    maybe_save_best_val_loss_checkpoint,
    prune_best_val_loss_checkpoints,
    save_checkpoint,
)

PLAIN_CFG = TrainConfig()
LORA_CFG = replace(PLAIN_CFG, lora_enabled=True)
SPEAKER_INVERSION_CFG = replace(PLAIN_CFG, speaker_inversion_enabled=True)


def _touch(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).write_text("x", encoding="utf-8")


def _names(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Linear(2, 2, bias=True)


def _tiny_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.1)


def test_periodic_checkpoint_path_plain(tmp_path: Path) -> None:
    assert _periodic_checkpoint_path(tmp_path, 100, PLAIN_CFG) == tmp_path / "checkpoint_0000100.pt"


def test_periodic_checkpoint_path_lora_is_extensionless_dir_name(tmp_path: Path) -> None:
    assert _periodic_checkpoint_path(tmp_path, 100, LORA_CFG) == tmp_path / "checkpoint_0000100"


def test_periodic_checkpoint_path_speaker_inversion(tmp_path: Path) -> None:
    assert (
        _periodic_checkpoint_path(tmp_path, 100, SPEAKER_INVERSION_CFG)
        == tmp_path / "checkpoint_0000100.speaker.safetensors"
    )


def test_speaker_inversion_takes_precedence_over_lora(tmp_path: Path) -> None:
    both = replace(PLAIN_CFG, speaker_inversion_enabled=True, lora_enabled=True)
    assert (
        _periodic_checkpoint_path(tmp_path, 7, both).name
        == "checkpoint_0000007.speaker.safetensors"
    )
    assert _final_checkpoint_path(tmp_path, both).name == "checkpoint_final.speaker.safetensors"


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, "checkpoint_0000000.pt"),
        (1, "checkpoint_0000001.pt"),
        (1000000, "checkpoint_1000000.pt"),
        (123456789, "checkpoint_123456789.pt"),
    ],
)
def test_periodic_checkpoint_path_zero_pads_without_truncating(
    tmp_path: Path, step: int, expected: str
) -> None:
    assert _periodic_checkpoint_path(tmp_path, step, PLAIN_CFG).name == expected


@pytest.mark.parametrize(
    ("val_loss", "expected"),
    [
        (0.5, "checkpoint_best_val_loss_0000100_0.500000.pt"),
        (-0.5, "checkpoint_best_val_loss_0000100_-0.500000.pt"),
        (0.1234567, "checkpoint_best_val_loss_0000100_0.123457.pt"),
        (2, "checkpoint_best_val_loss_0000100_2.000000.pt"),
    ],
)
def test_best_checkpoint_path_formats_val_loss_to_six_decimals(
    tmp_path: Path, val_loss: float, expected: str
) -> None:
    path = _best_checkpoint_path(tmp_path, step=100, val_loss=val_loss, train_cfg=PLAIN_CFG)
    assert path.name == expected


def test_best_checkpoint_path_variants(tmp_path: Path) -> None:
    kwargs = {"step": 100, "val_loss": 0.5}
    assert (
        _best_checkpoint_path(tmp_path, train_cfg=LORA_CFG, **kwargs).name
        == "checkpoint_best_val_loss_0000100_0.500000"
    )
    assert (
        _best_checkpoint_path(tmp_path, train_cfg=SPEAKER_INVERSION_CFG, **kwargs).name
        == "checkpoint_best_val_loss_0000100_0.500000.speaker.safetensors"
    )


def test_final_checkpoint_path_variants(tmp_path: Path) -> None:
    assert _final_checkpoint_path(tmp_path, PLAIN_CFG).name == "checkpoint_final.pt"
    assert _final_checkpoint_path(tmp_path, LORA_CFG).name == "checkpoint_final"
    assert (
        _final_checkpoint_path(tmp_path, SPEAKER_INVERSION_CFG).name
        == "checkpoint_final.speaker.safetensors"
    )


def test_list_periodic_checkpoints_empty_dir(tmp_path: Path) -> None:
    assert list_periodic_checkpoints(tmp_path) == []


def test_list_periodic_checkpoints_sorted_newest_first(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "checkpoint_0000100.pt",
        "checkpoint_0002000.pt",
        "checkpoint_0000300.pt",
    )
    assert [step for step, _ in list_periodic_checkpoints(tmp_path)] == [2000, 300, 100]


def test_list_periodic_checkpoints_accepts_all_three_naming_variants(tmp_path: Path) -> None:
    _touch(tmp_path, "checkpoint_0000100.pt", "checkpoint_0000400.speaker.safetensors")
    (tmp_path / "checkpoint_0000300").mkdir()
    found = list_periodic_checkpoints(tmp_path)
    assert [step for step, _ in found] == [400, 300, 100]


@pytest.mark.parametrize(
    "name",
    [
        "checkpoint_final.pt",
        "checkpoint_abc.pt",
        "checkpoint_.pt",
        "checkpoint_0000100.pt.tmp",
        "checkpoint_0000100.safetensors",
        "checkpoint_best_val_loss_0000100_0.500000.pt",
        "model_0000100.pt",
        "checkpoint_0000100.PT",
    ],
)
def test_list_periodic_checkpoints_ignores_non_matching_names(tmp_path: Path, name: str) -> None:
    _touch(tmp_path, name)
    assert list_periodic_checkpoints(tmp_path) == []


@pytest.mark.parametrize("variant", ["plain", "lora", "speaker_inversion"])
def test_list_periodic_checkpoints_round_trips_path_builder(tmp_path: Path, variant: str) -> None:
    cfg = {"plain": PLAIN_CFG, "lora": LORA_CFG, "speaker_inversion": SPEAKER_INVERSION_CFG}[
        variant
    ]
    path = _periodic_checkpoint_path(tmp_path, 4200, cfg)
    path.write_text("x", encoding="utf-8")
    assert list_periodic_checkpoints(tmp_path) == [(4200, path)]


@pytest.mark.parametrize("keep_count", [0, -1])
def test_enforce_periodic_checkpoint_limit_disabled(tmp_path: Path, keep_count: int) -> None:
    _touch(tmp_path, "checkpoint_0000100.pt", "checkpoint_0000200.pt", "checkpoint_0000300.pt")
    enforce_periodic_checkpoint_limit(tmp_path, keep_count)
    assert len(_names(tmp_path)) == 3


def test_enforce_periodic_checkpoint_limit_keeps_newest(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "checkpoint_0000100.pt",
        "checkpoint_0000200.pt",
        "checkpoint_0000300.pt",
        "checkpoint_0000400.pt",
    )
    enforce_periodic_checkpoint_limit(tmp_path, 2)
    assert _names(tmp_path) == ["checkpoint_0000300.pt", "checkpoint_0000400.pt"]


def test_enforce_periodic_checkpoint_limit_spares_unrelated_files(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "checkpoint_0000100.pt",
        "checkpoint_0000200.pt",
        "checkpoint_final.pt",
        "checkpoint_best_val_loss_0000100_0.500000.pt",
    )
    enforce_periodic_checkpoint_limit(tmp_path, 1)
    assert _names(tmp_path) == [
        "checkpoint_0000200.pt",
        "checkpoint_best_val_loss_0000100_0.500000.pt",
        "checkpoint_final.pt",
    ]


def test_enforce_periodic_checkpoint_limit_no_op_when_under_limit(tmp_path: Path) -> None:
    _touch(tmp_path, "checkpoint_0000100.pt", "checkpoint_0000200.pt")
    enforce_periodic_checkpoint_limit(tmp_path, 5)
    assert len(_names(tmp_path)) == 2


def test_enforce_periodic_checkpoint_limit_removes_lora_directories(tmp_path: Path) -> None:
    for step in (100, 200):
        directory = tmp_path / f"checkpoint_{step:07d}"
        directory.mkdir()
        _touch(directory, "adapter_model.safetensors")
    enforce_periodic_checkpoint_limit(tmp_path, 1)
    assert _names(tmp_path) == ["checkpoint_0000200"]


def test_enforce_periodic_checkpoint_limit_on_empty_dir(tmp_path: Path) -> None:
    enforce_periodic_checkpoint_limit(tmp_path, 1)
    assert _names(tmp_path) == []


def test_safe_unlink_tolerates_missing_path(tmp_path: Path) -> None:
    _safe_unlink(tmp_path / "does_not_exist.pt")


def test_list_best_val_loss_checkpoints_empty_dir(tmp_path: Path) -> None:
    assert list_best_val_loss_checkpoints(tmp_path) == []


def test_list_best_val_loss_checkpoints_sorted_by_score_then_step(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "checkpoint_best_val_loss_0000100_0.500000.pt",
        "checkpoint_best_val_loss_0000200_0.500000.pt",
        "checkpoint_best_val_loss_0000300_-1.250000.pt",
        "checkpoint_best_val_loss_0000400_0.100000.pt",
    )
    assert [(score, step) for score, step, _ in list_best_val_loss_checkpoints(tmp_path)] == [
        (-1.25, 300),
        (0.1, 400),
        (0.5, 100),
        (0.5, 200),
    ]


def test_list_best_val_loss_checkpoints_accepts_bare_and_speaker_suffixes(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "checkpoint_best_val_loss_0000500_1.5.speaker.safetensors",
        "checkpoint_best_val_loss_0000400_2",
    )
    assert [(score, step) for score, step, _ in list_best_val_loss_checkpoints(tmp_path)] == [
        (1.5, 500),
        (2.0, 400),
    ]


@pytest.mark.parametrize(
    "name",
    [
        "checkpoint_best_val_loss_0000600_abc.pt",
        "checkpoint_best_val_loss_x_1.000000.pt",
        "checkpoint_best_val_loss_0000100_0.500000.safetensors",
        "checkpoint_best_val_loss_0000100.pt",
        "checkpoint_0000100.pt",
    ],
)
def test_list_best_val_loss_checkpoints_ignores_non_matching_names(
    tmp_path: Path, name: str
) -> None:
    _touch(tmp_path, name)
    assert list_best_val_loss_checkpoints(tmp_path) == []


def test_list_best_val_loss_checkpoints_round_trips_path_builder(tmp_path: Path) -> None:
    path = _best_checkpoint_path(tmp_path, step=4200, val_loss=0.1234567, train_cfg=PLAIN_CFG)
    path.write_text("x", encoding="utf-8")
    # The score survives only to the six decimals baked into the filename.
    assert list_best_val_loss_checkpoints(tmp_path) == [(0.123457, 4200, path)]


@pytest.mark.parametrize("keep_best_n", [0, -1])
def test_prune_best_val_loss_checkpoints_disabled_returns_input_unsorted(
    tmp_path: Path, keep_best_n: int
) -> None:
    _touch(tmp_path, "a", "b")
    items = [(0.5, 2, tmp_path / "a"), (0.1, 1, tmp_path / "b")]
    assert prune_best_val_loss_checkpoints(items, keep_best_n) is items
    assert _names(tmp_path) == ["a", "b"]


def test_prune_best_val_loss_checkpoints_sorts_and_deletes_worst(tmp_path: Path) -> None:
    _touch(tmp_path, "a", "b", "c")
    items = [(0.9, 3, tmp_path / "c"), (0.1, 1, tmp_path / "a"), (0.5, 2, tmp_path / "b")]
    kept = prune_best_val_loss_checkpoints(items, 2)
    assert [path.name for _, _, path in kept] == ["a", "b"]
    assert _names(tmp_path) == ["a", "b"]


def test_prune_best_val_loss_checkpoints_breaks_score_ties_by_higher_step(tmp_path: Path) -> None:
    _touch(tmp_path, "old", "new")
    items = [(0.5, 100, tmp_path / "old"), (0.5, 200, tmp_path / "new")]
    kept = prune_best_val_loss_checkpoints(items, 1)
    assert [path.name for _, _, path in kept] == ["old"]
    assert _names(tmp_path) == ["old"]


def test_prune_best_val_loss_checkpoints_within_limit_does_not_mutate_input(
    tmp_path: Path,
) -> None:
    _touch(tmp_path, "a", "b")
    items = [(0.5, 2, tmp_path / "b"), (0.1, 1, tmp_path / "a")]
    kept = prune_best_val_loss_checkpoints(items, 5)
    assert [path.name for _, _, path in kept] == ["a", "b"]
    assert [path.name for _, _, path in items] == ["b", "a"]
    assert _names(tmp_path) == ["a", "b"]


def test_save_checkpoint_writes_payload_and_creates_parent_dirs(tmp_path: Path) -> None:
    model = _tiny_model()
    path = tmp_path / "nested" / "checkpoint_0000100.pt"
    save_checkpoint(
        path,
        model,
        _tiny_optimizer(model),
        None,
        100,
        ModelConfig(),
        PLAIN_CFG,
        es_best_val=0.5,
        es_no_improve=2,
        manifest_size=7,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert sorted(payload) == [
        "dataloader_state",
        "es_best_val",
        "es_no_improve",
        "manifest_size",
        "model",
        "model_config",
        "optimizer",
        "runtime_state",
        "scheduler",
        "step",
        "text_encoder_config",
        "train_config",
    ]
    assert payload["step"] == 100
    assert payload["es_best_val"] == 0.5
    assert payload["es_no_improve"] == 2
    assert payload["manifest_size"] == 7
    assert payload["scheduler"] is None
    assert payload["text_encoder_config"] is None
    assert sorted(payload["model"]) == ["bias", "weight"]


def test_save_checkpoint_defaults_optional_metadata_to_none(tmp_path: Path) -> None:
    model = _tiny_model()
    path = tmp_path / "checkpoint_0000001.pt"
    save_checkpoint(path, model, _tiny_optimizer(model), None, 1, ModelConfig(), PLAIN_CFG)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["es_best_val"] is None
    assert payload["es_no_improve"] is None
    assert payload["manifest_size"] is None
    assert payload["dataloader_state"] is None
    assert payload["runtime_state"] is None


@pytest.mark.parametrize(
    ("epoch", "expected_sampler_epoch"),
    [(0, 0), (1, 0), (3, 2)],
)
def test_runtime_state_for_checkpoint(epoch: int, expected_sampler_epoch: int) -> None:
    assert _runtime_state_for_checkpoint(epoch=epoch, epoch_step=12) == {
        "epoch": epoch,
        "sampler_epoch": expected_sampler_epoch,
        "epoch_step": 12,
    }


def _save_best(
    tmp_path: Path,
    checkpoints: list[tuple[float, int, Path]],
    *,
    val_loss: float,
    step: int,
    keep_best_n: int = 2,
) -> tuple[list[tuple[float, int, Path]], Path | None]:
    model = _tiny_model()
    return maybe_save_best_val_loss_checkpoint(
        output_dir=tmp_path,
        checkpoints=checkpoints,
        keep_best_n=keep_best_n,
        val_loss=val_loss,
        step=step,
        model=model,
        optimizer=_tiny_optimizer(model),
        scheduler=None,
        model_cfg=ModelConfig(),
        train_cfg=PLAIN_CFG,
        base_init=None,
    )


def test_maybe_save_best_disabled(tmp_path: Path) -> None:
    checkpoints, path = _save_best(tmp_path, [], val_loss=0.1, step=100, keep_best_n=0)
    assert checkpoints == []
    assert path is None
    assert _names(tmp_path) == []


def test_maybe_save_best_saves_while_under_capacity_even_if_worse(tmp_path: Path) -> None:
    checkpoints, _ = _save_best(tmp_path, [], val_loss=0.1, step=100)
    checkpoints, path = _save_best(tmp_path, checkpoints, val_loss=99.0, step=200)
    assert path is not None
    assert path.name == "checkpoint_best_val_loss_0000200_99.000000.pt"
    assert [step for _, step, _ in checkpoints] == [100, 200]


def test_maybe_save_best_rejects_score_equal_to_worst(tmp_path: Path) -> None:
    checkpoints, _ = _save_best(tmp_path, [], val_loss=0.1, step=100)
    checkpoints, _ = _save_best(tmp_path, checkpoints, val_loss=0.5, step=200)
    before = _names(tmp_path)
    kept, path = _save_best(tmp_path, checkpoints, val_loss=0.5, step=300)
    assert path is None
    assert kept == checkpoints
    assert _names(tmp_path) == before


def test_maybe_save_best_rejects_worse_score_at_capacity(tmp_path: Path) -> None:
    checkpoints, _ = _save_best(tmp_path, [], val_loss=0.1, step=100)
    checkpoints, _ = _save_best(tmp_path, checkpoints, val_loss=0.5, step=200)
    _, path = _save_best(tmp_path, checkpoints, val_loss=0.6, step=300)
    assert path is None


def test_maybe_save_best_evicts_worst_file_when_better_arrives(tmp_path: Path) -> None:
    checkpoints, _ = _save_best(tmp_path, [], val_loss=0.9, step=100)
    checkpoints, _ = _save_best(tmp_path, checkpoints, val_loss=0.5, step=200)
    checkpoints, path = _save_best(tmp_path, checkpoints, val_loss=0.7, step=300)
    assert path is not None
    assert _names(tmp_path) == [
        "checkpoint_best_val_loss_0000200_0.500000.pt",
        "checkpoint_best_val_loss_0000300_0.700000.pt",
    ]
    assert [(score, step) for score, step, _ in checkpoints] == [(0.5, 200), (0.7, 300)]


def test_maybe_save_best_replaces_existing_entry_for_same_step(tmp_path: Path) -> None:
    checkpoints, _ = _save_best(tmp_path, [], val_loss=0.9, step=100)
    checkpoints, _ = _save_best(tmp_path, checkpoints, val_loss=0.5, step=200)
    checkpoints, path = _save_best(tmp_path, checkpoints, val_loss=0.1, step=200)
    assert path is not None
    assert path.name == "checkpoint_best_val_loss_0000200_0.100000.pt"
    assert _names(tmp_path) == [
        "checkpoint_best_val_loss_0000100_0.900000.pt",
        "checkpoint_best_val_loss_0000200_0.100000.pt",
    ]
    assert [(score, step) for score, step, _ in checkpoints] == [(0.1, 200), (0.9, 100)]


# The speaker-inversion branch of save_checkpoint is not covered: it requires a
# model carrying an enabled SpeakerInversionEmbedding module, not a dummy one.
