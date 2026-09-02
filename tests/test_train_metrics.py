"""Characterization tests for the metric, split and sampling helpers in train.py.

These pin the behavior of the pure helpers that are about to be moved out of
train.py into a training subpackage. They are deliberately written against the
current implementation, so a verbatim move can be proven behavior-preserving.
"""

from __future__ import annotations

import pytest
import torch

import train

PRED = torch.tensor(
    [
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ]
)
TARGET = torch.tensor(
    [
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    ]
)
# Per-token mean squared error of PRED/TARGET: [[1, 4, 9], [1, 1, 1]].
MASK = torch.tensor([[True, True, False], [True, False, False]])


def _sampler(lengths: torch.Tensor, **kwargs) -> train.LengthGroupedSampler:
    options = {"batch_size": 4, "window_batches": 2, "seed": 0}
    options.update(kwargs)
    return train.LengthGroupedSampler(lengths, **options)


class TestUtteranceMeanMaskedMse:
    def test_matches_hand_computation(self) -> None:
        loss = train.utterance_mean_masked_mse(PRED, TARGET, MASK)
        assert loss.item() == pytest.approx(1.75)

    def test_finite_garbage_behind_mask_is_ignored(self) -> None:
        polluted = PRED.clone()
        polluted[0, 2, :] = 1e6
        polluted[1, 1, :] = -1e9
        loss = train.utterance_mean_masked_mse(polluted, TARGET, MASK)
        assert loss.item() == pytest.approx(1.75)

    def test_nan_behind_mask_propagates(self) -> None:
        # Multiplying by a zero weight does not neutralize NaN, so masked-out
        # NaNs poison the result. Current behavior, not a desirable one.
        polluted = PRED.clone()
        polluted[0, 2, :] = float("nan")
        loss = train.utterance_mean_masked_mse(polluted, TARGET, MASK)
        assert torch.isnan(loss)

    def test_fully_masked_sample_contributes_zero(self) -> None:
        mask = torch.tensor([[True, True, False], [False, False, False]])
        loss = train.utterance_mean_masked_mse(PRED, TARGET, mask)
        assert loss.item() == pytest.approx(1.25)

    def test_returns_scalar_tensor(self) -> None:
        loss = train.utterance_mean_masked_mse(PRED, TARGET, MASK)
        assert loss.shape == ()


class TestEchoStyleMaskedMse:
    def test_matches_hand_computation(self) -> None:
        loss = train.echo_style_masked_mse(PRED, TARGET, MASK, MASK)
        assert loss.item() == pytest.approx(2.0)

    def test_reduces_to_masked_mean_when_masks_match(self) -> None:
        loss = train.echo_style_masked_mse(PRED, TARGET, MASK, MASK)
        diff = ((PRED - TARGET) ** 2).mean(dim=-1)
        weight = MASK.float()
        expected = (diff * weight).sum() / weight.sum()
        torch.testing.assert_close(loss, expected)

    def test_finite_garbage_behind_loss_mask_is_ignored(self) -> None:
        polluted = PRED.clone()
        polluted[0, 2, :] = 1e6
        polluted[1, 1, :] = -1e9
        loss = train.echo_style_masked_mse(polluted, TARGET, MASK, MASK)
        assert loss.item() == pytest.approx(2.0)

    def test_nan_behind_loss_mask_propagates(self) -> None:
        polluted = PRED.clone()
        polluted[0, 2, :] = float("nan")
        loss = train.echo_style_masked_mse(polluted, TARGET, MASK, MASK)
        assert torch.isnan(loss)

    def test_shorter_valid_span_upweights_loss(self) -> None:
        loss_mask = torch.ones(2, 3, dtype=torch.bool)
        wide = train.echo_style_masked_mse(PRED, TARGET, loss_mask, loss_mask)
        narrow = train.echo_style_masked_mse(PRED, TARGET, loss_mask, MASK)
        assert narrow.item() > wide.item()

    def test_sample_without_valid_tokens_leaves_denominator(self) -> None:
        mask = torch.tensor([[True, True, False], [False, False, False]])
        loss = train.echo_style_masked_mse(PRED, TARGET, mask, mask)
        assert loss.item() == pytest.approx(2.5, rel=1e-6)

    def test_denominator_is_clamped(self) -> None:
        loss_mask = torch.ones(2, 3, dtype=torch.bool)
        valid_mask = torch.zeros(2, 3, dtype=torch.bool)
        loss = train.echo_style_masked_mse(PRED, TARGET, loss_mask, valid_mask)
        assert loss.item() == pytest.approx((17.0 / 6.0) * 1e6, rel=1e-4)


class TestComputeRfLoss:
    def test_echo_mode_dispatch(self) -> None:
        loss = train.compute_rf_loss(
            pred=PRED, target=TARGET, loss_mask=MASK, valid_mask=MASK, mode="echo"
        )
        torch.testing.assert_close(loss, train.echo_style_masked_mse(PRED, TARGET, MASK, MASK))

    def test_utterance_mean_mode_dispatch(self) -> None:
        loss = train.compute_rf_loss(
            pred=PRED, target=TARGET, loss_mask=MASK, valid_mask=MASK, mode="utterance_mean"
        )
        torch.testing.assert_close(loss, train.utterance_mean_masked_mse(PRED, TARGET, MASK))

    @pytest.mark.parametrize("mode", ["  ECHO  ", "Echo", "ECHO"])
    def test_mode_is_normalized(self, mode: str) -> None:
        loss = train.compute_rf_loss(
            pred=PRED, target=TARGET, loss_mask=MASK, valid_mask=MASK, mode=mode
        )
        assert loss.item() == pytest.approx(2.0)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported rf_loss_mode"):
            train.compute_rf_loss(
                pred=PRED, target=TARGET, loss_mask=MASK, valid_mask=MASK, mode="mse"
            )


class TestSplitTrainValidIndices:
    def test_is_deterministic(self) -> None:
        first = train.split_train_valid_indices(num_samples=500, valid_ratio=0.2, seed=11)
        second = train.split_train_valid_indices(num_samples=500, valid_ratio=0.2, seed=11)
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])

    def test_partitions_exactly(self) -> None:
        train_idx, valid_idx = train.split_train_valid_indices(
            num_samples=500, valid_ratio=0.2, seed=11
        )
        combined = torch.cat([train_idx, valid_idx])
        assert combined.numel() == 500
        assert sorted(combined.tolist()) == list(range(500))
        assert set(train_idx.tolist()).isdisjoint(valid_idx.tolist())

    def test_both_sides_are_sorted_ascending(self) -> None:
        train_idx, valid_idx = train.split_train_valid_indices(
            num_samples=500, valid_ratio=0.2, seed=11
        )
        assert train_idx.tolist() == sorted(train_idx.tolist())
        assert valid_idx.tolist() == sorted(valid_idx.tolist())
        assert train_idx.dtype == torch.int64
        assert valid_idx.dtype == torch.int64

    def test_different_seeds_give_different_splits(self) -> None:
        _, valid_a = train.split_train_valid_indices(num_samples=500, valid_ratio=0.2, seed=1)
        _, valid_b = train.split_train_valid_indices(num_samples=500, valid_ratio=0.2, seed=2)
        assert valid_a.tolist() != valid_b.tolist()

    @pytest.mark.parametrize("valid_ratio", [0.0, -0.5])
    def test_non_positive_ratio_keeps_everything_for_training(self, valid_ratio: float) -> None:
        train_idx, valid_idx = train.split_train_valid_indices(
            num_samples=7, valid_ratio=valid_ratio, seed=0
        )
        assert train_idx.tolist() == list(range(7))
        assert valid_idx.numel() == 0

    def test_non_positive_ratio_skips_the_minimum_size_check(self) -> None:
        train_idx, valid_idx = train.split_train_valid_indices(
            num_samples=1, valid_ratio=0.0, seed=0
        )
        assert train_idx.tolist() == [0]
        assert valid_idx.numel() == 0

    def test_single_sample_with_validation_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            train.split_train_valid_indices(num_samples=1, valid_ratio=0.1, seed=0)

    def test_small_ratio_is_raised_to_the_floor(self) -> None:
        # int(1000 * 0.01) == 10, but VALID_MIN_COUNT forces it up to 50.
        _, valid_idx = train.split_train_valid_indices(num_samples=1000, valid_ratio=0.01, seed=0)
        assert valid_idx.numel() == train.VALID_MIN_COUNT

    def test_large_ratio_is_capped_at_the_ceiling(self) -> None:
        _, valid_idx = train.split_train_valid_indices(num_samples=1000, valid_ratio=0.5, seed=0)
        assert valid_idx.numel() == train.VALID_MAX_COUNT

    def test_small_dataset_leaves_a_single_training_sample(self) -> None:
        # The floor of 50 exceeds the dataset, so validation swallows all but one
        # sample. Surprising, but this is what the current code does.
        train_idx, valid_idx = train.split_train_valid_indices(
            num_samples=10, valid_ratio=0.5, seed=0
        )
        assert train_idx.numel() == 1
        assert valid_idx.numel() == 9

    def test_two_samples_split_one_and_one(self) -> None:
        train_idx, valid_idx = train.split_train_valid_indices(
            num_samples=2, valid_ratio=0.1, seed=0
        )
        assert sorted(train_idx.tolist() + valid_idx.tolist()) == [0, 1]
        assert train_idx.numel() == 1
        assert valid_idx.numel() == 1


class TestLengthGroupedSampler:
    def test_repeated_iteration_is_deterministic(self) -> None:
        sampler = _sampler(torch.arange(16), seed=7, drop_last=False)
        assert list(sampler) == list(sampler)

    def test_fresh_instances_with_same_seed_agree(self) -> None:
        lengths = torch.arange(16)
        assert list(_sampler(lengths, seed=7, drop_last=False)) == list(
            _sampler(lengths, seed=7, drop_last=False)
        )

    def test_without_drop_last_every_index_appears_once(self) -> None:
        sampler = _sampler(torch.arange(10), drop_last=False)
        indices = list(sampler)
        assert sorted(indices) == list(range(10))
        assert len(sampler) == 10

    def test_drop_last_truncates_to_whole_batches(self) -> None:
        sampler = _sampler(torch.arange(10), drop_last=True)
        indices = list(sampler)
        assert len(sampler) == 8
        assert len(indices) == 8
        assert len(set(indices)) == 8
        assert set(indices).issubset(range(10))

    def test_similar_lengths_share_a_batch(self) -> None:
        lengths = torch.tensor([5, 1, 9, 3, 7, 2, 8, 4, 6, 0, 10, 11, 12, 13, 14, 15])
        # A window wide enough for the whole dataset means every emitted batch is
        # a contiguous run of the globally length-sorted order.
        sampler = _sampler(lengths, batch_size=4, window_batches=100, seed=7, drop_last=False)
        indices = list(sampler)
        for start in range(0, len(indices), 4):
            batch = [int(lengths[i]) for i in indices[start : start + 4]]
            assert max(batch) - min(batch) == 3

    def test_set_epoch_changes_the_order(self) -> None:
        lengths = torch.arange(16)
        sampler = _sampler(lengths, seed=7, drop_last=False)
        first_epoch = list(sampler)
        sampler.set_epoch(1)
        assert list(sampler) != first_epoch

    def test_epoch_shifts_the_seed(self) -> None:
        # seed + epoch is used directly, so (seed=7, epoch=1) aliases (seed=8, epoch=0).
        lengths = torch.arange(16)
        shifted = _sampler(lengths, seed=7, drop_last=False)
        shifted.set_epoch(1)
        assert list(shifted) == list(_sampler(lengths, seed=8, drop_last=False))

    def test_each_epoch_still_covers_the_dataset(self) -> None:
        sampler = _sampler(torch.arange(10), drop_last=False)
        for epoch in range(3):
            sampler.set_epoch(epoch)
            assert sorted(sampler) == list(range(10))

    def test_replicas_together_cover_the_padded_dataset(self) -> None:
        lengths = torch.arange(10)
        rank0 = _sampler(lengths, batch_size=2, num_replicas=2, rank=0, seed=3, drop_last=False)
        rank1 = _sampler(lengths, batch_size=2, num_replicas=2, rank=1, seed=3, drop_last=False)
        left, right = list(rank0), list(rank1)
        assert len(rank0) == len(rank1) == 6
        assert len(left) == len(right) == 6
        # Padding up to a whole global batch repeats indices, so the union is a
        # multiset of 12 entries rather than a clean partition of the 10 samples.
        assert set(left + right) == set(range(10))
        assert len(left + right) == 12

    def test_empty_dataset_yields_nothing(self) -> None:
        sampler = _sampler(torch.empty(0, dtype=torch.int64))
        assert list(sampler) == []
        assert len(sampler) == 0

    def test_rejects_multi_dimensional_lengths(self) -> None:
        with pytest.raises(ValueError, match="lengths must be 1D"):
            _sampler(torch.arange(8).reshape(2, 4))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"batch_size": 0}, "batch_size must be > 0"),
            ({"window_batches": 0}, "window_batches must be > 0"),
            ({"num_replicas": 0}, "num_replicas must be > 0"),
            ({"num_replicas": 2, "rank": 2}, "rank must be in"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            _sampler(torch.arange(8), **kwargs)


def _totals() -> torch.Tensor:
    return train.duration_condition_group_totals(
        duration_loss_per_sample=torch.tensor([1.0, 2.0, 3.0]),
        pred_frames=torch.tensor([10.0, 20.0, 30.0]),
        target_frames=torch.tensor([12.0, 17.0, 30.0]),
        has_speaker=torch.tensor([True, False, True]),
        has_caption=torch.tensor([True, True, False]),
    )


class TestDurationConditionGroupTotals:
    def test_layout(self) -> None:
        totals = _totals()
        assert totals.numel() == train.DURATION_CONDITION_GROUP_TOTAL_SIZE
        assert totals.dtype == torch.float64

    @pytest.mark.parametrize(
        ("group", "loss_sum", "mae_sum", "count"),
        [
            ("speaker", 4.0, 2.0, 2.0),
            ("no_speaker", 2.0, 3.0, 1.0),
            ("caption", 3.0, 5.0, 2.0),
            ("no_caption", 3.0, 0.0, 1.0),
            ("speaker_caption", 1.0, 2.0, 1.0),
            ("speaker_no_caption", 3.0, 0.0, 1.0),
            ("no_speaker_caption", 2.0, 3.0, 1.0),
            ("no_speaker_no_caption", 0.0, 0.0, 0.0),
        ],
    )
    def test_group_sums(self, group: str, loss_sum: float, mae_sum: float, count: float) -> None:
        totals = _totals()
        offset = train.DURATION_CONDITION_GROUPS.index(group) * 3
        assert totals[offset].item() == pytest.approx(loss_sum)
        assert totals[offset + 1].item() == pytest.approx(mae_sum)
        assert totals[offset + 2].item() == pytest.approx(count)

    def test_missing_speaker_mask_zeroes_everything(self) -> None:
        totals = train.duration_condition_group_totals(
            duration_loss_per_sample=torch.tensor([1.0, 2.0]),
            pred_frames=torch.tensor([1.0, 2.0]),
            target_frames=torch.tensor([3.0, 4.0]),
            has_speaker=None,
        )
        assert totals.abs().sum().item() == pytest.approx(0.0)

    def test_missing_caption_mask_leaves_only_speaker_groups(self) -> None:
        totals = train.duration_condition_group_totals(
            duration_loss_per_sample=torch.tensor([1.0, 2.0]),
            pred_frames=torch.tensor([1.0, 2.0]),
            target_frames=torch.tensor([3.0, 4.0]),
            has_speaker=torch.tensor([True, False]),
        )
        populated = {
            group
            for index, group in enumerate(train.DURATION_CONDITION_GROUPS)
            if totals[index * 3 + 2].item() > 0.0
        }
        assert populated == {"speaker", "no_speaker"}


class TestDurationConditionGroupMetrics:
    def test_averages_by_count(self) -> None:
        metrics = train.duration_condition_group_metrics(_totals())
        assert metrics["duration_loss_speaker"] == pytest.approx(2.0)
        assert metrics["duration_mae_frames_speaker"] == pytest.approx(1.0)
        assert metrics["duration_samples_speaker"] == pytest.approx(2.0)
        assert metrics["duration_loss_caption"] == pytest.approx(1.5)
        assert metrics["duration_mae_frames_caption"] == pytest.approx(2.5)

    def test_empty_group_reports_zeros(self) -> None:
        metrics = train.duration_condition_group_metrics(_totals())
        assert metrics["duration_loss_no_speaker_no_caption"] == pytest.approx(0.0)
        assert metrics["duration_mae_frames_no_speaker_no_caption"] == pytest.approx(0.0)
        assert metrics["duration_samples_no_speaker_no_caption"] == pytest.approx(0.0)

    def test_covers_every_group(self) -> None:
        metrics = train.duration_condition_group_metrics(_totals())
        assert len(metrics) == train.DURATION_CONDITION_GROUP_TOTAL_SIZE


class TestDurationConditionGroupLogSuffix:
    def test_formats_populated_groups_in_order(self) -> None:
        metrics = train.duration_condition_group_metrics(_totals())
        suffix = train.duration_condition_group_log_suffix(metrics)
        assert suffix == (
            "dur_sp=2.000000 mae_sp=1.00 n_sp=2 "
            "dur_no_sp=2.000000 mae_no_sp=3.00 n_no_sp=1 "
            "dur_cap=1.500000 mae_cap=2.50 n_cap=2 "
            "dur_no_cap=3.000000 mae_no_cap=0.00 n_no_cap=1 "
            "dur_sp_cap=1.000000 mae_sp_cap=2.00 n_sp_cap=1 "
            "dur_sp_no_cap=3.000000 mae_sp_no_cap=0.00 n_sp_no_cap=1 "
            "dur_no_sp_cap=2.000000 mae_no_sp_cap=3.00 n_no_sp_cap=1"
        )

    def test_skips_empty_groups(self) -> None:
        metrics = train.duration_condition_group_metrics(_totals())
        assert "no_sp_no_cap" not in train.duration_condition_group_log_suffix(metrics)

    def test_returns_empty_string_when_nothing_populated(self) -> None:
        totals = torch.zeros(train.DURATION_CONDITION_GROUP_TOTAL_SIZE, dtype=torch.float64)
        metrics = train.duration_condition_group_metrics(totals)
        assert train.duration_condition_group_log_suffix(metrics) == ""

    def test_tolerates_missing_keys_for_empty_groups(self) -> None:
        assert train.duration_condition_group_log_suffix({}) == ""


class TestDurationConditionGroupWandbMetrics:
    def test_prefixes_every_metric(self) -> None:
        metrics = train.duration_condition_group_metrics(_totals())
        logged = train.duration_condition_group_wandb_metrics("val", metrics)
        assert len(logged) == train.DURATION_CONDITION_GROUP_TOTAL_SIZE
        assert all(key.startswith("val/") for key in logged)
        assert logged["val/duration_loss_speaker"] == pytest.approx(2.0)
        assert logged["val/duration_samples_no_speaker_no_caption"] == pytest.approx(0.0)

    def test_includes_empty_groups(self) -> None:
        metrics = train.duration_condition_group_metrics(_totals())
        logged = train.duration_condition_group_wandb_metrics("train", metrics)
        expected = {
            f"train/{name}_{group}"
            for group in train.DURATION_CONDITION_GROUPS
            for name in ("duration_loss", "duration_mae_frames", "duration_samples")
        }
        assert set(logged) == expected
