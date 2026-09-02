"""Per-condition-group duration metrics and their log/wandb formatting."""

from __future__ import annotations

import torch

DURATION_CONDITION_GROUPS = (
    "speaker",
    "no_speaker",
    "caption",
    "no_caption",
    "speaker_caption",
    "speaker_no_caption",
    "no_speaker_caption",
    "no_speaker_no_caption",
)
DURATION_CONDITION_GROUP_TOTAL_SIZE = len(DURATION_CONDITION_GROUPS) * 3


def duration_condition_group_totals(
    *,
    duration_loss_per_sample: torch.Tensor,
    pred_frames: torch.Tensor,
    target_frames: torch.Tensor,
    has_speaker: torch.Tensor | None,
    has_caption: torch.Tensor | None = None,
) -> torch.Tensor:
    totals = torch.zeros(
        DURATION_CONDITION_GROUP_TOTAL_SIZE,
        device=duration_loss_per_sample.device,
        dtype=torch.float64,
    )
    device = duration_loss_per_sample.device
    speaker_mask = (
        has_speaker.to(device=device, dtype=torch.bool) if has_speaker is not None else None
    )
    caption_mask = (
        has_caption.to(device=device, dtype=torch.bool) if has_caption is not None else None
    )
    mae_per_sample = (pred_frames.float() - target_frames.float()).abs()

    group_masks: dict[str, torch.Tensor | None] = {
        "speaker": speaker_mask,
        "no_speaker": None if speaker_mask is None else ~speaker_mask,
        "caption": caption_mask,
        "no_caption": None if caption_mask is None else ~caption_mask,
        "speaker_caption": None
        if speaker_mask is None or caption_mask is None
        else speaker_mask & caption_mask,
        "speaker_no_caption": None
        if speaker_mask is None or caption_mask is None
        else speaker_mask & (~caption_mask),
        "no_speaker_caption": None
        if speaker_mask is None or caption_mask is None
        else (~speaker_mask) & caption_mask,
        "no_speaker_no_caption": None
        if speaker_mask is None or caption_mask is None
        else (~speaker_mask) & (~caption_mask),
    }

    for group_index, group_name in enumerate(DURATION_CONDITION_GROUPS):
        mask = group_masks[group_name]
        if mask is None or not mask.any():
            continue
        offset = group_index * 3
        totals[offset] = duration_loss_per_sample[mask].detach().double().sum()
        totals[offset + 1] = mae_per_sample[mask].detach().double().sum()
        totals[offset + 2] = mask.sum().double()
    return totals


def duration_condition_group_metrics(totals: torch.Tensor) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for group_index, group_name in enumerate(DURATION_CONDITION_GROUPS):
        offset = group_index * 3
        count = max(float(totals[offset + 2].item()), 0.0)
        metrics[f"duration_loss_{group_name}"] = (
            float(totals[offset].item() / count) if count > 0.0 else 0.0
        )
        metrics[f"duration_mae_frames_{group_name}"] = (
            float(totals[offset + 1].item() / count) if count > 0.0 else 0.0
        )
        metrics[f"duration_samples_{group_name}"] = count
    return metrics


def duration_condition_group_log_suffix(metrics: dict[str, float]) -> str:
    groups = [
        ("sp", "speaker"),
        ("no_sp", "no_speaker"),
        ("cap", "caption"),
        ("no_cap", "no_caption"),
        ("sp_cap", "speaker_caption"),
        ("sp_no_cap", "speaker_no_caption"),
        ("no_sp_cap", "no_speaker_caption"),
        ("no_sp_no_cap", "no_speaker_no_caption"),
    ]
    chunks: list[str] = []
    for label, group in groups:
        count = metrics.get(f"duration_samples_{group}", 0.0)
        if count <= 0.0:
            continue
        chunks.append(
            "{}={:.6f} mae_{}={:.2f} n_{}={:.0f}".format(
                f"dur_{label}",
                metrics[f"duration_loss_{group}"],
                label,
                metrics[f"duration_mae_frames_{group}"],
                label,
                count,
            )
        )
    return " ".join(chunks)


def duration_condition_group_wandb_metrics(
    prefix: str,
    metrics: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for group_name in DURATION_CONDITION_GROUPS:
        for metric_name in (
            f"duration_loss_{group_name}",
            f"duration_mae_frames_{group_name}",
            f"duration_samples_{group_name}",
        ):
            out[f"{prefix}/{metric_name}"] = metrics[metric_name]
    return out
