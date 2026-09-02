"""Rectified-flow training losses."""

from __future__ import annotations

import torch


def echo_style_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Echo/JAX-style diffusion loss:
    - take mean squared error over loss_masked tokens
    - divide by mean valid-token ratio (short samples get up-weighted)

    If loss_mask == valid_mask, this reduces to standard masked MSE.
    """
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1)  # (B, S)
    loss_weight = loss_mask.float()
    valid_weight = valid_mask.float()

    # Keep normalization stable for degenerate samples with no valid target tokens.
    has_valid = (valid_weight.sum(dim=-1) > 0).float()[:, None]
    denom = (loss_weight * valid_weight * has_valid).mean().clamp_min(1e-6)
    # Zero-fill masked positions instead of multiplying by the mask afterwards:
    # NaN * 0.0 is NaN, so a NaN behind the mask would poison the whole loss.
    # Kept positions are untouched, so the unmasked numerics are unchanged.
    masked_diff = torch.where(loss_mask.bool(), diff, torch.zeros_like(diff))
    return masked_diff.mean() / denom


def utterance_mean_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1)
    weight = valid_mask.float()
    # Zero-fill masked positions instead of multiplying by the mask afterwards:
    # NaN * 0.0 is NaN, so a NaN behind the mask would poison the whole loss.
    masked_diff = torch.where(valid_mask.bool(), diff, torch.zeros_like(diff))
    per_sample = masked_diff.sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
    return per_sample.mean()


def compute_rf_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "echo":
        return echo_style_masked_mse(
            pred,
            target,
            loss_mask=loss_mask,
            valid_mask=valid_mask,
        )
    if mode == "utterance_mean":
        return utterance_mean_masked_mse(pred, target, valid_mask=valid_mask)
    raise ValueError(f"Unsupported rf_loss_mode={mode!r}. Expected 'echo' or 'utterance_mean'.")
