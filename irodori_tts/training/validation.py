"""The validation pass over a held-out loader."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader as TorchDataLoader

from irodori_tts.config import TrainConfig
from irodori_tts.duration import set_duration_has_speaker_feature
from irodori_tts.rf import (
    rf_interpolate,
    rf_velocity_target,
    sample_logit_normal_t,
    sample_stratified_logit_normal_t,
)
from irodori_tts.training.duration_metrics import (
    DURATION_CONDITION_GROUP_TOTAL_SIZE,
    duration_condition_group_metrics,
    duration_condition_group_totals,
)
from irodori_tts.training.losses import compute_rf_loss


def run_validation(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    model,
    loader: TorchDataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
    use_bf16: bool,
    distributed: bool,
) -> dict[str, float]:
    was_training = model.training
    model_cfg = model.module.cfg if isinstance(model, DDP) else model.cfg
    duration_only = train_cfg.train_mode == "duration_only"
    model.eval()
    try:
        totals = torch.zeros(
            6 + DURATION_CONDITION_GROUP_TOTAL_SIZE,
            device=device,
            dtype=torch.float64,
        )

        with torch.no_grad():
            for batch in loader:
                text_ids = batch["text_ids"].to(device, non_blocking=True)
                text_mask = batch["text_mask"].to(device, non_blocking=True)
                caption_ids = None
                caption_mask = None
                has_caption = None
                if model_cfg.use_caption_condition:
                    caption_ids = batch["caption_ids"].to(device, non_blocking=True)
                    caption_mask = batch["caption_mask"].to(device, non_blocking=True)
                    has_caption = batch["has_caption"].to(device, non_blocking=True)
                num_frames = batch["num_frames"].to(device, non_blocking=True)
                duration_features = batch["duration_features"].to(device, non_blocking=True)
                ref_latent = None
                ref_mask = None
                if model_cfg.use_speaker_condition_resolved:
                    ref_latent = batch["ref_latent_patched"].to(device, non_blocking=True)
                    ref_mask = batch["ref_latent_mask_patched"].to(device, non_blocking=True)
                    has_speaker = batch["has_speaker"].to(device, non_blocking=True)
                else:
                    has_speaker = None

                bsz = text_ids.shape[0]
                x0 = None
                x_mask = None
                x_mask_valid = None
                x_t = None
                t = None
                v_target = None
                if not duration_only:
                    x0 = batch["latent_patched"].to(device, non_blocking=True)
                    x_mask = batch["latent_mask_patched"].to(device, non_blocking=True)
                    x_mask_valid = batch["latent_mask_valid_patched"].to(device, non_blocking=True)
                    if train_cfg.timestep_stratified:
                        t = sample_stratified_logit_normal_t(
                            batch_size=bsz,
                            device=device,
                            mean=train_cfg.timestep_logit_mean,
                            std=train_cfg.timestep_logit_std,
                            t_min=train_cfg.timestep_min,
                            t_max=train_cfg.timestep_max,
                        )
                    else:
                        t = sample_logit_normal_t(
                            batch_size=bsz,
                            device=device,
                            mean=train_cfg.timestep_logit_mean,
                            std=train_cfg.timestep_logit_std,
                            t_min=train_cfg.timestep_min,
                            t_max=train_cfg.timestep_max,
                        )
                    noise = torch.randn_like(x0)
                    x_t = rf_interpolate(x0, noise, t)
                    v_target = rf_velocity_target(x0, noise)

                if model_cfg.use_speaker_condition_resolved:
                    if train_cfg.speaker_inversion_enabled:
                        # Speaker Inversion learns one embedding for this run, so validation
                        # should match training and treat every sample as speaker-conditioned.
                        use_speaker = torch.ones((bsz,), device=device, dtype=torch.bool)
                    else:
                        use_speaker = has_speaker
                    speaker_condition_dropout = ~use_speaker
                    duration_has_speaker = use_speaker
                    duration_features = set_duration_has_speaker_feature(
                        duration_features,
                        duration_has_speaker,
                    )
                else:
                    speaker_condition_dropout = None
                    duration_has_speaker = None
                duration_has_caption = has_caption if model_cfg.use_caption_condition else None

                with (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16
                    else nullcontext()
                ):
                    if duration_only:
                        duration_pred = model(
                            x_t=None,
                            t=None,
                            text_input_ids=text_ids,
                            text_mask=text_mask,
                            ref_latent=ref_latent,
                            ref_mask=ref_mask,
                            caption_input_ids=caption_ids,
                            caption_mask=caption_mask,
                            latent_mask=None,
                            duration_features=duration_features,
                            duration_has_speaker=duration_has_speaker,
                            duration_has_caption=duration_has_caption,
                            duration_only=True,
                        )
                        v_pred = None
                    elif model_cfg.use_duration_predictor:
                        v_pred, duration_pred = model(
                            x_t=x_t,
                            t=t,
                            text_input_ids=text_ids,
                            text_mask=text_mask,
                            ref_latent=ref_latent,
                            ref_mask=ref_mask,
                            caption_input_ids=caption_ids,
                            caption_mask=caption_mask,
                            latent_mask=x_mask,
                            speaker_condition_dropout=speaker_condition_dropout,
                            duration_features=duration_features,
                            duration_has_speaker=duration_has_speaker,
                            duration_has_caption=duration_has_caption,
                        )
                    else:
                        if model_cfg.use_speaker_condition_resolved:
                            ref_mask = ref_mask & use_speaker[:, None]
                            ref_latent = ref_latent * use_speaker[:, None, None].to(
                                ref_latent.dtype
                            )
                        v_pred = model(
                            x_t=x_t,
                            t=t,
                            text_input_ids=text_ids,
                            text_mask=text_mask,
                            ref_latent=ref_latent,
                            ref_mask=ref_mask,
                            caption_input_ids=caption_ids,
                            caption_mask=caption_mask,
                            latent_mask=x_mask,
                        )
                        duration_pred = None

                rf_loss = torch.zeros((), device=device, dtype=torch.float32)
                if not duration_only:
                    if v_pred is None or v_target is None or x_mask is None or x_mask_valid is None:
                        raise RuntimeError("RF validation tensors are missing.")
                    v_pred = v_pred.float()
                    rf_loss = compute_rf_loss(
                        pred=v_pred,
                        target=v_target.float(),
                        loss_mask=x_mask,
                        valid_mask=x_mask_valid,
                        mode=train_cfg.rf_loss_mode,
                    )
                duration_loss = torch.zeros((), device=device, dtype=torch.float32)
                duration_mae_frames = torch.zeros((), device=device, dtype=torch.float32)
                if model_cfg.use_duration_predictor:
                    if duration_pred is None:
                        raise RuntimeError(
                            "Duration predictor is enabled but duration_pred is missing."
                        )
                    duration_target = torch.log1p(num_frames.float())
                    duration_loss_per_sample = F.huber_loss(
                        duration_pred.float(),
                        duration_target,
                        delta=float(train_cfg.duration_huber_delta),
                        reduction="none",
                    )
                    duration_loss = duration_loss_per_sample.mean()
                    pred_frames = torch.expm1(duration_pred.float()).clamp_min(0.0)
                    duration_mae_frames = (pred_frames - num_frames.float()).abs().mean()
                    if duration_only:
                        totals[6:] += duration_condition_group_totals(
                            duration_loss_per_sample=duration_loss_per_sample,
                            pred_frames=pred_frames,
                            target_frames=num_frames.float(),
                            has_speaker=has_speaker,
                            has_caption=has_caption,
                        )
                if duration_only:
                    loss = duration_loss
                else:
                    loss = rf_loss + (float(train_cfg.duration_loss_weight) * duration_loss)

                weight = float(bsz)
                totals[0] += loss.detach().double() * weight
                totals[1] += rf_loss.detach().double() * weight
                totals[2] += duration_loss.detach().double() * weight
                totals[3] += duration_mae_frames.detach().double() * weight
                totals[4] += float(num_frames.detach().float().mean().item()) * weight
                totals[5] += weight

        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        denom = max(float(totals[5].item()), 1.0)
        metrics = {
            "loss": float(totals[0].item() / denom),
            "rf_loss": float(totals[1].item() / denom),
            "duration_loss": float(totals[2].item() / denom),
            "duration_mae_frames": float(totals[3].item() / denom),
            "target_frames_mean": float(totals[4].item() / denom),
            "num_samples": float(totals[5].item()),
        }
        if duration_only:
            metrics.update(duration_condition_group_metrics(totals[6:]))
    finally:
        model.train(was_training)
    return metrics
