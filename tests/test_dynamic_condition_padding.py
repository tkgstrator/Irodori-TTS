"""Opt-in condition padding keeps tokenization and training outputs unchanged."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from irodori_tts.config import ModelConfig, TrainConfig
from irodori_tts.dataset import TTSCollator
from irodori_tts.lora import apply_lora
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer


class LocalTokenizer:
    """Small HF-compatible tokenizer without downloaded vocabulary or weights."""

    padding_side = "right"
    pad_token_id = 0
    bos_token_id = 1

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [2 + ord(char) % 30 for char in text]

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        width = kwargs["max_length"]
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            tokens = self.encode(text)[:width]
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[row, : len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def make_collator(*, add_bos: bool = True, **kwargs: Any) -> TTSCollator:
    return TTSCollator(
        tokenizer=PretrainedTextTokenizer(LocalTokenizer(), add_bos=add_bos),
        caption_tokenizer=PretrainedTextTokenizer(LocalTokenizer(), add_bos=add_bos),
        latent_dim=4,
        latent_patch_size=1,
        max_text_len=8,
        max_caption_len=12,
        **kwargs,
    )


def make_batch(
    texts: tuple[str, ...], captions: tuple[str, ...], present: tuple[bool, ...]
) -> list[dict[str, Any]]:
    return [
        {
            "text": text,
            "caption": caption,
            "has_caption": has_caption,
            "has_speaker": True,
            "latent": torch.arange(24, dtype=torch.float32).reshape(6, 4) / 24,
            "ref_latent": torch.ones(8, 4),
            "num_frames": 6,
        }
        for text, caption, has_caption in zip(texts, captions, present, strict=True)
    ]


@pytest.mark.parametrize("add_bos", [False, True])
@pytest.mark.parametrize(
    ("texts", "captions", "present"),
    [
        (("a", "abc"), ("", "xy"), (False, True)),
        (("", ""), ("", ""), (False, False)),
        (("", ""), ("", ""), (True, True)),
        (("abcdefghijklmnop", "b"), ("abcdefghijklmnop", ""), (True, False)),
        (("ab", "b"), ("discarded", ""), (False, False)),
    ],
)
def test_trim_preserves_tokens_masks_and_features(
    add_bos: bool, texts: tuple[str, ...], captions: tuple[str, ...], present: tuple[bool, ...]
) -> None:
    collator = make_collator(add_bos=add_bos)
    batch = make_batch(texts, captions, present)
    fixed = collator(batch)
    trimmed = replace(collator, dynamic_condition_padding=True)(batch)
    for prefix in ("text", "caption"):
        ids_key, mask_key = f"{prefix}_ids", f"{prefix}_mask"
        width = trimmed[ids_key].shape[1]
        columns = fixed[mask_key].any(dim=0).nonzero(as_tuple=True)[0]
        expected_width = int(columns[-1]) + 1 if columns.numel() else 1
        assert width == expected_width
        assert torch.equal(trimmed[ids_key], fixed[ids_key][:, :width])
        assert torch.equal(trimmed[mask_key], fixed[mask_key][:, :width])
        assert not fixed[mask_key][:, width:].any()
        assert torch.equal(trimmed[mask_key].sum(dim=1), fixed[mask_key].sum(dim=1))
    for key in fixed.keys() - {"text_ids", "text_mask", "caption_ids", "caption_mask"}:
        assert torch.equal(trimmed[key], fixed[key]), key


def test_padding_is_disabled_by_default() -> None:
    assert not TrainConfig().dynamic_condition_padding
    out = make_collator()(make_batch(("a",), ("b",), (True,)))
    assert out["text_ids"].shape == (1, 8)
    assert out["caption_ids"].shape == (1, 12)


@pytest.mark.parametrize("branch", ["tokenizer", "caption_tokenizer"])
def test_trim_rejects_left_padding(branch: str) -> None:
    collator = make_collator(dynamic_condition_padding=True)
    getattr(collator, branch).tokenizer.padding_side = "left"
    with pytest.raises(ValueError, match="right-padding"):
        collator(make_batch(("a",), ("b",), (True,)))


def test_trim_without_caption_branch() -> None:
    collator = replace(make_collator(dynamic_condition_padding=True), caption_tokenizer=None)
    out = collator(make_batch(("a",), ("",), (False,)))
    assert out["text_ids"].shape == (1, 2)
    assert "caption_ids" not in out


def test_bos_only_at_single_token_limit() -> None:
    collator = replace(make_collator(dynamic_condition_padding=True), max_text_len=1)
    out = collator(make_batch(("long text",), ("",), (False,)))
    assert out["text_ids"].tolist() == [[1]]
    assert out["text_mask"].tolist() == [[True]]
    assert out["caption_ids"].shape == (1, 1)


@pytest.mark.parametrize("drop_conditions", [False, True])
@pytest.mark.parametrize("use_lora", [False, True])
def test_tiny_model_loss_and_gradients_match(drop_conditions: bool, use_lora: bool) -> None:
    torch.manual_seed(7)
    cfg = ModelConfig(
        latent_dim=4,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        text_vocab_size=32,
        text_dim=16,
        text_layers=1,
        text_heads=2,
        speaker_dim=16,
        speaker_layers=1,
        speaker_heads=2,
        speaker_patch_size=2,
        use_caption_condition=True,
        use_speaker_condition=True,
        caption_vocab_size=32,
        caption_dim=16,
        caption_layers=1,
        caption_heads=2,
        timestep_embed_dim=16,
        adaln_rank=4,
        use_duration_predictor=True,
        duration_hidden_dim=16,
        duration_layers=1,
        duration_attention_heads=2,
        duration_dropout=0.0,
        duration_architecture="token_sum_dual_adarn_zero_no_aux",
    )
    model = TextToLatentRFDiT(cfg)
    # Exercise conditioning gradients rather than the decoder's zero-init path.
    with torch.no_grad():
        for parameter in model.parameters():
            if not torch.count_nonzero(parameter):
                torch.nn.init.normal_(parameter, std=0.05)
    if use_lora:
        model = apply_lora(
            model,
            TrainConfig(
                lora_enabled=True,
                lora_r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                lora_target_modules="diffusion_attn",
                lora_modules_to_save="auto",
            ),
        )
    model.train()
    collator = make_collator()
    batch = make_batch(("a", "abc"), ("", "xy"), (False, True))
    noise = torch.randn(2, 6, 4)
    targets = torch.randn_like(noise)
    dropout = torch.full((2,), drop_conditions, dtype=torch.bool)

    def run(out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad(set_to_none=True)
        prediction, duration = model(
            x_t=noise,
            t=torch.tensor([0.2, 0.8]),
            text_input_ids=out["text_ids"],
            text_mask=out["text_mask"],
            ref_latent=out["ref_latent_patched"],
            ref_mask=out["ref_latent_mask_patched"],
            caption_input_ids=out["caption_ids"],
            caption_mask=out["caption_mask"],
            latent_mask=out["latent_mask_patched"],
            duration_features=out["duration_features"],
            duration_has_speaker=out["has_speaker"],
            duration_has_caption=out["has_caption"],
            text_condition_dropout=dropout,
            caption_condition_dropout=dropout,
            speaker_condition_dropout=dropout,
        )
        loss = (prediction - targets).square().mean() + duration.square().mean()
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return loss.detach(), gradients

    fixed_loss, fixed_grads = run(collator(batch))
    trimmed_loss, trimmed_grads = run(replace(collator, dynamic_condition_padding=True)(batch))
    torch.testing.assert_close(trimmed_loss, fixed_loss, rtol=1e-5, atol=1e-6)
    assert fixed_grads.keys() == trimmed_grads.keys()
    if use_lora:
        assert any(gradient.any() for name, gradient in fixed_grads.items() if "lora_B" in name)
    for name, gradient in fixed_grads.items():
        torch.testing.assert_close(trimmed_grads[name], gradient, rtol=1e-4, atol=1e-6)
