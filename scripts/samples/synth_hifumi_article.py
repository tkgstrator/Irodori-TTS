#!/usr/bin/env python3
"""Synthesize a multi-sentence Japanese passage with the hifumi LoRA and concatenate."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)

CHECKPOINT = "outputs/hifumi_lora/merged_final.safetensors"
OUT_DIR = Path("outputs/hifumi_lora/samples_article_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENTENCES = [
    "将棋の棋士は、順位戦の所属クラスがＡ～Ｃ級まであります。",
    "過去の名人経験者には、Ａ級に在籍したまま引退した方もいます。",
    "最高峰を究めた棋士が、降級して下のクラスで戦うのはどうなのか、という美学もあったのかもしれません。",
    "これは全く、それぞれの個人の選択です。",
    "私は６２歳までＡ級に通算３６期在籍した後、Ｂ級やＣ級でも戦い、Ｃ級からの陥落に伴って規定により引退となった７７歳まで指し続けました。",
    "何より「勝てる」「また上に」という気持ちが強かったからです。",
    "実際、６２歳まではＢ級からＡ級に４回復帰していました。",
    "その時々の勝ち負けはあっても、性格上、相手に歯が立たないと思ったことはないのです。",
    "それに将棋界は、名人こそＡ級で勝たないと挑めませんが、他の棋戦はＣ級でも出られ、引き続きタイトル獲得のチャンスがありました。",
]

SILENCE_SEC = 0.35


def main() -> None:
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=CHECKPOINT,
            model_device="cuda",
            model_precision="bf16",
            codec_device="cuda",
            codec_precision="bf16",
        )
    )

    clips: list[torch.Tensor] = []
    sr: int | None = None
    for i, text in enumerate(SENTENCES):
        print(f"[{i+1}/{len(SENTENCES)}] synth: {text[:40]}...")
        result = runtime.synthesize(
            SamplingRequest(
                text=text,
                no_ref=True,
                num_candidates=1,
                decode_mode="sequential",
                num_steps=40,
                cfg_scale_text=3.0,
                cfg_scale_speaker=5.0,
                cfg_guidance_mode="independent",
                seed=42,
                seconds=30.0,
                trim_tail=True,
            )
        )
        audio = result.audio.detach().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        sr = int(result.sample_rate)
        part_path = OUT_DIR / f"part_{i:02d}.wav"
        save_wav(part_path, audio, sr)
        clips.append(audio)

    assert sr is not None
    silence = torch.zeros(clips[0].shape[0], int(sr * SILENCE_SEC))
    concat_parts: list[torch.Tensor] = []
    for i, clip in enumerate(clips):
        if i > 0:
            concat_parts.append(silence)
        concat_parts.append(clip)
    merged = torch.cat(concat_parts, dim=-1)
    merged_path = OUT_DIR / "hifumi_article.wav"
    save_wav(merged_path, merged, sr)
    total_sec = merged.shape[-1] / sr
    print(f"done. merged={merged_path} duration={total_sec:.1f}s sample_rate={sr}")


if __name__ == "__main__":
    main()
