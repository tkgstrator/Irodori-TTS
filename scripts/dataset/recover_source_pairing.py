#!/usr/bin/env python3
"""Recover which source clip each manifest record came from.

The magical-girl speakers were preprocessed and encoded, then the intermediate
metadata.jsonl that paired each curated transcript with its source audio file
was discarded. The transcripts survive inside the archives' manifest.jsonl, but
nothing there names the audio it came from.

Preprocessing dropped clips and trimmed silence, so durations do not match and
indices do not line up. It did preserve order, though, which is enough: decode
each latent back to audio, score it against every source clip by normalized
cross-correlation over log-mel frames, then pick the assignment with a
monotonic dynamic program rather than per-record argmax. Individual ties (two
takes of one line) are resolved by their neighbours.

Usage:
  uv run --no-sync python scripts/dataset/recover_source_pairing.py \\
    --speaker anan --character AnAn \\
    --repo-id ultemica/magical-girl-witch-trials-voice \\
    --manifest data/anan/manifest.jsonl \\
    --out data/anan/pairing.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F  # noqa: N812
import torchaudio
from tqdm import tqdm

MEL_SAMPLE_RATE = 16000
MEL = torchaudio.transforms.MelSpectrogram(MEL_SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=40)
NEG_INF = -1e9


def _mel(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate != MEL_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, MEL_SAMPLE_RATE)
    spec = torch.log1p(MEL(waveform)).squeeze(0)
    return spec - spec.mean(1, keepdim=True)


def _mel_from_bytes(raw: bytes) -> torch.Tensor:
    data, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
    if data.ndim > 1:
        data = data.mean(1)
    return _mel(torch.from_numpy(data)[None], sample_rate)


class _SourceBank:
    """All source clips laid end to end so one convolution scores them together.

    Scoring clip by clip means a kernel launch per pair, which the larger
    speakers cannot afford. Concatenating lets each query run as a single
    convolution; windows straddling a boundary are simply never read back.
    """

    def __init__(self, mels: list[torch.Tensor], device: torch.device) -> None:
        self.lengths = [m.shape[1] for m in mels]
        self.starts: list[int] = []
        offset = 0
        for length in self.lengths:
            self.starts.append(offset)
            offset += length
        self.frames = torch.cat(mels, dim=1).to(device)
        self.energy = (self.frames**2).sum(0, keepdim=True)[None]
        self.device = device

    def scores(self, query: torch.Tensor) -> torch.Tensor:
        """Best NCC of `query` within each clip; NEG_INF where the clip is too short."""
        span = query.shape[1]
        out = torch.full((len(self.lengths),), NEG_INF, device=self.device)
        unit = (query / (query.norm() + 1e-8)).to(self.device)
        numerator = F.conv1d(self.frames[None], unit[None]).squeeze()
        window = torch.ones(1, 1, span, device=self.device)
        denominator = F.conv1d(self.energy, window).squeeze().clamp_min(1e-8).sqrt()
        correlation = numerator / denominator
        for i, (start, length) in enumerate(zip(self.starts, self.lengths, strict=True)):
            if length < span:
                continue
            out[i] = correlation[start : start + length - span + 1].max()
        return out.cpu()


def _monotonic_assignment(scores: torch.Tensor) -> list[int]:
    """Assign each record a distinct, strictly increasing source index.

    scores is (records, sources). Maximises the total score subject to the
    order preprocessing already guaranteed, so a record that is ambiguous on
    its own is decided by the records around it.
    """
    n_rec, n_src = scores.shape
    best = torch.full((n_rec + 1, n_src + 1), NEG_INF)
    best[n_rec, :] = 0.0
    choice = torch.zeros((n_rec, n_src + 1), dtype=torch.long)
    for rec in range(n_rec - 1, -1, -1):
        for src in range(n_src - 1, -1, -1):
            skip = best[rec, src + 1]
            take = scores[rec, src] + best[rec + 1, src + 1]
            if take >= skip:
                best[rec, src], choice[rec, src] = take, 1
            else:
                best[rec, src], choice[rec, src] = skip, 0
    out: list[int] = []
    src = 0
    for rec in range(n_rec):
        while choice[rec, src] == 0:
            src += 1
        out.append(src)
        src += 1
    return out


def _load_source_rows(repo_id: str, character: str, token: str) -> list[dict]:
    import pyarrow.parquet as pq

    api = f"https://huggingface.co/api/datasets/{repo_id}"
    request = urllib.request.Request(api, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request) as response:
        meta = json.load(response)
    shards = sorted(
        s["rfilename"]
        for s in meta["siblings"]
        if s["rfilename"].startswith(f"data/{character}/") and s["rfilename"].endswith(".parquet")
    )
    if not shards:
        sys.exit(f"no parquet shards under data/{character}/ in {repo_id}")

    rows: list[dict] = []
    for shard in shards:
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{shard}"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request) as response:
            table = pq.read_table(io.BytesIO(response.read()))
        rows.extend(table.to_pylist())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", required=True, help="local speaker id, e.g. anan")
    parser.add_argument("--character", required=True, help="parquet subdir, e.g. AnAn")
    parser.add_argument("--repo-id", default="ultemica/magical-girl-witch-trials-voice")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--min-ncc",
        type=float,
        default=0.6,
        help="records scoring below this are flagged for review",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is required to read the source dataset")

    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows = _load_source_rows(args.repo_id, args.character, token)
    print(f"{args.speaker}: {len(records)} records against {len(rows)} source clips")

    source_mels = [_mel_from_bytes(row["audio"]["bytes"]) for row in tqdm(rows, desc="source mel")]

    from irodori_tts.codec import DACVAECodec

    codec = DACVAECodec.load(device=args.device)
    latent_root = args.manifest.parent

    bank = _SourceBank(source_mels, torch.device(args.device))
    scores = torch.full((len(records), len(rows)), NEG_INF)
    for i, record in enumerate(tqdm(records, desc="decode+score")):
        latent = torch.load(latent_root / record["latent_path"], map_location=args.device)
        with torch.no_grad():
            audio = codec.decode_latent(latent.unsqueeze(0).float())
        query = _mel(audio.squeeze()[None].float().cpu(), codec.sample_rate)
        scores[i] = bank.scores(query)

    assignment = _monotonic_assignment(scores)

    flagged = 0
    with args.out.open("w") as handle:
        for i, (record, source_index) in enumerate(zip(records, assignment, strict=True)):
            ncc = float(scores[i, source_index])
            ranked = scores[i].sort(descending=True).values
            entry = {
                "latent_path": record["latent_path"],
                "text": record["text"],
                "source_index": source_index,
                "source_path": rows[source_index]["source_path"],
                "ncc": round(ncc, 4),
                "margin": round(float(ranked[0] - ranked[1]), 4),
                "is_argmax": bool(int(scores[i].argmax()) == source_index),
            }
            if ncc < args.min_ncc:
                entry["review"] = True
                flagged += 1
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    picked = torch.tensor([scores[i, s] for i, s in enumerate(assignment)])
    forced = sum(1 for i, s in enumerate(assignment) if int(scores[i].argmax()) != s)
    print(f"wrote {args.out}")
    print(f"  ncc      min={picked.min():.3f} mean={picked.mean():.3f}")
    print(f"  below {args.min_ncc}: {flagged}")
    print(f"  decided by order rather than argmax: {forced}")


if __name__ == "__main__":
    main()
