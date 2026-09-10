"""Audio encoding for the six OpenAI ``response_format`` values, and the SSE framing."""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import soundfile as sf
import torch
import torchaudio

from irodori_tts.server.errors import ApiError

ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

MEDIA_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# OpenAI documents `pcm` as 24 kHz 16-bit mono. Headerless bytes carry no rate of
# their own, so that one format is resampled when the codec runs at another rate;
# the container formats state their rate and are left at the model's.
PCM_SAMPLE_RATE = 24000

# Written by the libsndfile bundled in the soundfile wheel — MP3 needs 1.1 or
# newer, which is why pyproject floors soundfile at 0.13.
_SOUNDFILE_ARGS: dict[str, dict[str, str]] = {
    "wav": {"format": "WAV", "subtype": "PCM_16"},
    "flac": {"format": "FLAC", "subtype": "PCM_16"},
    "mp3": {"format": "MP3"},
    "opus": {"format": "OGG", "subtype": "OPUS"},
}

_SSE_CHUNK_BYTES = 32 * 1024


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _resample(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate:
        return audio
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return torchaudio.functional.resample(tensor, sample_rate, target_rate).numpy()


def _encode_aac(audio: np.ndarray, sample_rate: int) -> bytes:
    """AAC through ffmpeg — libsndfile cannot write it, and ffmpeg is in the image."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ApiError(
            "response_format 'aac' requires ffmpeg, which this server does not have. "
            "Use mp3, opus, flac, wav or pcm.",
            param="response_format",
        )
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-f",
            "adts",
            "pipe:1",
        ],
        input=to_pcm16(audio),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ApiError(
            f"aac encoding failed: {proc.stderr.decode('utf-8', 'replace').strip()}",
            status_code=500,
            error_type="server_error",
        )
    return proc.stdout


def encode(audio: np.ndarray, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    """Encode mono float32 audio, returning the bytes and their media type."""
    if response_format == "pcm":
        return to_pcm16(_resample(audio, sample_rate, PCM_SAMPLE_RATE)), MEDIA_TYPES["pcm"]
    if response_format == "aac":
        return _encode_aac(audio, sample_rate), MEDIA_TYPES["aac"]
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, **_SOUNDFILE_ARGS[response_format])
    return buf.getvalue(), MEDIA_TYPES[response_format]


def sse_events(audio_bytes: bytes, usage: dict[str, int]) -> Iterator[str]:
    """Frame finished audio as speech.audio.delta events followed by speech.audio.done.

    Synthesis produces the whole utterance at once, so the deltas are slices of a
    finished file rather than a head start. Concatenating them yields exactly the
    bytes a non-streaming request returns, which is what the format promises.
    """
    for start in range(0, len(audio_bytes), _SSE_CHUNK_BYTES):
        chunk = audio_bytes[start : start + _SSE_CHUNK_BYTES]
        yield _sse_line(
            {"type": "speech.audio.delta", "audio": base64.b64encode(chunk).decode("ascii")}
        )
    yield _sse_line({"type": "speech.audio.done", "usage": usage})


def _sse_line(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
