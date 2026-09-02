"""Audio post-processing helpers for the TTS server."""

from __future__ import annotations

import numpy as np

_FADE_MS = 50


def _apply_fade(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    fade_samples = int(sample_rate * _FADE_MS / 1000)
    if fade_samples <= 0 or len(audio) < fade_samples * 2:
        return audio
    audio = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio.dtype)
    audio[:fade_samples] *= fade_in
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=audio.dtype)
    audio[-fade_samples:] *= fade_out
    return audio
