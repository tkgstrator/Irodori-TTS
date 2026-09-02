"""Runtime registry holding the LoRA runtime and the optional caption runtime."""

from __future__ import annotations

import logging
import threading

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey
from irodori_tts.server.config import (
    ServerConfig,
    SpeakerSpec,
    _resolve_checkpoint,
    resolve_base_checkpoint,
)

logger = logging.getLogger("irodori_tts.server")


class RuntimeRegistry:
    """LoRA runtime + optional caption (VoiceDesign) runtime."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._by_uuid: dict[str, SpeakerSpec] = {s.uuid: s for s in cfg.speakers}
        self._runtime: InferenceRuntime | None = None
        self._caption_runtime: InferenceRuntime | None = None
        self._lock = threading.Lock()

    def list_speakers(self) -> list[SpeakerSpec]:
        return list(self.cfg.speakers)

    def get_spec(self, uuid: str) -> SpeakerSpec:
        spec = self._by_uuid.get(uuid)
        if spec is None:
            raise KeyError(uuid)
        return spec

    @property
    def caption_available(self) -> bool:
        return self._caption_runtime is not None

    @property
    def sample_rate(self) -> int | None:
        """Codec sample rate of whichever runtime is loaded, or None if neither is."""
        for rt in (self._runtime, self._caption_runtime):
            if rt is not None:
                return int(rt.codec.sample_rate)
        return None

    def _make_key(self, checkpoint: str) -> RuntimeKey:
        return RuntimeKey(
            checkpoint=checkpoint,
            model_device=self.cfg.model_device,
            codec_repo=self.cfg.codec_repo,
            model_precision=self.cfg.model_precision,
            codec_device=self.cfg.codec_device,
            codec_precision=self.cfg.codec_precision,
            codec_deterministic_encode=self.cfg.codec_deterministic_encode,
            codec_deterministic_decode=self.cfg.codec_deterministic_decode,
            compile_model=False,
            compile_dynamic=False,
        )

    def load(self) -> None:
        if self.cfg.speakers:
            base_path = resolve_base_checkpoint(self.cfg)
            adapters = {s.uuid: s.adapter for s in self.cfg.speakers}
            logger.info("Loading base + %d LoRA adapters", len(adapters))
            self._runtime = InferenceRuntime.from_base_with_adapters(
                key=self._make_key(str(base_path)),
                adapters=adapters,
                default_adapter=self.cfg.speakers[0].uuid,
            )
        else:
            logger.warning("No LoRA speakers configured — LoRA synthesis disabled")

        if self.cfg.caption_checkpoint or self.cfg.caption_hf_repo:
            caption_path = _resolve_checkpoint(
                self.cfg.caption_checkpoint,
                self.cfg.caption_hf_repo,
                self.cfg.caption_hf_filename,
                "caption",
            )
            logger.info("Loading caption (VoiceDesign) runtime")
            self._caption_runtime = InferenceRuntime.from_key(self._make_key(str(caption_path)))

    def acquire(self, uuid: str) -> tuple[InferenceRuntime, SpeakerSpec]:
        spec = self.get_spec(uuid)
        if self._runtime is None:
            raise RuntimeError("Registry not loaded. Call load() first.")
        with self._lock:
            self._runtime.set_active_adapter(uuid)
            return self._runtime, spec

    def acquire_caption(self) -> InferenceRuntime:
        if self._caption_runtime is None:
            raise RuntimeError("Caption runtime not configured.")
        return self._caption_runtime
