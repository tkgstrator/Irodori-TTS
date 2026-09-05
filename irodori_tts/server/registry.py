"""Runtime registry holding the LoRA runtime and the optional caption runtime."""

from __future__ import annotations

import logging
import os
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
    """LoRA runtime + caption support, from the base checkpoint or a separate VoiceDesign one."""

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

    def _base_caption_runtime(self) -> InferenceRuntime | None:
        """Base runtime, when its checkpoint carries caption conditioning (v4 and later)."""
        base = self._runtime
        if base is not None and base.model_cfg.use_caption_condition:
            return base
        return None

    @property
    def caption_available(self) -> bool:
        return self._caption_runtime is not None or self._base_caption_runtime() is not None

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

    def _resolve_slots(self, total: int) -> int:
        """Resident budget: the env override first, then the config, 0 = all.

        TTS_MAX_LOADED_ADAPTERS predates lora_slots (it came from a production
        hot-patch) and deployments still set it; honoring it keeps them working.
        """
        raw = (os.environ.get("TTS_MAX_LOADED_ADAPTERS") or "").strip()
        slots = int(self.cfg.lora_slots)
        if raw:
            try:
                slots = int(raw)
            except ValueError:
                logger.warning(
                    "invalid TTS_MAX_LOADED_ADAPTERS=%r — using lora_slots=%d", raw, slots
                )
        slots = max(0, slots)
        return slots if slots and slots < total else 0

    def _preload_spec(self) -> SpeakerSpec:
        """Speaker loaded (and pinned) at startup — the configured one, else the first.

        The pinned speaker is never evicted: a bot serves most requests with
        its default voice, and evicting it only to reload it moments later
        would make the common case pay for the rare one.
        """
        wanted = (
            os.environ.get("TTS_PRELOAD_SPEAKER_ID") or self.cfg.preload_speaker or ""
        ).strip()
        if wanted:
            spec = self._by_uuid.get(wanted)
            if spec is not None:
                return spec
            logger.warning(
                "preload speaker %s matches none of the %d discovered speakers "
                "— preloading the first one instead",
                wanted,
                len(self.cfg.speakers),
            )
        return self.cfg.speakers[0]

    def load(self) -> None:
        if self.cfg.speakers:
            base_path = resolve_base_checkpoint(self.cfg)
            adapters = {s.uuid: s.adapter for s in self.cfg.speakers}
            slots = self._resolve_slots(len(adapters))
            preload = self._preload_spec()
            if slots:
                logger.info(
                    "Loading base + 1 of %d LoRA adapters (%d resident at a time, pinned: %s)",
                    len(adapters),
                    slots,
                    preload.name,
                )
            else:
                logger.info("Loading base + %d LoRA adapters", len(adapters))
            self._runtime = InferenceRuntime.from_base_with_adapters(
                key=self._make_key(str(base_path)),
                adapters=adapters,
                default_adapter=preload.uuid,
                adapter_slots=slots,
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
        elif self._base_caption_runtime() is not None:
            logger.info("Caption conditioning served by the base checkpoint")

    def acquire(self, uuid: str) -> tuple[InferenceRuntime, SpeakerSpec]:
        spec = self.get_spec(uuid)
        if self._runtime is None:
            raise RuntimeError("Registry not loaded. Call load() first.")
        with self._lock:
            self._runtime.set_active_adapter(uuid)
            return self._runtime, spec

    def acquire_caption(self) -> InferenceRuntime:
        if self._caption_runtime is not None:
            return self._caption_runtime
        base = self._base_caption_runtime()
        if base is None:
            raise RuntimeError("Caption runtime not configured.")
        return base
