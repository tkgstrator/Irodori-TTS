"""Runtime registry holding the base model and its LoRA speaker adapters."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey
from irodori_tts.server.config import ServerConfig, SpeakerSpec, resolve_base_checkpoint

logger = logging.getLogger("irodori_tts.server")


class RuntimeRegistry:
    """One base runtime with the speaker adapters attached to it."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._by_uuid: dict[str, SpeakerSpec] = {s.uuid: s for s in cfg.speakers}
        self._runtime: InferenceRuntime | None = None
        self._lock = threading.Lock()

    def list_speakers(self) -> list[SpeakerSpec]:
        return list(self.cfg.speakers)

    def get_spec(self, uuid: str) -> SpeakerSpec:
        spec = self._by_uuid.get(uuid)
        if spec is None:
            raise KeyError(uuid)
        return spec

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
        if not self.cfg.speakers:
            # Every voice is a LoRA adapter, so a base with none attached has
            # nothing it could synthesize. Skip the load and let requests 400.
            logger.warning("No LoRA speakers configured — nothing to load")
            return

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

        if not self.cfg.enable_watermark:
            # The runtime watermarks whenever the SilentCipher backend loaded, with
            # no flag of its own, so dropping the backend is how the server opts out.
            logger.info("Watermarking disabled by config")
            self._runtime.watermarker.model = None

    @contextmanager
    def acquire(self, uuid: str) -> Iterator[tuple[InferenceRuntime, SpeakerSpec]]:
        """Activate `uuid`'s adapter and hold the runtime exclusively for the
        whole `with` block.

        The runtime is a single instance shared by every speaker — only the
        active LoRA adapter differs — so the lock must stay held through the
        entire synthesis call. Releasing it right after set_active_adapter()
        (as a plain getter would) lets a concurrent request swap the adapter
        mid-inference: the in-flight request ends up speaking as whichever
        speaker most recently grabbed the lock, and the underlying HF
        tokenizer (not reentrant) can also throw "Already borrowed" under the
        same race. Callers must do their synthesis inside this block, never
        after it returns.
        """
        spec = self.get_spec(uuid)
        if self._runtime is None:
            raise RuntimeError("Registry not loaded. Call load() first.")
        with self._lock:
            self._runtime.set_active_adapter(uuid)
            yield self._runtime, spec
