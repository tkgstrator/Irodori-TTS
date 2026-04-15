"""FastAPI TTS server for Irodori-TTS speaker LoRAs."""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import threading
import uuid as uuid_lib
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    resolve_cfg_scales,
)
from irodori_tts.lora import (
    is_lora_adapter_dir,
    is_lora_safetensors_file,
    read_lora_safetensors_metadata,
)

FIXED_SECONDS = 30.0

# Namespace for deterministic UUIDv5 derivation from LoRA filenames. Lets users
# drop .safetensors files into models/LoRA/ without assigning ids manually.
_LORA_UUID_NAMESPACE = uuid_lib.UUID("8e6d8a0e-5a52-4a1e-8c8d-4c3e2f6a1b9f")

logger = logging.getLogger("tts_server")


_DEMO_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Irodori-TTS Demo</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:680px;margin:2rem auto;padding:0 1rem;}
 label{display:block;margin-top:1rem;font-weight:600;}
 select,textarea,input{width:100%;box-sizing:border-box;font-size:1rem;padding:.5rem;}
 textarea{min-height:6rem;}
 .row{display:flex;gap:.5rem;}
 .row>div{flex:1;}
 button{margin-top:1rem;padding:.6rem 1.2rem;font-size:1rem;cursor:pointer;}
 #status{margin-top:1rem;color:#555;}
 audio{width:100%;margin-top:1rem;}
</style>
</head>
<body>
<h1>Irodori-TTS Demo</h1>
<label>話者<select id="speaker"></select></label>
<label>テキスト<textarea id="text">こんにちは、今日はいい天気ですね。</textarea></label>
<div class="row">
  <div><label>seed<input id="seed" type="number" placeholder="empty = random"></label></div>
  <div><label>num_steps<input id="num_steps" type="number" placeholder="default"></label></div>
</div>
<div class="row">
  <div><label>cfg_scale_text<input id="cfg_scale_text" type="number" step="0.1" placeholder="default"></label></div>
  <div><label>cfg_scale_speaker<input id="cfg_scale_speaker" type="number" step="0.1" placeholder="default"></label></div>
</div>
<button id="run">合成</button>
<div id="status"></div>
<audio id="player" controls></audio>
<script>
async function loadSpeakers(){
  const r = await fetch('/speakers');
  const j = await r.json();
  const sel = document.getElementById('speaker');
  for(const s of j.speakers){
    const o=document.createElement('option');
    o.value=s.uuid; o.textContent=`${s.name} (${s.uuid.slice(0,8)})`;
    sel.appendChild(o);
  }
}
function numOrNull(id){
  const v=document.getElementById(id).value.trim();
  return v===''?null:Number(v);
}
async function synth(){
  const status=document.getElementById('status');
  status.textContent='生成中...';
  const body={
    speaker_id: document.getElementById('speaker').value,
    text: document.getElementById('text').value,
  };
  for(const k of ['seed','num_steps','cfg_scale_text','cfg_scale_speaker']){
    const v=numOrNull(k);
    if(v!==null) body[k]=v;
  }
  const t0=performance.now();
  const r=await fetch('/synth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){
    const t=await r.text();
    status.textContent=`error ${r.status}: ${t}`;
    return;
  }
  const blob=await r.blob();
  const url=URL.createObjectURL(blob);
  const player=document.getElementById('player');
  player.src=url; player.play();
  const seed=r.headers.get('x-tts-used-seed');
  status.textContent=`done in ${((performance.now()-t0)/1000).toFixed(2)}s, seed=${seed}, ${(blob.size/1024).toFixed(1)} KB`;
}
document.getElementById('run').addEventListener('click',synth);
loadSpeakers();
</script>
</body>
</html>
"""


@dataclass
class SpeakerSpec:
    uuid: str
    name: str
    adapter: str
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerConfig:
    base_checkpoint: str | None
    base_hf_repo: str | None
    base_hf_filename: str
    model_device: str
    codec_device: str
    model_precision: str
    codec_precision: str
    codec_repo: str
    codec_deterministic_encode: bool
    codec_deterministic_decode: bool
    enable_watermark: bool
    speakers: list[SpeakerSpec]


def _discover_lora_dir(lora_dir: Path) -> list[SpeakerSpec]:
    """Discover standalone .safetensors LoRA exports under ``lora_dir``.

    Each file must carry Irodori-TTS metadata (``name``, ``uuid``,
    ``adapter_config``). ``defaults`` is optional.
    """
    if not lora_dir.is_dir():
        raise FileNotFoundError(f"lora_dir does not exist: {lora_dir}")
    specs: list[SpeakerSpec] = []
    for entry in sorted(lora_dir.glob("*.safetensors")):
        if not is_lora_safetensors_file(entry):
            logger.warning("skipping non-LoRA safetensors file: %s", entry)
            continue
        try:
            meta = read_lora_safetensors_metadata(entry)
        except Exception as exc:
            logger.warning("failed to read metadata from %s: %s", entry, exc)
            continue
        name = meta.get("name") or entry.stem
        speaker_uuid = meta.get("uuid") or str(
            uuid_lib.uuid5(_LORA_UUID_NAMESPACE, entry.stem)
        )
        defaults: dict[str, Any] = {}
        raw_defaults = meta.get("defaults")
        if raw_defaults:
            try:
                parsed = json.loads(raw_defaults)
                if isinstance(parsed, dict):
                    defaults = parsed
            except json.JSONDecodeError as exc:
                logger.warning("skipping defaults in %s: %s", entry, exc)
        specs.append(
            SpeakerSpec(
                uuid=str(speaker_uuid),
                name=str(name),
                adapter=str(entry),
                defaults=defaults,
            )
        )
        logger.info("discovered LoRA: %s (uuid=%s)", name, speaker_uuid)
    return specs


def load_config(path: Path) -> ServerConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    speakers: list[SpeakerSpec] = []
    lora_dir_raw = raw.get("lora_dir")
    if lora_dir_raw:
        lora_dir = Path(str(lora_dir_raw))
        if not lora_dir.is_absolute():
            lora_dir = (path.parent / lora_dir).resolve() if not lora_dir.exists() else lora_dir
        speakers.extend(_discover_lora_dir(lora_dir))

    for s in raw.get("speakers") or []:
        speakers.append(
            SpeakerSpec(
                uuid=str(s["uuid"]),
                name=str(s["name"]),
                adapter=str(s["adapter"]),
                defaults=dict(s.get("defaults") or {}),
            )
        )

    return ServerConfig(
        base_checkpoint=(str(raw["base_checkpoint"]) if raw.get("base_checkpoint") else None),
        base_hf_repo=(str(raw["base_hf_repo"]) if raw.get("base_hf_repo") else None),
        base_hf_filename=str(raw.get("base_hf_filename", "model.safetensors")),
        model_device=str(raw.get("model_device", "cuda")),
        codec_device=str(raw.get("codec_device", "cuda")),
        model_precision=str(raw.get("model_precision", "bf16")),
        codec_precision=str(raw.get("codec_precision", "fp32")),
        codec_repo=str(raw.get("codec_repo", "Aratako/Semantic-DACVAE-Japanese-32dim")),
        codec_deterministic_encode=bool(raw.get("codec_deterministic_encode", True)),
        codec_deterministic_decode=bool(raw.get("codec_deterministic_decode", True)),
        enable_watermark=bool(raw.get("enable_watermark", False)),
        speakers=speakers,
    )


def resolve_base_checkpoint(cfg: ServerConfig) -> Path:
    local = Path(cfg.base_checkpoint) if cfg.base_checkpoint else None
    if local is not None and local.exists():
        logger.info("Using local base checkpoint: %s", local)
        return local
    if not cfg.base_hf_repo:
        raise FileNotFoundError(
            f"base checkpoint not found at {cfg.base_checkpoint!r} and base_hf_repo is not set."
        )
    from huggingface_hub import hf_hub_download

    logger.info(
        "Downloading base checkpoint from HF: %s/%s", cfg.base_hf_repo, cfg.base_hf_filename
    )
    cached = hf_hub_download(repo_id=cfg.base_hf_repo, filename=cfg.base_hf_filename)
    logger.info("Base checkpoint cached at: %s", cached)
    return Path(cached)


class RuntimeRegistry:
    """Single InferenceRuntime with one base model and many LoRA adapters."""

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

    def load(self) -> None:
        if not self.cfg.speakers:
            raise ValueError("No speakers configured.")
        base_path = resolve_base_checkpoint(self.cfg)
        key = RuntimeKey(
            checkpoint=str(base_path),
            model_device=self.cfg.model_device,
            codec_repo=self.cfg.codec_repo,
            model_precision=self.cfg.model_precision,
            codec_device=self.cfg.codec_device,
            codec_precision=self.cfg.codec_precision,
            codec_deterministic_encode=self.cfg.codec_deterministic_encode,
            codec_deterministic_decode=self.cfg.codec_deterministic_decode,
            enable_watermark=self.cfg.enable_watermark,
            compile_model=False,
            compile_dynamic=False,
        )
        adapters = {s.uuid: s.adapter for s in self.cfg.speakers}
        logger.info("Loading base + %d LoRA adapters", len(adapters))
        self._runtime = InferenceRuntime.from_base_with_adapters(
            key=key,
            adapters=adapters,
            default_adapter=self.cfg.speakers[0].uuid,
        )

    def acquire(self, uuid: str) -> tuple[InferenceRuntime, SpeakerSpec]:
        spec = self.get_spec(uuid)
        if self._runtime is None:
            raise RuntimeError("Registry not loaded. Call load() first.")
        with self._lock:
            self._runtime.set_active_adapter(uuid)
            return self._runtime, spec


class SynthRequest(BaseModel):
    speaker_id: str = Field(
        ...,
        description="Registered speaker UUID.",
        examples=["7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb"],
    )
    text: str = Field(..., min_length=1, examples=["こんにちは、今日はいい天気ですね。"])
    seed: int | None = Field(
        default=None, description="Sampling seed. Omit or set <0 for random."
    )
    num_steps: int | None = Field(
        default=None, description="RF sampling steps. Omit or set <=0 to use speaker default."
    )
    cfg_scale_text: float | None = Field(
        default=None, description="Text CFG scale. Omit or set <=0 to use speaker default."
    )
    cfg_scale_speaker: float | None = Field(
        default=None, description="Speaker CFG scale. Omit or set <=0 to use speaker default."
    )
    speaker_kv_scale: float | None = Field(
        default=None, description="Speaker KV scale (>1 strengthens identity). Omit or set <=0 to disable."
    )
    truncation_factor: float | None = Field(
        default=None, description="Noise truncation (e.g. 0.8). Omit or set <=0 to disable."
    )


_POSITIVE_ONLY = {"num_steps", "cfg_scale_text", "cfg_scale_speaker", "speaker_kv_scale", "truncation_factor"}


def _merge_defaults(req: SynthRequest, defaults: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "num_steps": 40,
        "cfg_scale_text": 3.0,
        "cfg_scale_speaker": 5.0,
        "speaker_kv_scale": None,
        "truncation_factor": None,
    }
    for k, v in defaults.items():
        if k in resolved:
            resolved[k] = v
    for k in list(resolved.keys()):
        override = getattr(req, k, None)
        if override is None:
            continue
        if k in _POSITIVE_ONLY and float(override) <= 0:
            continue
        resolved[k] = override
    resolved["seed"] = req.seed if (req.seed is not None and req.seed >= 0) else None
    return resolved


def build_app(cfg_path: Path, *, eager_load: bool = True) -> FastAPI:
    cfg = load_config(cfg_path)
    registry = RuntimeRegistry(cfg)

    app = FastAPI(title="Irodori-TTS Server", version="0.1.0")

    if eager_load:
        registry.load()

    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> HTMLResponse:
        return HTMLResponse(_DEMO_HTML)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "speakers": len(cfg.speakers)}

    @app.get("/speakers")
    def list_speakers() -> dict[str, Any]:
        return {
            "speakers": [
                {"uuid": s.uuid, "name": s.name, "defaults": s.defaults}
                for s in registry.list_speakers()
            ]
        }

    @app.post(
        "/synth",
        response_class=Response,
        responses={
            200: {
                "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}},
                "description": "Generated speech as a WAV file.",
            }
        },
    )
    def synth(req: SynthRequest) -> Response:
        try:
            runtime, spec = registry.acquire(req.speaker_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown speaker_id: {req.speaker_id}")

        params = _merge_defaults(req, spec.defaults)

        use_caption = bool(runtime.model_cfg.use_caption_condition)
        use_speaker = bool(runtime.model_cfg.use_speaker_condition)
        cfg_text, cfg_caption, cfg_speaker, _messages = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=float(params["cfg_scale_text"]),
            cfg_scale_caption=3.0,
            cfg_scale_speaker=float(params["cfg_scale_speaker"]),
            cfg_scale=None,
            use_caption_condition=False,
            use_speaker_condition=use_speaker,
        )

        sampling_req = SamplingRequest(
            text=req.text,
            caption=None,
            ref_wav=None,
            ref_latent=None,
            no_ref=True,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            seconds=FIXED_SECONDS,
            max_ref_seconds=30.0,
            max_text_len=None,
            max_caption_len=None,
            num_steps=int(params["num_steps"]),
            cfg_scale_text=cfg_text,
            cfg_scale_caption=cfg_caption,
            cfg_scale_speaker=cfg_speaker,
            cfg_guidance_mode="independent",
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=params["truncation_factor"],
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=params["speaker_kv_scale"],
            speaker_kv_min_t=0.9 if params["speaker_kv_scale"] is not None else None,
            speaker_kv_max_layers=None,
            seed=params["seed"],
            trim_tail=True,
            tail_window_size=20,
            tail_std_threshold=0.05,
            tail_mean_threshold=0.1,
        )

        try:
            result = runtime.synthesize(sampling_req, log_fn=None)
        except Exception as e:
            logger.exception("synthesis failed")
            raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

        audio = result.audio
        if audio.ndim == 2:
            audio_np = audio.squeeze(0).cpu().numpy()
        else:
            audio_np = audio.cpu().numpy()

        buf = io.BytesIO()
        sf.write(buf, audio_np, int(result.sample_rate), format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()
        headers = {
            "X-TTS-Speaker-Id": spec.uuid,
            "X-TTS-Speaker-Name": urllib.parse.quote(spec.name),
            "X-TTS-Used-Seed": str(int(result.used_seed)),
            "X-TTS-Sample-Rate": str(int(result.sample_rate)),
        }
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("TTS_CONFIG", "config.yaml"))
    parser.add_argument("--host", default=os.environ.get("TTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", "8765")))
    parser.add_argument("--no-eager-load", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = build_app(Path(args.config), eager_load=not args.no_eager_load)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
