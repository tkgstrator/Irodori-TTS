"""Tests for the OpenAI-compatible HTTP surface.

The runtime is faked, so these run without a GPU, a checkpoint or the network,
but the encoding is real: every response_format is produced by the same
libsndfile (or ffmpeg) the server uses in production.
"""

from __future__ import annotations

import base64
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import build_app
from tests.helpers import (
    UUID_A,
    UUID_B,
    install_fake_runtime,
    lora_test_config,
    write_config,
)

MODEL = "irodori-tts"


def speech_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": MODEL, "input": "こんにちは", "voice": UUID_A}
    body.update(overrides)
    return body


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return install_fake_runtime(monkeypatch)


@pytest.fixture
def client(tmp_path: Path, calls: dict[str, Any]) -> TestClient:
    del calls  # the fixture installs the fake runtime; tests that inspect it ask for it too
    return TestClient(build_app(lora_test_config(tmp_path), eager_load=True))


@pytest.fixture
def empty_client(tmp_path: Path) -> TestClient:
    return TestClient(build_app(write_config(tmp_path / "c.yaml", {}), eager_load=False))


class TestHealth:
    def test_reports_the_model_and_voice_count(self, client: TestClient):
        assert client.get("/health").json() == {
            "status": "ok",
            "model": MODEL,
            "voices": 1,
        }

    def test_empty_config(self, empty_client: TestClient):
        assert empty_client.get("/health").json() == {
            "status": "ok",
            "model": MODEL,
            "voices": 0,
        }


class TestModels:
    def test_list_shape(self, client: TestClient):
        body = client.get("/v1/models").json()
        assert body["object"] == "list"
        (model,) = body["data"]
        assert model["id"] == MODEL
        assert model["object"] == "model"
        assert model["owned_by"] == "irodori-tts"
        assert isinstance(model["created"], int)

    def test_retrieve(self, client: TestClient):
        assert client.get(f"/v1/models/{MODEL}").json()["id"] == MODEL

    def test_retrieve_unknown_is_404(self, client: TestClient):
        response = client.get("/v1/models/tts-1")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "model_not_found"
        assert error["param"] == "model"
        assert "tts-1" in error["message"]


class TestVoices:
    def test_payload_shape(self, client: TestClient):
        body = client.get("/v1/audio/voices").json()
        assert body["object"] == "list"
        assert body["data"] == [
            {
                "id": UUID_A,
                "object": "voice",
                "name": "Alice",
                "cv": "Alice Actor",
                "category": {"id": "female", "label": "女性"},
                "defaults": {"num_steps": 30},
            }
        ]

    def test_empty_config(self, empty_client: TestClient):
        assert empty_client.get("/v1/audio/voices").json()["data"] == []


class TestSpeechValidation:
    def test_unknown_voice(self, client: TestClient):
        response = client.post("/v1/audio/speech", json=speech_body(voice=UUID_B))
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "voice"
        assert error["code"] == "voice_not_found"

    def test_unknown_model(self, client: TestClient):
        response = client.post("/v1/audio/speech", json=speech_body(model="tts-1"))
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"

    def test_instructions_are_refused(self, client: TestClient):
        response = client.post("/v1/audio/speech", json=speech_body(instructions="やわらかく"))
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "instructions"

    @pytest.mark.parametrize(
        ("overrides", "param"),
        [
            ({"input": ""}, "input"),
            ({"speed": 5.0}, "speed"),
            ({"response_format": "ogg"}, "response_format"),
            ({"stream_format": "chunked"}, "stream_format"),
            ({"min_seconds": 5, "max_seconds": 1}, None),
        ],
    )
    def test_schema_failures_use_the_error_envelope(
        self, client: TestClient, overrides: dict[str, Any], param: str | None
    ):
        response = client.post("/v1/audio/speech", json=speech_body(**overrides))
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == param
        assert error["message"]

    def test_missing_body_field(self, client: TestClient):
        response = client.post("/v1/audio/speech", json={"model": MODEL, "input": "hi"})
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "voice"


class TestSpeechFormats:
    @pytest.mark.parametrize(
        ("response_format", "media_type", "magic"),
        [
            ("wav", "audio/wav", b"RIFF"),
            ("flac", "audio/flac", b"fLaC"),
            ("opus", "audio/ogg", b"OggS"),
            ("mp3", "audio/mpeg", None),
        ],
    )
    def test_container_formats(
        self,
        client: TestClient,
        response_format: str,
        media_type: str,
        magic: bytes | None,
    ):
        response = client.post(
            "/v1/audio/speech", json=speech_body(response_format=response_format)
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == media_type
        assert response.content
        if magic is not None:
            assert response.content.startswith(magic)

    def test_mp3_starts_with_a_frame_or_a_tag(self, client: TestClient):
        content = client.post("/v1/audio/speech", json=speech_body(response_format="mp3")).content
        assert content.startswith(b"ID3") or content[0] == 0xFF

    def test_default_format_is_mp3(self, client: TestClient):
        response = client.post("/v1/audio/speech", json=speech_body())
        assert response.headers["content-type"] == "audio/mpeg"

    def test_pcm_is_resampled_to_24k(self, client: TestClient):
        """The fake codec runs at 48 kHz; OpenAI promises 24 kHz for headerless pcm."""
        response = client.post("/v1/audio/speech", json=speech_body(response_format="pcm"))
        assert response.headers["content-type"] == "audio/pcm"
        # 4800 samples at 48 kHz become 2400 at 24 kHz, 2 bytes each.
        assert len(response.content) == 4800

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="aac needs ffmpeg")
    def test_aac(self, client: TestClient):
        response = client.post("/v1/audio/speech", json=speech_body(response_format="aac"))
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/aac"
        assert response.content[0] == 0xFF

    def test_headers_report_the_voice_and_seed(self, client: TestClient):
        response = client.post("/v1/audio/speech", json=speech_body())
        assert response.headers["X-TTS-Voice-Id"] == UUID_A
        assert response.headers["X-TTS-Used-Seed"] == "7"
        assert response.headers["X-TTS-Sample-Rate"] == "48000"


def sse_payloads(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


class TestSse:
    def test_event_sequence(self, client: TestClient):
        response = client.post(
            "/v1/audio/speech", json=speech_body(stream_format="sse", response_format="wav")
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse_payloads(response.text)
        assert [e["type"] for e in events[:-1]] == ["speech.audio.delta"] * (len(events) - 1)
        assert events[-1]["type"] == "speech.audio.done"

    def test_deltas_concatenate_to_the_whole_file(self, client: TestClient):
        streamed = client.post(
            "/v1/audio/speech", json=speech_body(stream_format="sse", response_format="wav")
        )
        whole = client.post("/v1/audio/speech", json=speech_body(response_format="wav"))
        joined = b"".join(
            base64.b64decode(e["audio"])
            for e in sse_payloads(streamed.text)
            if e["type"] == "speech.audio.delta"
        )
        assert joined == whole.content

    def test_done_carries_usage(self, client: TestClient):
        response = client.post("/v1/audio/speech", json=speech_body(stream_format="sse"))
        usage = sse_payloads(response.text)[-1]["usage"]
        assert usage["input_tokens"] == len("こんにちは")
        assert usage["output_tokens"] > 0
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


class TestSamplingRequest:
    def test_adapter_survives_synthesis(self, client: TestClient, calls: dict[str, Any]):
        """SamplingRequest defaults to keep_adapter=False, under which every LoRA
        speaker comes out as the base voice. The speech path must opt out."""
        assert client.post("/v1/audio/speech", json=speech_body()).status_code == 200
        assert calls["made"]["base"].last_request.keep_adapter is True

    def test_speed_divides_the_duration(self, client: TestClient, calls: dict[str, Any]):
        client.post("/v1/audio/speech", json=speech_body(speed=2.0))
        assert calls["made"]["base"].last_request.duration_scale == pytest.approx(0.5)

    def test_extensions_reach_the_sampler(self, client: TestClient, calls: dict[str, Any]):
        client.post("/v1/audio/speech", json=speech_body(seed=42, num_steps=12))
        request = calls["made"]["base"].last_request
        assert request.seed == 42
        assert request.num_steps == 12

    def test_shortcodes_are_expanded(self, client: TestClient, calls: dict[str, Any]):
        client.post("/v1/audio/speech", json=speech_body(input="ねえ{cheerful}"))
        assert calls["made"]["base"].last_request.text == "ねえ😊"


class TestOpenApiContract:
    def test_routes_are_registered(self, client: TestClient):
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths) == {
            "/health",
            "/v1/models",
            "/v1/models/{model}",
            "/v1/audio/voices",
            "/v1/audio/speech",
        }
        assert set(paths["/v1/audio/speech"]) == {"post"}

    def test_audio_media_types_documented(self, client: TestClient):
        paths = client.get("/openapi.json").json()["paths"]
        content = paths["/v1/audio/speech"]["post"]["responses"]["200"]["content"]
        assert set(content) >= {"audio/mpeg", "audio/wav", "audio/pcm", "text/event-stream"}


def _start_live_server(tmp_path: Path, *, extra_speaker: bool = False) -> tuple[Any, threading.Thread, int]:
    import uvicorn

    app = build_app(lora_test_config(tmp_path, extra_speaker=extra_speaker), eager_load=True)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, port


@pytest.fixture
def live_url(tmp_path: Path, calls: dict[str, Any]) -> Any:
    """A real socket. httpx2's ASGITransport is async-only, and the SDK is sync."""
    del calls
    server, thread, port = _start_live_server(tmp_path)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def live_url_two_speakers(tmp_path: Path, calls: dict[str, Any]) -> Any:
    """Same as `live_url`, but with two discoverable speakers (Alice/Bob) so
    concurrency tests have a second voice to race against the first."""
    del calls
    server, thread, port = _start_live_server(tmp_path, extra_speaker=True)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


class TestOfficialSdk:
    """The compatibility claim, checked with the real client rather than our own JSON."""

    @pytest.fixture
    def sdk(self, live_url: str) -> Any:
        from openai import OpenAI

        return OpenAI(api_key="not-checked", base_url=live_url)

    def test_create_speech(self, sdk: Any):
        response = sdk.audio.speech.create(
            model=MODEL, voice=UUID_A, input="こんにちは", response_format="wav"
        )
        assert response.content.startswith(b"RIFF")

    def test_extra_body_carries_the_seed(self, sdk: Any, calls: dict[str, Any]):
        sdk.audio.speech.create(
            model=MODEL, voice=UUID_A, input="こんにちは", extra_body={"seed": 5}
        )
        assert calls["made"]["base"].last_request.seed == 5

    def test_list_models(self, sdk: Any):
        assert [m.id for m in sdk.models.list()] == [MODEL]

    def test_unknown_model_raises_not_found(self, sdk: Any):
        import openai

        with pytest.raises(openai.NotFoundError):
            sdk.audio.speech.create(model="tts-1", voice=UUID_A, input="hi")

    def test_unknown_voice_raises_bad_request(self, sdk: Any):
        import openai

        with pytest.raises(openai.BadRequestError) as excinfo:
            sdk.audio.speech.create(model=MODEL, voice=UUID_B, input="hi")
        assert excinfo.value.param == "voice"


class TestConcurrentSpeech:
    """The runtime is a single instance shared across every speaker — only the
    active adapter differs. registry.acquire() must hold its lock through the
    whole synthesis call, or a concurrent request's adapter switch bleeds into
    an in-flight one and the wrong voice ends up speaking."""

    def test_a_concurrent_request_cannot_swap_the_adapter_mid_synthesis(
        self, live_url_two_speakers: str, calls: dict[str, Any]
    ):
        import httpx

        runtime = calls["made"]["base"]
        a_reached_synthesis = threading.Event()
        release_a = threading.Event()
        adapter_seen_by_a: list[str] = []

        def on_synthesize(rt: Any) -> None:
            if rt.last_request.text != "こんにちは":
                return  # request B: let it run straight through
            a_reached_synthesis.set()
            release_a.wait(timeout=5)
            adapter_seen_by_a.append(rt.active_adapter)

        runtime.on_synthesize = on_synthesize

        results: dict[str, int] = {}

        def call_a() -> None:
            response = httpx.post(
                f"{live_url_two_speakers}/audio/speech", json=speech_body(voice=UUID_A), timeout=10
            )
            results["a_status"] = response.status_code

        def call_b() -> None:
            assert a_reached_synthesis.wait(timeout=5), "request A never reached synthesis"
            # Give request B a moment to actually reach (and, once the fix is in
            # place, block on) the registry lock before request A is released.
            time.sleep(0.2)
            response = httpx.post(
                f"{live_url_two_speakers}/audio/speech",
                json=speech_body(voice=UUID_B, input="hi"),
                timeout=10,
            )
            results["b_status"] = response.status_code

        thread_a = threading.Thread(target=call_a)
        thread_b = threading.Thread(target=call_b)
        thread_a.start()
        thread_b.start()

        assert a_reached_synthesis.wait(timeout=5), "request A never reached synthesis"
        time.sleep(0.3)
        release_a.set()

        thread_a.join(timeout=5)
        thread_b.join(timeout=5)

        assert results.get("a_status") == 200
        assert results.get("b_status") == 200
        # Request A's own synthesis must see Alice's adapter throughout — never
        # Bob's, no matter what request B tried to do in between.
        assert adapter_seen_by_a == [UUID_A]
