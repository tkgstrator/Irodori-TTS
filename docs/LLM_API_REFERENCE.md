# Irodori-TTS API リファレンス（LLM / 外部クライアント向け）

このドキュメントは Irodori-TTS サーバの HTTP API を外部から利用するクライアント（Discord Bot 等）向けにまとめたものです。サーバの内部実装や学習パイプラインには触れません。

API は OpenAI の音声合成エンドポイントに準拠しています。OpenAI の SDK がそのまま使えるので、専用のクライアントを書く必要はありません。

---

## 接続情報

| 項目 | 値 |
|------|----|
| プロトコル | HTTP |
| デフォルトポート | `8765` |
| ベース URL | `http://<host>:8765/v1` |
| 認証 | 無し。SDK が送る `Authorization` は読み捨てる。適当な文字列で通る |

---

## エンドポイント一覧

| メソッド | パス | 用途 |
|----------|------|------|
| `POST` | `/v1/audio/speech` | 音声合成 |
| `GET` | `/v1/models` | モデル ID の取得 |
| `GET` | `/v1/models/{model}` | モデル 1 件の取得 |
| `GET` | `/v1/audio/voices` | 話者一覧（`voice` に入れる UUID） |
| `GET` | `/health` | ヘルスチェック（OpenAI の仕様外） |

---

## 最短の使い方

```python
import httpx
from openai import OpenAI

BASE = "http://localhost:8765/v1"
client = OpenAI(base_url=BASE, api_key="not-checked")

model = client.models.list().data[0].id
voice = httpx.get(f"{BASE}/audio/voices").json()["data"][0]["id"]

response = client.audio.speech.create(
    model=model,
    voice=voice,
    input="こんにちは、今日はいい天気ですね。",
    response_format="wav",
    extra_body={"seed": 42},
)
response.write_to_file("out.wav")
```

---

## POST /v1/audio/speech

| フィールド | 必須 | 説明 |
|------------|------|------|
| `model` | ◯ | `/v1/models` が返す ID。違う値なら 404 `model_not_found` |
| `input` | ◯ | 合成するテキスト。4096 文字まで |
| `voice` | ◯ | 話者 UUID。`{"id": "..."}` の形も可 |
| `response_format` | | `mp3`（既定）/ `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | | 0.25 から 4.0、既定 1.0 |
| `stream_format` | | `audio`（既定）または `sse` |
| `instructions` | | **受け付けない**。渡すと 400 |

`input` の中の `{whisper}` `{cheerful}` のようなショートコードは絵文字に展開されてからモデルに渡ります。使える一覧は `irodori_tts/server/shortcodes.py` にあります。

### 追加フィールド

OpenAI の仕様には無いつまみです。SDK からは `extra_body` に入れて渡します。

| フィールド | 説明 |
|------------|------|
| `seed` | 乱数の種。省略か負値でランダム。**同じ話者を毎回同じ声で出したいなら固定する** |
| `num_steps` | RF のサンプリング回数 |
| `cfg_scale_text` / `cfg_scale_speaker` | CFG の強さ |
| `speaker_kv_scale` | 1 より大きくすると話者性が強まる |
| `truncation_factor` | ノイズの切り詰め。0.8 など |
| `seconds` | 長さを秒で固定し、長さ予測を上書きする |
| `min_seconds` / `max_seconds` | 長さ予測の下限と上限。既定 0.5 と 30.0 |

省略した項目は話者ごとの既定値（LoRA に埋め込まれた `defaults`）で埋まります。

### レスポンス

音声そのものが返ります。

| `response_format` | Content-Type | 備考 |
|-------------------|--------------|------|
| `mp3` | `audio/mpeg` | 既定 |
| `opus` | `audio/ogg` | Ogg 容器に入った Opus |
| `aac` | `audio/aac` | ADTS |
| `flac` | `audio/flac` | |
| `wav` | `audio/wav` | PCM16 |
| `pcm` | `audio/pcm` | ヘッダ無し。24 kHz 16 bit モノラルに揃えて返る |

参考情報がヘッダに付きます。`X-TTS-Voice-Id`、`X-TTS-Used-Seed`（実際に使われた seed）、`X-TTS-Sample-Rate`。

### SSE

`stream_format` に `sse` を指定すると `text/event-stream` になります。

```
data: {"type": "speech.audio.delta", "audio": "<base64>"}

data: {"type": "speech.audio.done", "usage": {"input_tokens": 14, "output_tokens": 101, "total_tokens": 115}}
```

`delta` をつないで base64 を戻すと、通常のリクエストと同じバイト列になります。ただし合成は文全体を一度に作るので、**最初の音が出るまでの時間は縮みません**。頭出しを早くしたいなら、文ごとにリクエストを分けてください。

---

## GET /v1/audio/voices

`voice` に入れる UUID を拾う口です。OpenAI 本家には無いパスです。

```json
{
  "object": "list",
  "data": [
    {
      "id": "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb",
      "object": "voice",
      "name": "Alice",
      "cv": "声優名",
      "category": {"id": "female", "label": "女性"}
    }
  ]
}
```

---

## GET /health

```json
{"status": "ok", "model": "irodori-tts-v4.1-small", "voices": 3}
```

---

## エラー

OpenAI と同じ形で返ります。

```json
{"error": {"message": "unknown voice: ...", "type": "invalid_request_error", "param": "voice", "code": "voice_not_found"}}
```

| 状況 | ステータス | `code` |
|------|-----------|--------|
| `model` が一致しない | 404 | `model_not_found` |
| `voice` が未登録 | 400 | `voice_not_found` |
| 値の検証に落ちた | 400 | なし |
| 合成そのものが失敗 | 500 | なし（`type` は `server_error`） |

---

## 注意点

- 合成はサーバ内で 1 本ずつしか進みません。同時に投げても順番待ちになります。
- 話者アダプタは起動時にしかスキャンされません。新しい話者を足したらサーバを再起動してください。
- `seed` を固定し、テキストも同じなら、出力はバイト単位で同じになります。
