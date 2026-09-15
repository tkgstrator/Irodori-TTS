# Irodori-TTS 推論サーバガイド

学習済みの LoRA 話者アダプタを FastAPI で配信する `server.py` の使い方です。1 つのベースモデル + 複数 LoRA を 1 プロセスに載せ、リクエストごとに active adapter を切り替えて合成します。

API は OpenAI の音声合成エンドポイントに合わせてあります。OpenAI 公式の SDK の `base_url` をこのサーバに向ければ、変換層を挟まずにそのまま喋らせられます。

---

## 1. 全体像

- **ベースモデル**: `configs/runtime.yaml` の `base_checkpoint`（ローカルに無ければ `base_hf_repo` から HF に取りに行きます）を 1 回だけ読み込み。
- **話者アダプタ**: `lora_dir`（既定 `models/LoRA/`）配下の `.safetensors` を起動時にスキャンし、それぞれに埋め込まれた metadata (`name` / `uuid` / `defaults` / `adapter_config`) から話者を自動登録します。YAML 側に話者ブロックを書く必要はありません。声はすべて LoRA なので、1 体も見つからなければ何も読み込まず、合成要求は 400 を返します。
- **推論**: `POST /v1/audio/speech` に `model` / `input` / `voice`（話者 UUID）を渡すと音声が返ります。UUID は `GET /v1/audio/voices` で拾います。
- **VoiceDesign（自然文で声を作る caption モード）は廃止しました**。同じ seed でも台詞が変わると声が変わってしまい、キャラクターの声を固定する用途に使えなかったためです。声は LoRA 話者だけになりました。

---

## 2. `configs/runtime.yaml`

```yaml
base_checkpoint: models/Irodori-TTS-v4.1-Small/model.safetensors
base_hf_repo: Aratako/Irodori-TTS-v4.1-Small
base_hf_filename: model.safetensors

model_device: cuda
codec_device: cuda
model_precision: bf16
codec_precision: fp32
codec_repo: Aratako/Semantic-DACVAE-Japanese-32dim
codec_deterministic_encode: true
codec_deterministic_decode: true
enable_watermark: false

# GET /v1/models が名乗る ID。省略すると base_version から作られる。
# model_id: irodori-tts-v4.1-small

tail_window_size: 20
tail_std_threshold: 0.05
tail_mean_threshold: 0.1
show_timings: true

lora_dir: models/LoRA
```

主要フィールド:

| フィールド                   | 説明 |
|------------------------------|------|
| `base_checkpoint`            | ベースモデルのローカルパス。見つからなければ `base_hf_repo` から pull |
| `base_hf_repo` / `base_hf_filename` | HF fallback 用 |
| `model_device` / `codec_device` | `cuda` / `cpu` / `cuda:0` など |
| `model_precision`            | `bf16` / `fp32` |
| `codec_precision`            | DACVAE codec の精度。通常 `fp32` |
| `codec_repo`                 | DACVAE codec の HF repo |
| `codec_deterministic_encode/decode` | 決定論モード（同じ入力 → 同じ出力） |
| `enable_watermark`           | watermark 付与を有効にするか（通常 `false`） |
| `model_id`                   | `GET /v1/models` が返す ID で、`POST /v1/audio/speech` の `model` に一致が要る。省略時は `base_version` から `irodori-tts-v4.1-small` のように作られる |
| `tail_window_size`           | 末尾トリミングのウィンドウサイズ（デフォルト `20`） |
| `tail_std_threshold`         | 末尾トリミングの標準偏差閾値（デフォルト `0.05`） |
| `tail_mean_threshold`        | 末尾トリミングの平均値閾値（デフォルト `0.1`） |
| `show_timings`               | 合成のステージ別タイミングをログ出力（デフォルト `true`） |
| `lora_dir`                   | `.safetensors` LoRA を探すディレクトリ |

話者は `lora_dir` に置かれた LoRA から自動検出されます。

---

## 3. 話者 LoRA のエクスポート

学習後の PEFT checkpoint ディレクトリをサーバが読める単一 `.safetensors` に書き出します:

```bash
uv run python scripts/lora/export_lora_to_safetensors.py \
  --input  outputs/<speaker>_lora/checkpoint_best_val_loss_0002400_0.312100 \
  --output models/LoRA/<speaker>.safetensors \
  --defaults '{"num_steps": 40, "cfg_scale_text": 3.0, "cfg_scale_speaker": 5.0}'
```

`name` は `data/<speaker>/config.yaml` の `speaker.label` を正として自動で入ります。`--name` は `speaker.label` / `speaker.name` が無い古いデータ向けの fallback です。canonical なラベルがある場合は `--name` を渡してもそちらが優先されます。

書き出される `__metadata__` (str→str):

| key               | 用途 |
|-------------------|------|
| `format`          | `irodori-tts-lora/v1` 固定。LoRA 単一ファイル export の schema version で、ベースモデル世代（v2 / v3）とは別物 |
| `name`            | サーバ / デモ UI に表示される名前。通常は `speaker.label` と同じ |
| `uuid`            | `speaker_id`。`--uuid` 未指定時は出力ファイル名から UUIDv5 で決定論的に生成 |
| `defaults`        | JSON。`num_steps` / `cfg_scale_text` / `cfg_scale_speaker` / `speaker_kv_scale` / `truncation_factor` / `seed` / `seconds` / `min_seconds` / `max_seconds` / `duration_scale` の既定値 |
| `adapter_config`  | JSON。PEFT の `adapter_config.json`（rank / target modules 等） |
| `base_init`       | JSON。学習時の `base_init.json` |
| `model_config`    | JSON。学習時の `config.json`（モデル / train config dump） |
| `manifest_size`   | 学習データの件数 |
| `speaker.label`   | 任意。`data/<speaker>/config.yaml` の `speaker.label` |
| `speaker.cv`      | 任意。`data/<speaker>/config.yaml` の `speaker.cv` |
| `category.id`     | 任意。`data/<speaker>/config.yaml` の `category.id` |
| `category.label`  | 任意。`data/<speaker>/config.yaml` の `category.label` |

さらに学習時に `train.py` が埋め込むフラットキー (`uuid` / `model_name` / `speaker` / `base_model` / `step` / `epoch` / `val_loss` / `created_at` / `lora_r` / `lora_alpha` / `lora_dropout` / `lora_target_modules`) もそのまま持ち回されます。export 側の `uuid` は別 namespace で再生成されるので、学習時 UUID とサーバ UUID は別物です（識別文字列が欲しければ `safetensors.safe_open(...).metadata()` で両方取れます）。

エクスポート後は `models/LoRA/` にファイルを置けば起動時に自動的に拾われます。

---

## 4. ローカルでの起動

```bash
uv run python server.py \
  --config configs/runtime.yaml \
  --host 127.0.0.1 \
  --port 8765
```

環境変数でも上書きできます:

| 変数            | 既定値         | 説明 |
|-----------------|----------------|------|
| `TTS_CONFIG`    | `config.yaml`  | config YAML のパス |
| `TTS_HOST`      | `127.0.0.1`    | listen host |
| `TTS_PORT`      | `8765`         | listen port |

`--no-eager-load` を付けると base + LoRA のロードを初回リクエストまで遅延できます（起動時間を短くしたいとき）。

---

## 5. Docker での起動

`docker/runtime/Dockerfile` が最小イメージを作ります。依存は起動時に `uv sync` でコンテナの system Python (`/usr/local`) へ入れるので、イメージ自体は薄いままです。ベースは `nvidia/cuda` ではなく `python:3.12-slim` です。PyPI の torch wheel が `nvidia-*-cu12` として CUDA ランタイムと cuDNN を持ってくるため、CUDA ベースイメージを敷くと同じ物が二重に載ります（約 4.4GB）。ドライバはこれまで通り `--gpus all` / compose の device reservation でホストから来ます。

### ビルド

リポジトリに `docker/runtime/compose.yaml` が同梱されており、`build:` と `image:` の両方を持っているので `docker compose` だけでビルド〜起動まで通ります。

```bash
docker compose -f docker/runtime/compose.yaml build      # ビルド
docker compose -f docker/runtime/compose.yaml up -d      # 起動
docker compose -f docker/runtime/compose.yaml logs -f    # ログ追跡
```

`.dockerignore` は `docker/runtime/Dockerfile.dockerignore` (BuildKit の `<Dockerfile>.dockerignore` 規約) に置いています。

ボリュームの要点:

- **`uv_cache`**: 初回の `uv sync --frozen --no-dev` で落とした wheel を保持。2 回目以降の起動が数秒で済みます。
- **`hf_cache`**: DACVAE codec / tokenizer などの HF hub キャッシュを永続化。
- **`../../models`**: ベースモデル (`model.safetensors`) と LoRA 話者 `.safetensors` を置く場所。未マウント / ベースモデル未配置なら `base_hf_repo` から取りに行きます。LoRA は `models/LoRA/` 配下に置いてください。新しい LoRA を追加したら**コンテナの再起動が必要**（起動時にだけスキャンするため）。

---

## 6. API

OpenAI の音声合成 API に合わせてあります。差分は次の三つです。

- `voice` に入れるのは話者 UUID です。`alloy` のような短い名前ではありません。
- `instructions` は受け付けません（VoiceDesign を廃止したため）。渡すと 400 になります。
- OpenAI に無い調整つまみを追加フィールドとして受けます。公式 SDK からは `extra_body` で渡せます。

鍵の検査はしません。SDK は `Authorization` ヘッダを必ず送りますが、こちらは読まずに捨てます。

### `POST /v1/audio/speech`

| フィールド          | 必須 | 説明 |
|---------------------|------|------|
| `model`             | ◯    | `GET /v1/models` が返す ID。違う値なら 404 |
| `input`             | ◯    | 合成するテキスト。4096 文字まで。`{whisper}` などのショートコードは絵文字に展開される |
| `voice`             | ◯    | 話者 UUID。`{"id": "..."}` の形でも受ける |
| `response_format`   |      | `mp3`（既定）/ `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed`             |      | 0.25 から 4.0、既定 1.0。予測された長さを割る形で効く |
| `stream_format`     |      | `audio`（既定）または `sse` |
| `instructions`      |      | 受け付けない。渡すと 400 |

追加フィールド（OpenAI の仕様には無い）:

| フィールド | 説明 |
|------------|------|
| `seed` | 乱数の種。省略か負値でランダム。**同じ話者を別のリクエストで同じ声にしたいなら固定する** |
| `num_steps` | RF のサンプリング回数 |
| `cfg_scale_text` / `cfg_scale_speaker` | CFG の強さ |
| `speaker_kv_scale` | 1 より大きくすると話者性が強まる |
| `truncation_factor` | ノイズの切り詰め。0.8 など |
| `seconds` | 長さを秒で固定し、長さ予測を上書きする |
| `min_seconds` / `max_seconds` | 長さ予測の下限と上限。既定 0.5 と 30.0 |

値の決まり方は「ハードコードの既定値、LoRA の `defaults`、リクエスト」の順で後が勝ちます。`num_steps` / `cfg_scale_*` / `speaker_kv_scale` / `truncation_factor` に 0 以下を渡した場合は指定が無かったものとして扱われ、`defaults` の値に戻ります。`speed` だけは上書きではなく、決まった `duration_scale` を割ります。

返るのは音声そのもので、Content-Type は `mp3` が `audio/mpeg`、`opus` が `audio/ogg`、`aac` が `audio/aac`、`flac` が `audio/flac`、`wav` が `audio/wav`、`pcm` が `audio/pcm` です。`pcm` はヘッダを持たないので、OpenAI の約束どおり 24 kHz 16 bit モノラルに揃えて返します。容器を持つ他の五つはモデルのサンプルレートのままで、レートはファイルのヘッダに入ります。

参考情報としてヘッダも付けます。

| ヘッダ | 内容 |
|--------|------|
| `X-TTS-Voice-Id` | 使った話者 UUID |
| `X-TTS-Used-Seed` | 実際に使われた seed。ランダムだったときの値を拾える |
| `X-TTS-Sample-Rate` | 合成時のサンプルレート |

```bash
curl -s http://localhost:8765/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "irodori-tts-v4.1-small",
        "input": "こんにちは、今日はいい天気ですね。",
        "voice": "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb",
        "response_format": "wav",
        "seed": 42
      }' \
  -o out.wav
```

### SSE

`stream_format` に `sse` を指定すると `text/event-stream` で返ります。

```
data: {"type": "speech.audio.delta", "audio": "<base64>"}

data: {"type": "speech.audio.done", "usage": {"input_tokens": 14, "output_tokens": 101, "total_tokens": 115}}
```

`delta` の中身をつないで base64 を戻すと、`stream_format` を付けずに叩いたときと同じバイト列になります。

ただし**最初の音が出るまでの時間は縮みません**。合成が文全体を一度に作る作りなので、出来上がったファイルを刻んで流しているだけです。頭出しを早くしたいなら、いまのところ文ごとにリクエストを分けてください。

### `GET /v1/models` と `GET /v1/models/{model}`

読み込んでいるモデル 1 つを返します。

```json
{"object": "list", "data": [{"id": "irodori-tts-v4.1-small", "object": "model", "created": 1757000000, "owned_by": "irodori-tts"}]}
```

### `GET /v1/audio/voices`

話者一覧です。OpenAI 本家には無く、`voice` に入れる UUID を知るために置いています。

```json
{
  "object": "list",
  "data": [
    {
      "id": "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb",
      "object": "voice",
      "name": "Alice",
      "cv": "Alice Actor",
      "category": {"id": "female", "label": "女性"},
      "defaults": {"num_steps": 30}
    }
  ]
}
```

### `GET /health`

OpenAI の仕様外です。コンテナの healthcheck が叩きます。

```json
{"status": "ok", "model": "irodori-tts-v4.1-small", "voices": 3}
```

### エラー

OpenAI と同じ封筒で返します。SDK はこの中身を見て例外の種類を決めるので、形が揃っていないと `BadRequestError` などに翻訳されません。

```json
{"error": {"message": "unknown voice: ...", "type": "invalid_request_error", "param": "voice", "code": "voice_not_found"}}
```

| 状況 | ステータス | `code` |
|------|-----------|--------|
| `model` が一致しない | 404 | `model_not_found` |
| `voice` が未登録 | 400 | `voice_not_found` |
| 値の検証に落ちた（空の `input`、範囲外の `speed` など） | 400 | なし |
| 合成そのものが失敗 | 500 | なし（`type` は `server_error`） |

### 公式 SDK から

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8765/v1", api_key="not-checked")

response = client.audio.speech.create(
    model="irodori-tts-v4.1-small",
    voice="7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb",
    input="こんにちは、今日はいい天気ですね。",
    response_format="wav",
    extra_body={"seed": 42},
)
response.write_to_file("out.wav")
```

## 7. 新しい話者の追加フロー

1. LoRA を学習して `outputs/<speaker>_lora/checkpoint_best_val_loss_*` を得る（`docs/TRAINING.md` 参照）。
2. `samples/` の per-step wav を聴いて採用する checkpoint を決める。
3. `scripts/lora/export_lora_to_safetensors.py` で `.safetensors` にエクスポートし、`speaker.label` 由来の `name` と `--defaults` を埋め込む。
4. `.safetensors` を `models/LoRA/` に置く。
5. サーバを再起動（`docker compose restart tts` など）。
6. `GET /v1/audio/voices` で新 UUID を確認、`POST /v1/audio/speech` で動作確認。

---

## 8. トラブルシュート

| 症状                                                 | 対処 |
|------------------------------------------------------|------|
| 起動時 `lora_dir does not exist`                     | `lora_dir` の解決先を確認。Docker なら `./models/LoRA` が正しくマウントされているか |
| `skipping non-LoRA safetensors file`                 | `format=irodori-tts-lora/v1` が入っていない。`export_lora_to_safetensors.py` 経由で書き出す |
| 話者一覧に出ているのに `voice` が 400 になる           | UUID をコピペミスしていないか確認（`/v1/audio/voices` の値をそのまま使う） |
| `model_not_found` が返る                             | `GET /v1/models` の `id` をそのまま `model` に入れる。`tts-1` は通らない |
| GPU を認識しない                                     | `--gpus all` / compose の `deploy.resources.reservations.devices` を確認 |
| ベースモデル pull で 401/403                         | private repo の場合は `HF_TOKEN` を環境変数に入れる |
| 音質が学習時サンプルより悪い                         | `defaults` の `num_steps` / `cfg_scale_*` を調整、または別の checkpoint をエクスポートし直す |
