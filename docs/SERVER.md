# Irodori-TTS 推論サーバガイド

学習済みの LoRA 話者アダプタを FastAPI で配信する `tts_server.py` の使い方です。1 つのベースモデル + 複数 LoRA を 1 プロセスに載せ、リクエストごとに active adapter を切り替えて合成します。

---

## 1. 全体像

- **ベースモデル**: `configs/runtime.yaml` の `base_checkpoint`（ローカルに無ければ `base_hf_repo` から HF に取りに行きます）を 1 回だけ読み込み。
- **話者アダプタ**: `lora_dir`（既定 `data/LoRA/`）配下の `.safetensors` を起動時にスキャンし、それぞれに埋め込まれた metadata (`name` / `uuid` / `defaults` / `adapter_config`) から話者を自動登録します。YAML 側に話者ブロックを書く必要はありません。
- **推論**: `/synth` にテキストと `speaker_id` (= 話者 UUID) を POST すると、WAV が返ります。`no_ref=true` 固定 (voice-cloning ではなく LoRA で話者情報を持つ運用) です。

---

## 2. `configs/runtime.yaml`

```yaml
base_checkpoint: models/Irodori-TTS-500M-v2/model.safetensors
base_hf_repo: Aratako/Irodori-TTS-500M-v2
base_hf_filename: model.safetensors

model_device: cuda
codec_device: cuda
model_precision: bf16
codec_precision: fp32
codec_repo: Aratako/Semantic-DACVAE-Japanese-32dim
codec_deterministic_encode: true
codec_deterministic_decode: true
enable_watermark: false

lora_dir: data/LoRA
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
| `lora_dir`                   | `.safetensors` LoRA を探すディレクトリ |

`speakers:` ブロックを使った手動登録もまだサポートされていますが、`lora_dir` による auto-discover が標準ルートです。

---

## 3. 話者 LoRA のエクスポート

学習後の PEFT checkpoint ディレクトリをサーバが読める単一 `.safetensors` に書き出します:

```bash
uv run python scripts/lora/export_lora_to_safetensors.py \
  --input  outputs/<speaker>_lora/checkpoint_best_val_loss_0002400_0.312100 \
  --output data/LoRA/<speaker>.safetensors \
  --name   "<表示名>" \
  --defaults '{"num_steps": 40, "cfg_scale_text": 3.0, "cfg_scale_speaker": 5.0}'
```

書き出される `__metadata__` (str→str):

| key               | 用途 |
|-------------------|------|
| `format`          | `irodori-tts-lora/v1` 固定 |
| `name`            | サーバ / デモ UI に表示される名前 |
| `uuid`            | `speaker_id`。`--uuid` 未指定時は出力ファイル名から UUIDv5 で決定論的に生成 |
| `defaults`        | JSON。`num_steps` / `cfg_scale_text` / `cfg_scale_speaker` / `speaker_kv_scale` / `truncation_factor` の既定値 |
| `adapter_config`  | JSON。PEFT の `adapter_config.json`（rank / target modules 等） |
| `base_init`       | JSON。学習時の `base_init.json` |
| `model_config`    | JSON。学習時の `config.json`（モデル / train config dump） |
| `manifest_size`   | 学習データの件数 |

さらに学習時に `train.py` が埋め込むフラットキー (`uuid` / `model_name` / `speaker` / `base_model` / `step` / `epoch` / `val_loss` / `created_at` / `lora_r` / `lora_alpha` / `lora_dropout` / `lora_target_modules`) もそのまま持ち回されます。export 側の `uuid` は別 namespace で再生成されるので、学習時 UUID とサーバ UUID は別物です（識別文字列が欲しければ `safetensors.safe_open(...).metadata()` で両方取れます）。

エクスポート後は `data/LoRA/` にファイルを置けば起動時に自動的に拾われます。

---

## 4. ローカルでの起動

```bash
uv run python tts_server.py \
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

`docker/runtime/Dockerfile` が最小イメージを作ります。`.venv` は起動時に `uv sync` でマウント先 volume に展開するので、イメージ自体は薄いままです。

### ビルド

```bash
docker build -f docker/runtime/Dockerfile -t <org>/irodori-tts .
```

### compose 例 (`docker/runtime/compose.yaml` 同梱)

```yaml
services:
  tts:
    image: <org>/irodori-tts:latest
    container_name: irodori-tts
    ports:
      - 8765:8765
    volumes:
      - ./configs/runtime.yaml:/app/config.yaml:ro
      - ./models:/app/models:ro
      - ./data/LoRA:/app/data/LoRA:ro
      - hf_cache:/root/.cache/huggingface
      - tts_venv:/app/.venv
    environment:
      TTS_HOST: 0.0.0.0
      TTS_PORT: 8765
      HF_HOME: /root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  hf_cache:
  tts_venv:
```

起動:

```bash
docker compose -f docker/runtime/compose.yaml up -d
docker compose -f docker/runtime/compose.yaml logs -f
```

- **`tts_venv`**: 初回に `uv sync --frozen --no-dev` で構築。2 回目以降は数秒で起動。
- **`hf_cache`**: DACVAE codec / tokenizer などの HF hub キャッシュを永続化。
- **`./models`**: ベースモデル (`model.safetensors`) を置く場所。未マウント / 未配置なら `base_hf_repo` から取りに行きます。
- **`./data/LoRA`**: 話者 `.safetensors` を置く場所。ここに新しいファイルを追加したら**コンテナの再起動が必要**（起動時にだけスキャンするため）。

---

## 6. API

### `GET /health`

```json
{"status": "ok", "speakers": 3}
```

### `GET /speakers`

登録済み話者の一覧。UUID を拾って `/synth` で使います。

```json
{
  "speakers": [
    {
      "uuid": "00000000-0000-0000-0000-000000000000",
      "name": "<話者表示名>",
      "defaults": {"num_steps": 40, "cfg_scale_text": 3.0, "cfg_scale_speaker": 5.0}
    }
  ]
}
```

### `POST /synth`

リクエスト:

```json
{
  "speaker_id": "00000000-0000-0000-0000-000000000000",
  "text": "こんにちは、今日はいい天気ですね。",
  "seed": 42,
  "num_steps": 40,
  "cfg_scale_text": 3.0,
  "cfg_scale_speaker": 5.0,
  "speaker_kv_scale": null,
  "truncation_factor": null
}
```

| フィールド          | 必須 | 説明 |
|---------------------|------|------|
| `speaker_id`        | ◯    | `/speakers` で返る UUID |
| `text`              | ◯    | 合成するテキスト |
| `seed`              | 任意 | 省略 / `<0` でランダム |
| `num_steps`         | 任意 | RF サンプリングステップ。省略 / `<=0` で speaker default |
| `cfg_scale_text`    | 任意 | テキスト CFG scale |
| `cfg_scale_speaker` | 任意 | 話者 CFG scale |
| `speaker_kv_scale`  | 任意 | `>1` で話者性を強める |
| `truncation_factor` | 任意 | 例: `0.8`。ノイズトランケーション |

省略した項目は LoRA metadata の `defaults` → サーバ内部の既定値 (`num_steps=40`, `cfg_scale_text=3.0`, `cfg_scale_speaker=5.0`) の順にフォールバックします。

レスポンス: `audio/wav` バイナリ。ヘッダに

- `X-TTS-Speaker-Id`
- `X-TTS-Speaker-Name` (URL-encoded)
- `X-TTS-Used-Seed`
- `X-TTS-Sample-Rate`

が付きます。固定で 30 秒長の波形が返るので、無音末尾が気になる場合はクライアント側で trim してください。

### `GET /demo`

ブラウザから叩ける簡易 UI。話者プルダウン + テキスト入力 + 任意パラメータで即試聴できます。`http://localhost:8765/demo` を開くだけ。

---

## 7. 新しい話者の追加フロー

1. LoRA を学習して `outputs/<speaker>_lora/checkpoint_best_val_loss_*` を得る（`docs/TRAINING.md` 参照）。
2. `samples/` の per-step wav を聴いて採用する checkpoint を決める。
3. `scripts/lora/export_lora_to_safetensors.py` で `.safetensors` にエクスポートし、`--name` / `--defaults` を埋め込む。
4. `.safetensors` を `data/LoRA/` に置く。
5. サーバを再起動（`docker compose restart tts` など）。
6. `GET /speakers` で新 UUID を確認、`POST /synth` で動作確認。

---

## 8. トラブルシュート

| 症状                                                 | 対処 |
|------------------------------------------------------|------|
| 起動時 `lora_dir does not exist`                     | `lora_dir` の解決先を確認。Docker なら `./data/LoRA` が正しくマウントされているか |
| `skipping non-LoRA safetensors file`                 | `format=irodori-tts-lora/v1` が入っていない。`export_lora_to_safetensors.py` 経由で書き出す |
| 話者一覧に出ているのに `/synth` が 404 を返す         | UUID をコピペミスしていないか確認（`/speakers` の値をそのまま使う） |
| GPU を認識しない                                     | `--gpus all` / compose の `deploy.resources.reservations.devices` を確認 |
| ベースモデル pull で 401/403                         | private repo の場合は `HF_TOKEN` を環境変数に入れる |
| 音質が学習時サンプルより悪い                         | `defaults` の `num_steps` / `cfg_scale_*` を調整、または別の checkpoint をエクスポートし直す |
