# Irodori-TTS LoRA 学習ガイド

任意の話者について、Irodori-TTS-500M-v2 をベースに LoRA ファインチューニングするためのエンドツーエンド手順です。学習はコンテナ一発で回せるように設計されています。

---

## 1. データセットフォーマット

1 HuggingFace dataset repo = 全話者。repo 直下に話者名のサブディレクトリを並べ、それぞれに manifest + latents を置きます。

```
<repo root>/
├── margo/
│   ├── manifest.jsonl   # 必須: 学習マニフェスト
│   └── latents/         # 必須: DACVAE 事前エンコード済み潜在
│       ├── 00000000_00000000.pt
│       └── ...
├── leia/
│   ├── manifest.jsonl
│   └── latents/
└── ...
```

### `manifest.jsonl`

1 行 1 サンプルの JSONL。必須キー:

| key          | type   | description                                                 |
|--------------|--------|-------------------------------------------------------------|
| `text`       | str    | 書き起こしテキスト                                          |
| `latent_path`| str    | 話者ディレクトリからの相対パス。例: `latents/00000000_00000000.pt` |
| `num_frames` | int    | 潜在のフレーム数(時間方向長さ)                            |

例:
```json
{"text": "こんにちは、お元気ですか。", "latent_path": "latents/00000000_00000000.pt", "num_frames": 120}
```

### `latents/*.pt`

- DACVAE (`Aratako/Semantic-DACVAE-Japanese-32dim`) のエンコーダ出力を `torch.save` したもの。
- shape は `(latent_dim=32, num_frames)` で、1 サンプル = 1 ファイル。
- 生 wav はアップロード不要(学習ループは潜在だけ読みます)。

> **作り方**: ローカルで wav を揃えたあと、`prepare_manifest.py` を使うと `latents/` と `manifest.jsonl` の両方を一度に生成できます。

```bash
uv run --no-sync python prepare_manifest.py \
  --metadata   data/<speaker>/metadata.jsonl \
  --audio-root data/<speaker>/wavs \
  --output     data/<speaker>/manifest.jsonl \
  --latent-dir data/<speaker>/latents
```

### `config.yaml`（話者メタデータ）

各話者ディレクトリに `config.yaml` を置いておくと、前処理（LLM クリーニング）と学習 config 生成（`scripts/train/make_speaker_config.py`）の両方がこれを読みます。ファイルはユーザーが自分で作成します — repo には含まれていません。

```yaml
speaker:
  id: ema                    # ディレクトリ名と一致させる
  name: 桜羽エマ              # 正規フルネーム（任意）

cleaning:
  first_person: ボク          # 話者の規定一人称
  addressing: chan           # chan / san / kun / yobisute のいずれか

sample_texts:                # checkpoint A/B 試聴用の固定プロンプト
  - こんにちは、今日はいい天気ですね。
  - あのさ、ちょっと相談したいことがあるんだけど、今いい？
  - えっ、本当に？それって信じられないよ。
  - ごめんなさい、少し考える時間をもらえますか。
  - わかった、それじゃあ後で連絡するね。
```

- **`cleaning.first_person`** — 話者が使う一人称。LLM クリーニングは音響的に近い変種（例: `私`⇄`わたし`⇄`あたし`、`僕`⇄`ボク`、`わがはい`⇄`我輩`/`吾輩`）だけを canonical 形に寄せる。音響的に遠い語（`俺` → `私`）は誤認識ではなく別の原因の可能性が高いので置換しない。
- **`cleaning.addressing`** — 他話者への呼びかけ形式:
  - `chan` / `san` / `kun` — 該当の敬称が付いていれば人名の強い手掛かりになる
  - `yobisute` — 敬称を付けずに呼び捨てる。バッチ中の裸の名前も呼称である可能性があるという判定に使う
- **`sample_texts`** — `make_speaker_config.py` が `sample_generation.prompts` に展開する。話者の特徴が出る台詞を入れると ckpt 比較がしやすい。空だとエラーになる。

### HuggingFace へのアップロード

同梱の `scripts/dataset/hf_upload_dataset.py` で、`data/` 配下の全話者(または `--speakers` で指定したサブセット)を 1 つの repo にまとめて push できます。

```bash
# 全話者をまとめて push
uv run --no-sync python scripts/dataset/hf_upload_dataset.py \
  --repo-id <org>/irodori-tts-voices --private

# 一部の話者だけ push / 追加
uv run --no-sync python scripts/dataset/hf_upload_dataset.py \
  --repo-id <org>/irodori-tts-voices --speakers margo,leia --private
```

`HF_TOKEN` を環境変数 or `.env` に入れておけば自動で認証されます。1 話者あたり目安 10〜30MB。wav・ログ・中間ファイルは除外されます。

---

## 2. 学習コンテナのビルド

```bash
docker build -f docker/train/Dockerfile -t irodori-tts-train .
# もしくは compose 経由
docker compose build train
```

ベースイメージは `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`、依存は `uv.lock` に完全にピン留めされています。ベースチェックポイント (約 1.9GB) はイメージに焼かず、初回起動時に HuggingFace から自動ダウンロードします(デフォルト: `Aratako/Irodori-TTS-500M-v2`)。

---

## 3. 学習の実行

### 3.1 環境変数

| 変数              | 必須 | 説明                                                                                                 |
|-------------------|------|------------------------------------------------------------------------------------------------------|
| `HF_TOKEN`        | ◯    | HF private dataset を pull するためのトークン                                                        |
| `WANDB_API_KEY`   | 任意 | 指定すると online mode で W&B に自動ログインされ、進捗報告される                                     |
| `HF_DATASET`      | △    | 学習前に pull する HF dataset repo ID(全話者入りの単一 repo)。未指定なら `data/` にあるものをそのまま使う |
| `SPEAKERS`        | 任意 | カンマ区切りの話者名。`HF_DATASET` と併用するとその話者だけ pull。学習時も `SPEAKERS` に列挙された話者だけが対象(未指定なら `data/` 配下の全話者) |
| `BASE_MODEL_REPO` | 任意 | ベースチェックポイント repo。デフォルト: `Aratako/Irodori-TTS-500M-v2`                               |
| `BASE_CKPT`       | 任意 | ベースチェックポイントのコンテナ内パス                                                               |
| `NUM_GPUS`        | 任意 | 利用する GPU の数(整数)。指定すると index `0..NUM_GPUS-1` を自動割当                              |
| `GPUS`            | 任意 | 利用する GPU index をスペース区切り(例: `"0 1 2 3"`)。`NUM_GPUS` より優先。未指定なら見えてる GPU 全部 |

### 3.2 `compose.yaml` の例

学習サーバ側に `compose.yaml` を 1 個置いて、`.env` に `HF_TOKEN` / `WANDB_API_KEY` を入れるだけで回せる構成です。イメージは GHCR から pull する想定(自前ビルドする場合は `build:` セクションに差し替え)。

```yaml
services:
  train:
    image: tkgling/irodori-tts-train:latest
    # 自前ビルドするなら image: の代わりに:
    # build:
    #   context: .
    #   dockerfile: docker/train/Dockerfile
    env_file:
      - .env
    environment:
      # ---- 必須: 全話者入りの HF dataset repo ----
      HF_DATASET: ultemica/irodori-tts-voices
      # ---- 任意: 学習対象のサブセット指定(指定された話者だけを pull & 学習)----
      # SPEAKERS: margo,leia
      # ---- 任意: GPU 数 ----
      NUM_GPUS: 8
      # ---- 任意: 明示的な GPU index ----
      # GPUS: "0 2 4 6"
      HF_HOME: /root/.cache/huggingface
    volumes:
      # 永続化用の named volume
      - venv:/app/.venv                         # uv sync 結果を持ち回す
      - hf_cache:/root/.cache/huggingface       # tokenizer / codec などの HF キャッシュ
      # ホスト側ディレクトリをマウント
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  venv:
  hf_cache:
```

`.env`:
```
HF_TOKEN=hf_xxx
WANDB_API_KEY=xxxxxxxx
```

起動:
```bash
docker compose run --rm train
```

- **`venv` volume**: 初回起動時に `entrypoint.sh` が `uv sync --frozen --no-dev` を走らせて `/app/.venv` を構築します。named volume に載せておけば 2 回目以降は `uv sync` が数秒で終わり、即 training に入れます。イメージ自体は venv を焼き込まない軽量構成なので pull/push も速いです。
- **`hf_cache` volume**: tokenizer / Semantic-DACVAE codec など HF hub の取得物がキャッシュされるので、コンテナを作り直しても再ダウンロードが走りません。
- `./data` をマウントしておくと、次回以降は同じ潜在を再ダウンロードせずに済みます。
- `./outputs` には話者ごとに `outputs/<speaker>_lora/{checkpoint_best_val_loss_*, train.log, samples/}` が出ます。
- `./models` にベースチェックポイントが落ちてくるので、マウントしておけば全話者・全コンテナ再実行で使い回せます。
- `HF_DATASET` を外せばダウンロードをスキップし、既にマウント済みの `data/<speaker>/` をそのまま学習します。`SPEAKERS` も空なら `data/` 配下の全話者が対象です。
- `NUM_GPUS` を指定すると index `0..N-1` が使われます。話者は **LPT (Longest Processing Time first) 貪欲法**で `manifest.jsonl` の行数に基づき GPU に割り当てられ、総データ件数がおおむね均等になるようバランスされます(例: 2GPU で `100/100/100/500` → `(500)` と `(100,100,100)` に分割)。特定の index を明示したい場合は `GPUS` を使ってください。

---

## 4. コンテナ内部の処理フロー

`scripts/entrypoint.sh` が以下を順に実行します。

1. ベースチェックポイントを HF から pull(既に存在すればスキップ)
2. `HF_DATASET` から dataset repo を pull(`SPEAKERS` 指定時はそれに該当するサブディレクトリだけを `allow_patterns` で pull)
3. 学習対象話者を決定(`SPEAKERS` → 未指定なら `data/` を自動スキャン)
4. 各話者の `manifest.jsonl` と `latents/` の存在をチェック
5. 話者ごとの config `configs/train_500m_v2_<speaker>_lora.yaml` を必要に応じて生成 (`scripts/train/make_speaker_config.py`)
6. `scripts/train/train_multi_speaker.sh` で GPU 並列起動(LPT 貪欲法で各 GPU に話者を負荷分散、manifest の行数に基づきおおむね均等になるよう割り当てる)

各 GPU は `CUDA_VISIBLE_DEVICES` で 1 話者に pin され、stdout/stderr は `outputs/<speaker>_lora/train.log` に追記されます。

---

## 5. ハイパラとスケジューラ

デフォルトは `configs/train_500m_v2_speaker_lora.yaml`。主要値:

| 項目                        | 値                                        |
|-----------------------------|-------------------------------------------|
| `max_epochs`                | 200(早期停止前提の上限)                |
| `batch_size`                | 8                                         |
| `gradient_accumulation_steps`| 4                                        |
| `learning_rate`             | 1e-4                                      |
| `lora_r` / `lora_alpha`     | 32 / 64                                   |
| `speaker_condition_dropout` | 1.0(テキストのみで学習、inference で話者指定不要) |
| `lr_scheduler`              | WSD                                       |
| `warmup_ratio`              | 0.02                                      |
| `decay_ratio`               | 0.10                                      |
| `valid_ratio`               | 0.05(`[50, 100]` 件に clamp)            |
| `valid_every`               | 50 ステップ                               |
| `early_stop_enabled`        | `true`                                    |
| `early_stop_min_step`       | 1000                                      |
| `early_stop_patience`       | 20(val eval 回数)                      |
| `early_stop_min_delta`      | 1e-3                                      |
| `early_stop_regression_ratio`| 0.10                                     |

`warmup_ratio` / `decay_ratio` を指定すると、`max_epochs × ステップ/epoch` から導出された `max_steps` に比例して warmup / stable / decay の step 数が自動計算されます(公式の base 学習と同じ 2% / 88% / 10% が初期値)。データ件数を変えても再チューニング不要です。

### 環境変数でハイパラを上書き

テンプレートをいじらず、環境変数だけで主要ハイパラを上書きできます(設定した値が `train.py` の CLI フラグに変換されて全話者の run に適用)。

| 環境変数                         | 対応 CLI フラグ                    |
|----------------------------------|------------------------------------|
| `MAX_EPOCHS`                     | `--max-epochs`                     |
| `BATCH_SIZE`                     | `--batch-size`                     |
| `GRADIENT_ACCUMULATION_STEPS`    | `--gradient-accumulation-steps`    |
| `LEARNING_RATE`                  | `--lr`                             |
| `WEIGHT_DECAY`                   | `--weight-decay`                   |
| `WARMUP_RATIO`                   | `--warmup-ratio`                   |
| `DECAY_RATIO`                    | `--decay-ratio`                    |
| `MIN_LR_SCALE`                   | `--min-lr-scale`                   |
| `LORA_R`                         | `--lora-r`                         |
| `LORA_ALPHA`                     | `--lora-alpha`                     |
| `LORA_DROPOUT`                   | `--lora-dropout`                   |
| `TEXT_CONDITION_DROPOUT`         | `--text-condition-dropout`         |
| `SPEAKER_CONDITION_DROPOUT`      | `--speaker-condition-dropout`      |
| `VALID_EVERY`                    | `--valid-every`                    |
| `SAVE_EVERY`                     | `--save-every`                     |
| `CHECKPOINT_BEST_N`              | `--checkpoint-best-n`              |
| `SEED`                           | `--seed`                           |
| `EXTRA_TRAIN_ARGS`               | 上記以外の任意フラグを素通しで追加(例: `"--some-flag 42"`) |

compose の `environment:` で指定する例:
```yaml
environment:
  HF_DATASET: ultemica/irodori-tts-voices
  NUM_GPUS: 8
  MAX_EPOCHS: 80           # 50 -> 80 に延長
  LEARNING_RATE: 5e-5      # 1e-4 -> 5e-5 に下げる
  LORA_R: 64               # rank を上げる
  LORA_ALPHA: 128
```

環境変数で指定された値は config テンプレートより優先されます。一切指定しなければ `configs/train_500m_v2_speaker_lora.yaml` のデフォルトがそのまま使われます。

### 早期停止 (Early Stopping)

`max_epochs=200` を上限として、`val_loss` を監視する保守的な早期停止が組み込まれています。`train.py` は各 validation eval の直後に以下を判定します:

1. **patience**: `val_loss` が `best_val_loss - early_stop_min_delta` を `early_stop_patience` 回連続で更新できなければ停止
2. **regression**: `val_loss > best_val_loss × (1 + early_stop_regression_ratio)` になった時点で即停止
3. どちらも `step >= early_stop_min_step` の条件を満たしている時にだけ発火(初期の不安定期で誤発火しない floor)

wandb には `es/no_improve` / `es/best_val` が各 eval ごとに記録されるので、発火の兆候を事後確認できます。

パラメータ設定のメモ:

- **保守寄りの既定値**(`patience=20`, `regression_ratio=0.10`): flow matching の `val_loss` は t サンプリング分散でノイジーなので、短い patience / 小さい regression_ratio は誤発火しやすい。実データ上の検証でも、既存ランのうち明確に過学習しているケース(ema_lora)のみが regression で捕捉され、継続改善中のケース(hiro_lora)は最後まで走る挙動。
- **val 側の下支え**: `valid_ratio=0.05` + clamp `[50, 100]` で最低 50 件の検証セットを確保。件数が少なすぎると `val_loss` がノイズに引きずられ ES が機能しなくなるため、この clamp は必須。
- **最良 ckpt の選択**: `val_loss` ≠ 聴感品質なので、ES は「発散検知による無駄打ち打ち切り」が主目的。最終的な ckpt 選択は `outputs/<speaker>_lora/samples/` の per-step サンプルを耳で A/B して決めるのが既定運用。

環境変数 `EXTRA_TRAIN_ARGS` で個別に無効化・チューニングすることも可能です(例: `EXTRA_TRAIN_ARGS="--no-early-stop"` 相当のフラグは未実装なので現状 YAML を書き換えるのが確実)。

---

## 6. 成果物

学習完了後、`outputs/<speaker>_lora/` 配下:

```
outputs/<speaker>_lora/
├── best_val_loss.pt       # best n 個まで
├── last.pt                # 最終ステップ
├── train.log
└── samples/               # sample_generation.prompts に従って生成された音声
    ├── step00000200_short.wav
    ├── step00000200_long.wav
    └── ...
```

`samples/` を聴いて checkpoint を決定 → `configs/runtime.yaml` に登録 → `docker compose up tts` で推論サーバから呼べます。

---

## 7. トラブルシュート

| 症状                                             | 対処                                                                                     |
|--------------------------------------------------|------------------------------------------------------------------------------------------|
| `no speakers to train` で即終了                  | `HF_DATASET` を指定するか、`data/` を正しくマウントする                                   |
| `missing config configs/train_500m_v2_<s>_lora.yaml` | 通常は自動生成されるはず。それでも出る場合はベーステンプレートが壊れていないか確認     |
| W&B がオフラインになる                           | `WANDB_API_KEY` を `.env` に入れて再実行                                                 |
| GPU が見えない                                   | `docker run --gpus all` または compose の `deploy.resources` 指定を確認                  |
| ベースチェックポイントが pull できない           | `HF_TOKEN` をセット、もしくは `BASE_MODEL_REPO` を自前のミラーに変更                      |
