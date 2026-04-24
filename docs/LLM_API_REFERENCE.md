# Irodori-TTS API リファレンス（LLM / 外部クライアント向け）

このドキュメントは Irodori-TTS サーバの HTTP API を外部から利用するクライアント（Discord Bot 等）向けにまとめたものです。サーバの内部実装や学習パイプラインには触れません。

---

## 接続情報

| 項目 | 値 |
|------|----|
| プロトコル | HTTP |
| デフォルトポート | `8765` |
| Content-Type（リクエスト） | `application/json`（`/synth`）、`multipart/form-data`（`/synth/vds`） |
| Content-Type（レスポンス） | `audio/wav` または `audio/pcm`（後述） |

---

## エンドポイント一覧

| メソッド | パス | 用途 |
|----------|------|------|
| `GET` | `/health` | ヘルスチェック |
| `GET` | `/speakers` | 登録済み話者の一覧取得 |
| `POST` | `/synth` | 音声合成（単発 / ドラマ） |
| `POST` | `/synth/vds` | `.vds` テキストファイルからのドラマ合成 |

---

## GET /health

サーバの生存確認。

```json
{"status": "ok", "speakers": 3, "caption": true}
```

`caption` は VoiceDesign（caption）ランタイムが有効かどうかを示す。

---

## GET /speakers

登録済みの全話者を返す。`uuid` を `/synth` の `speaker_id` として使う。

### レスポンス例

```json
{
  "speakers": [
    {
      "uuid": "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb",
      "name": "ちえり",
      "defaults": {
        "num_steps": 40,
        "cfg_scale_text": 3.0,
        "cfg_scale_speaker": 5.0
      }
    },
    {
      "uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
      "name": "つむぎ",
      "defaults": {
        "num_steps": 50
      }
    }
  ]
}
```

### 話者について

- 話者はサーバ起動時に LoRA ファイルから自動登録される。動的に追加・削除されることはない。
- `uuid` は LoRA ファイルから決定論的に導出される固定値。サーバ再起動しても変わらない。
- `defaults` は話者ごとの推奨サンプリングパラメータ。リクエスト時に省略すると、この値がフォールバック先になる。

---

## POST /synth

音声合成のメインエンドポイント。3 つのモードがある。

### モード判定

| 条件 | モード |
|------|--------|
| `script` フィールドあり | **ドラマモード**（VDS-JSON） |
| `speaker_id` + `text` | **LoRA 単発モード** |
| `caption` + `text` | **VoiceDesign 単発モード** |

`speaker_id` と `caption` は**排他**（両方指定すると 422 エラー）。

---

### LoRA 単発モード

1 つのテキストを LoRA 話者で合成し、WAV を返す。

#### リクエスト

```json
{
  "speaker_id": "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb",
  "text": "こんにちは、今日はいい天気ですね。",
  "seed": 42,
  "num_steps": 40,
  "cfg_scale_text": 3.0,
  "cfg_scale_speaker": 5.0
}
```

| フィールド | 型 | 必須 | 説明 |
|------------|----|------|------|
| `speaker_id` | `string` | **必須** | `/speakers` で返る UUID |
| `text` | `string` | **必須** | 合成するテキスト（1文〜数文、30秒以内に収まる長さ） |
| `seed` | `integer` | 任意 | 乱数シード。省略 / 負値でランダム |
| `num_steps` | `integer` | 任意 | サンプリングステップ数。省略 / `<=0` で話者デフォルト |
| `cfg_scale_text` | `float` | 任意 | テキスト CFG スケール |
| `cfg_scale_speaker` | `float` | 任意 | 話者 CFG スケール |
| `speaker_kv_scale` | `float` | 任意 | `>1` で話者性を強調 |
| `truncation_factor` | `float` | 任意 | ノイズトランケーション（例: `0.8`） |

**パラメータ解決順序**: リクエスト値 → 話者 `defaults` → サーバ内部デフォルト（`num_steps=40`, `cfg_scale_text=3.0`, `cfg_scale_speaker=5.0`）

---

### VoiceDesign 単発モード

`caption`（自然文による話者記述）でモデルに話者特性を指示し合成する。LoRA 話者登録が不要なため、任意のキャラクターを即座に生成できる。`GET /health` の `caption: true` で VoiceDesign ランタイムの有無を確認できる（`false` の場合は 501 エラー）。

#### リクエスト

```json
{
  "caption": "落ち着いた女性の声で、やわらかく自然に",
  "text": "こんにちは、今日はいい天気ですね。",
  "num_steps": 40,
  "cfg_scale_text": 3.0,
  "cfg_scale_caption": 3.0
}
```

| フィールド | 型 | 必須 | 説明 |
|------------|----|------|------|
| `caption` | `string` | **必須** | 自然文による話者記述 |
| `text` | `string` | **必須** | 合成するテキスト |
| `cfg_scale_caption` | `float` | 任意 | caption CFG スケール（デフォルト `3.0`） |
| `seed` / `num_steps` / `cfg_scale_text` / `truncation_factor` | — | 任意 | LoRA モードと同じ |

`speaker_kv_scale` / `cfg_scale_speaker` は VoiceDesign モードでは無視される。

#### レスポンス

- **Content-Type**: `audio/wav`
- **Body**: PCM16 mono WAV バイナリ
- **固定長**: 常に 30 秒分の波形が返る。末尾に無音が含まれるため、クライアント側でトリムすること。

**レスポンスヘッダー**:

| ヘッダー | 例 | 説明 |
|----------|----|------|
| `X-TTS-Speaker-Id` | `7c9e6a55-...` | 使用した話者 UUID |
| `X-TTS-Speaker-Name` | `%E3%81%A1%E3%81%88%E3%82%8A` | 話者名（URL エンコード済み） |
| `X-TTS-Used-Seed` | `42` | 実際に使用されたシード |
| `X-TTS-Sample-Rate` | `24000` | サンプルレート (Hz) |

---

### ドラマモード（VDS-JSON）

複数話者・複数セリフの台本を一括合成する。`script` フィールドに VDS-JSON オブジェクトを渡す。

#### リクエスト

```json
{
  "script": {
    "version": 1,
    "defaults": { "num_steps": 40, "gap": 0.3 },
    "speakers": {
      "alice": { "type": "lora", "uuid": "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb" },
      "bob":   { "type": "lora", "uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" }
    },
    "cues": [
      { "kind": "speech", "speaker": "alice", "text": "おはよう。" },
      { "kind": "speech", "speaker": "bob",   "text": "おはようございます。" },
      { "kind": "pause",  "duration": 1.0 },
      { "kind": "speech", "speaker": "alice", "text": "今日はいい天気ですね。" }
    ]
  }
}
```

**VDS-JSON の構造は後述の「VDS-JSON フォーマット」セクションを参照。**

`script` を指定した場合、トップレベルの `speaker_id` / `text` / その他パラメータは**すべて無視**される。

#### 出力形式の選択（Accept ヘッダー）

| `Accept` ヘッダー | 出力形式 | 用途 |
|----|----|----|
| `audio/pcm`（デフォルト） | RAW PCM16 mono ストリーム | Discord Bot（低レイテンシ逐次再生） |
| `audio/wav` | 全 cue を合成後、gap/pause 込みの結合 WAV | ダウンロード・プレビュー |

`Accept` を省略した場合は `audio/pcm` として扱われる。

#### PCM ストリームレスポンス（`Accept: audio/pcm`）

- **Content-Type**: `audio/pcm`
- **Body**: RAW PCM16 signed little-endian mono のバイトストリーム
- cue ごとに合成完了次第、逐次送出される（chunked transfer）
- speech cue 間のデフォルトギャップ、明示的 pause は無音 PCM として挿入済み
- **クライアントは受信したバイト列をそのまま PCM バッファに追記して再生**できる

再生パラメータ:
- フォーマット: signed 16-bit little-endian
- チャンネル: mono (1ch)
- サンプルレート: `X-TTS-Sample-Rate` ヘッダーの値（通常 `24000` Hz）

#### WAV レスポンス（`Accept: audio/wav`）

- **Content-Type**: `audio/wav`
- **Body**: 全 cue + gap + pause を結合した単一 WAV ファイル
- 全 cue の合成が完了するまでレスポンスは返らない

#### レスポンスヘッダー（共通）

| ヘッダー | 例 | 説明 |
|----------|----|------|
| `X-TTS-Sample-Rate` | `24000` | サンプルレート (Hz) |
| `X-TTS-Cue-Count` | `3` | speech cue の総数 |

---

## POST /synth/vds

`.vds` テキストファイル（プレーンテキスト VDS 形式）をアップロードしてドラマ合成する。出力形式は `/synth` ドラマモードと同じく `Accept` ヘッダーで切り替え。

#### リクエスト

`multipart/form-data` で `file` フィールドに `.vds` ファイルを添付。

```
POST /synth/vds
Content-Type: multipart/form-data; boundary=...

--boundary
Content-Disposition: form-data; name="file"; filename="script.vds"
Content-Type: application/octet-stream

(VDS テキストの中身)
--boundary--
```

#### レスポンス

`/synth` ドラマモードと同一（`Accept` ヘッダーで PCM / WAV を選択）。

---

## VDS-JSON フォーマット

ドラマモードで `/synth` に渡す `script` オブジェクトの仕様。

### トップレベル

```typescript
{
  version: 1,                           // 必須。現在は 1 のみ
  title?: string,                       // 任意。台本タイトル
  defaults?: {                          // 任意。全 cue に適用される既定値
    gap?: number,                       //   speech→speech 間の無音秒数（デフォルト: 1.0）
    num_steps?: number,
    cfg_scale_text?: number,
    cfg_scale_speaker?: number,
    speaker_kv_scale?: number,
    truncation_factor?: number,
    seed?: number
  },
  speakers: {                           // 必須。エイリアス → 話者参照のマップ
    [alias: string]: SpeakerRef
  },
  cues: Cue[]                           // 必須。再生順の cue 配列
}
```

### SpeakerRef

```typescript
// LoRA 話者（現行 API で使用可能）
{ "type": "lora", "uuid": "<UUID>" }

// Caption 話者（VoiceDesign ランタイムが必要）
{ "type": "caption", "caption": "<自然文による話者記述>" }
```

`type: "lora"` は常に使用可能。`type: "caption"` は VoiceDesign ランタイムが有効な場合のみ使用可能（`GET /health` の `caption: true` で確認）。

### Cue

```typescript
// 音声合成
{
  "kind": "speech",
  "speaker": "<alias>",         // speakers に定義済みのエイリアス
  "text": "<合成テキスト>",       // 空文字不可。30秒以内に収まる長さ推奨
  "options"?: {                  // 任意。この cue 専用のパラメータ上書き
    "seed"?: number,
    "num_steps"?: number,
    "cfg_scale_text"?: number,
    "cfg_scale_speaker"?: number,
    "speaker_kv_scale"?: number,
    "truncation_factor"?: number
  }
}

// 無音挿入
{
  "kind": "pause",
  "duration": <正の数値（秒）>    // 例: 0.5, 1.0, 2.5
}

// シーン区切り（v1 では合成に影響しない、メタデータのみ）
{
  "kind": "scene",
  "name": "<シーン名>"
}
```

### ギャップと pause の関係

- **speech → speech** の間に暗黙のギャップ（`defaults.gap`、デフォルト 1.0 秒）が自動挿入される
- 間に **pause** がある場合、ギャップは挿入されない（pause が代替する）
- 連続する pause は**加算**される
- `gap: 0` でギャップ無効化

### パラメータ解決順序（ドラマモード）

cue ごとに以下の順でマージされる（後が優先）:

1. サーバ内部デフォルト（`num_steps=40`, `cfg_scale_text=3.0`, `cfg_scale_speaker=5.0`）
2. 話者 LoRA の `defaults`（`/speakers` で確認可能）
3. VDS-JSON の `defaults` ブロック
4. 各 cue の `options`

### エイリアス命名規則

`[A-Za-z_][A-Za-z0-9_-]*`（ASCII 英数字、アンダースコア、ハイフン。先頭は英字またはアンダースコア）

---

## エラーレスポンス

すべてのエラーは JSON 形式で返る。

```json
{
  "detail": "<エラーメッセージ>"
}
```

| HTTP ステータス | 原因 |
|----------------|------|
| `404` | `speaker_id` が未登録 / VDS 内の UUID が不明 |
| `422` | バリデーションエラー（必須フィールド欠落、VDS パースエラー等） |
| `500` | 合成処理の内部エラー |
| `501` | caption 話者を指定したが API 未対応 |

VDS パースエラーの `detail` には行番号が含まれる（例: `"line 5: undefined speaker alias: 'unknown'"`）。

---

## Discord Bot 実装のための注意点

### 推奨フロー

1. 起動時に `GET /speakers` で話者 UUID 一覧を取得・キャッシュ
2. ユーザーから VDS テキスト or 構造化データを受け取る
3. VDS-JSON を組み立てて `POST /synth` に `Accept: audio/pcm` で送信
4. レスポンスの PCM ストリームを受信しながら Discord の音声チャンネルに流す

### PCM ストリーム再生

```
サーバ: [cue1 PCM][gap 無音][cue2 PCM][pause 無音][cue3 PCM]...
         ↓ chunked transfer で逐次到着
Bot:     受信 → PCM バッファ追記 → AudioPlayer で再生
```

- `X-TTS-Sample-Rate` ヘッダーでサンプルレートを確認（通常 24000 Hz）
- バイト列はそのまま signed 16-bit LE mono PCM
- 最初のチャンクが届いた時点で再生開始できる（全 cue の合成完了を待つ必要なし）

### タイムアウト

- 1 cue あたり最大 30〜60 秒かかる想定
- ドラマモードでは `cue数 × 30秒` 程度の余裕を見てタイムアウトを設定する
- PCM ストリームは chunked なので、HTTP クライアントの読み取りタイムアウト（idle timeout）に注意

### 合成失敗時の挙動

- ドラマモードで個別の cue の合成が失敗した場合、その cue はスキップされ、ストリームは継続する
- 前後の pause/gap はそのまま維持される
- 全 cue が失敗した場合のみ 500 エラー

---

## リクエスト例

### LoRA 単発合成

```bash
curl -s http://localhost:8765/synth \
  -H 'Content-Type: application/json' \
  -d '{"speaker_id":"7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb","text":"こんにちは"}' \
  -o output.wav
```

### VoiceDesign 単発合成（caption）

```bash
curl -s http://localhost:8765/synth \
  -H 'Content-Type: application/json' \
  -d '{"caption":"落ち着いた女性の声","text":"こんにちは"}' \
  -o output.wav
```

### ドラマ — PCM ストリーム

```bash
curl -s http://localhost:8765/synth \
  -H 'Content-Type: application/json' \
  -H 'Accept: audio/pcm' \
  -d '{
    "script": {
      "version": 1,
      "defaults": {"num_steps": 40, "gap": 0.3},
      "speakers": {
        "alice": {"type": "lora", "uuid": "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb"},
        "bob":   {"type": "lora", "uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}
      },
      "cues": [
        {"kind": "speech", "speaker": "alice", "text": "おはよう。"},
        {"kind": "speech", "speaker": "bob",   "text": "おはようございます。"},
        {"kind": "pause",  "duration": 1.0},
        {"kind": "speech", "speaker": "alice", "text": "今日はいい天気ですね。"}
      ]
    }
  }' \
  -o output.pcm

# PCM → WAV 変換
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

### ドラマ — 結合 WAV

```bash
curl -s http://localhost:8765/synth \
  -H 'Content-Type: application/json' \
  -H 'Accept: audio/wav' \
  -d '{
    "script": {
      "version": 1,
      "speakers": {
        "alice": {"type": "lora", "uuid": "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb"}
      },
      "cues": [
        {"kind": "speech", "speaker": "alice", "text": "おはようございます。"},
        {"kind": "pause",  "duration": 0.5},
        {"kind": "speech", "speaker": "alice", "text": "今日はいい天気ですね。"}
      ]
    }
  }' \
  -o drama.wav
```

### .vds ファイルアップロード

```bash
# PCM ストリーム
curl -s http://localhost:8765/synth/vds \
  -F 'file=@script.vds' \
  -o output.pcm

# 結合 WAV
curl -s http://localhost:8765/synth/vds \
  -H 'Accept: audio/wav' \
  -F 'file=@script.vds' \
  -o drama.wav
```
