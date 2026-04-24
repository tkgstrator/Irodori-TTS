# Voice Drama Script (VDS) v1 仕様

**バージョン:** 1
**ステータス:** 正式
**最終更新:** 2026-04-24

Voice Drama Script (VDS) は、複数話者・複数セリフのボイスドラマを順次合成・再生するための入力フォーマットである。テキスト形式（`.vds`）と JSON 形式（VDS-JSON）の 2 つの等価な表現を持ち、パーサはどちらも同一の内部表現（`VdsScript`）に変換する。

---

## 目次

1. [用語](#1-用語)
2. [テキスト形式（.vds）](#2-テキスト形式vds)
3. [JSON 形式（VDS-JSON）](#3-json-形式vds-json)
4. [Annotation（Shortcode）](#4-annotationshortcode)
5. [セマンティクス](#5-セマンティクス)
6. [出力形式](#6-出力形式)
7. [バリデーション](#7-バリデーション)
8. [制限事項と将来拡張](#8-制限事項と将来拡張)

---

## 1. 用語

| 用語 | 定義 |
|------|------|
| **cue** | 合成の最小単位。speech / pause / scene のいずれか |
| **speech cue** | テキストを 1 人の話者で合成する指示 |
| **pause cue** | 指定秒数の無音を挿入する指示 |
| **scene cue** | シーン区切りのメタデータ。v1 では合成・再生に影響しない |
| **alias** | 台本中で話者を指し示す短い識別子 |
| **gap** | 連続する speech cue 間に自動挿入される無音 |
| **SpeakerRef** | 話者の実体参照。LoRA UUID または caption（自然文記述） |
| **shortcode** | `{name}` 形式のインライン記法。音声のスタイル・効果を指定する |

---

## 2. テキスト形式（.vds）

### 2.1 ファイル

- 拡張子: `.vds`
- 文字コード: UTF-8（BOM あり/なし いずれも可）
- 改行: LF / CRLF（パース時に LF に正規化）

### 2.2 行構造

1 行は以下のいずれかに分類される。先頭・末尾の空白はトリムされる。

| 種別 | パターン | 処理 |
|------|----------|------|
| 空行 | 空白のみ | 無視 |
| コメント | `#` で始まる | 無視 |
| ディレクティブ | `@` で始まる | §2.3 |
| pause | `(pause <seconds>)` | §2.5 |
| speech cue | `<alias>[<options>]: <text>` | §2.4 |

認識できない行はパースエラーとなる。

### 2.3 ディレクティブ

`@<key>` に続く値で構成される。

#### `@version: <int>`

**必須。** フォーマットバージョン。v1 では `1` のみ受理。省略はパースエラー。

```
@version: 1
```

#### `@title: <text>`

任意。台本のタイトル。合成動作には影響しない。

```
@title: 夜明けの対話
```

#### `@speaker <alias> = <ref>`

話者の定義。`<alias>` は `[A-Za-z_][A-Za-z0-9_-]*` に適合する ASCII 識別子。

**LoRA 話者**（UUID 指定）:
```
@speaker chieri = 7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb
```

**Caption 話者**（自然文記述）:
```
@speaker young_woman = caption "落ち着いた女性の声で、やわらかく自然に"
```

caption の値はダブルクォートで囲む。エスケープは `\"` と `\\` のみ。

制約:
- 最初の cue より**前**に宣言すること
- 同じエイリアスの再定義は不可
- 1 エイリアスにつき LoRA または caption のいずれか一方

#### `@defaults <key>=<value>, ...`

全 cue に適用されるサンプリングパラメータの既定値。最初の cue より**前**に宣言すること。

使用可能なキー:

| キー | 型 | 説明 |
|------|----|------|
| `gap` | `float` | speech→speech 間の暗黙無音（秒）。デフォルト `1.0`。`0` で無効化 |
| `seed` | `int` | 乱数シード |
| `num_steps` | `int` | サンプリングステップ数 |
| `cfg_scale_text` | `float` | テキスト CFG スケール |
| `cfg_scale_speaker` | `float` | 話者 CFG スケール |
| `speaker_kv_scale` | `float` | 話者 KV スケール |
| `truncation_factor` | `float` | ノイズトランケーション |

未知キーは警告（エラーにしない）。`gap` は負値不可。

```
@defaults num_steps=40, gap=0.3
```

#### `@scene: <text>`

シーン区切り。v1 ではメタデータとして保持されるが、合成・再生に影響しない。cue の前後どこにでも配置可能。

```
@scene: 第1幕 朝の会話
```

### 2.4 Speech cue

```
<alias>: <text>
<alias> [<key>=<value>, ...]: <text>
```

- `<alias>` は `@speaker` で事前定義されたエイリアス
- `:` の前後に空白を許容する
- `<text>` は行末までの任意のテキスト。先頭・末尾の空白はトリムされる。空テキストはエラー
- テキスト中の `#` はコメント開始にならない（行頭の `#` のみがコメント）
- テキスト中の `{shortcode}` はパーサがアノテーションとして展開する（§4）
- **1 cue は必ず 1 行**。継続行は v1 では未サポート

**オプション** は `[key=value, key=value]` の形式で alias と `:` の間に記述する。使用可能なキーは `seed` / `num_steps` / `cfg_scale_text` / `cfg_scale_speaker` / `speaker_kv_scale` / `truncation_factor`。未知キーはエラー。値は数値リテラル（`seed=-1` のように負値も可）。

### 2.5 Pause

```
(pause <seconds>)
```

`<seconds>` は正の数値リテラル（整数または浮動小数）。0 以下はエラー。

### 2.6 完全な例

```vds
@version: 1
@title: 夜明けの対話
@speaker chieri = 7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb
@speaker young_woman = caption "落ち着いた女性の声で、やわらかく自然に読み上げてください。"
@defaults num_steps=40, gap=0.3

@scene: 朝の挨拶
chieri: {cheerful}おはよう、つむぎ。
# ↑ {cheerful} で明るい声色になる
# ↑↓ speech→speech 間にデフォルト gap（0.3秒）の無音
young_woman: おはようございます、ちえりさん。

(pause 0.8)
# ↑ 明示的 pause があるので gap は挿入されない

chieri [seed=42, cfg_scale_text=3.5]: {whisper}今日はね、特別な日なんだ。
# ↑ {whisper} でささやき声になる

@scene: 沈黙
(pause 1.0)
(pause 0.5)
# ↑ 連続 pause は加算される（1.5秒の無音）。警告あり

young_woman: {surprised}何かあるんですか、わたし全然聞いてませんでした。
```

---

## 3. JSON 形式（VDS-JSON）

テキスト形式と意味的に等価な JSON 表現。API ペイロードや機械生成に適する。

### 3.1 スキーマ

```typescript
interface VdsJson {
  version: 1;                                  // 必須
  title?: string;                              // 任意
  defaults?: Defaults;                         // 任意
  speakers: Record<string, SpeakerRef>;        // 必須
  cues: Cue[];                                 // 必須
}

interface Defaults {
  gap?: number;               // デフォルト: 1.0（秒）。非負
  seed?: number;
  num_steps?: number;
  cfg_scale_text?: number;
  cfg_scale_speaker?: number;
  speaker_kv_scale?: number;
  truncation_factor?: number;
}

type SpeakerRef =
  | { type: "lora"; uuid: string }
  | { type: "caption"; caption: string };

type Cue =
  | SpeechCue
  | PauseCue
  | SceneCue;

interface SpeechCue {
  kind: "speech";
  speaker: string;            // speakers に定義済みのエイリアス
  text: string;               // 非空。{shortcode} はパーサが展開する（§4）
  options?: SynthOptions;     // 任意
}

interface SynthOptions {
  seed?: number;
  num_steps?: number;
  cfg_scale_text?: number;
  cfg_scale_speaker?: number;
  speaker_kv_scale?: number;
  truncation_factor?: number;
}

interface PauseCue {
  kind: "pause";
  duration: number;           // 正の数値（秒）
}

interface SceneCue {
  kind: "scene";
  name: string;               // 非空
}
```

**厳密性:** 各オブジェクトの未知フィールドはエラーとする（`additionalProperties: false`）。

### 3.2 エイリアス制約

`speakers` のキーは `[A-Za-z_][A-Za-z0-9_-]*` に適合すること。

### 3.3 完全な例

```json
{
  "version": 1,
  "title": "夜明けの対話",
  "defaults": { "num_steps": 40, "gap": 0.3 },
  "speakers": {
    "chieri":      { "type": "lora", "uuid": "7c9e6a55-5b6a-4a4d-9c49-1d5a3b2f6cbb" },
    "young_woman": { "type": "caption", "caption": "落ち着いた女性の声で、やわらかく自然に読み上げてください。" }
  },
  "cues": [
    { "kind": "scene",  "name": "朝の挨拶" },
    { "kind": "speech", "speaker": "chieri",      "text": "{cheerful}おはよう、つむぎ。" },
    { "kind": "speech", "speaker": "young_woman",  "text": "おはようございます、ちえりさん。" },
    { "kind": "pause",  "duration": 0.8 },
    { "kind": "speech", "speaker": "chieri",      "text": "{whisper}今日はね、特別な日なんだ。",
      "options": { "seed": 42, "cfg_scale_text": 3.5 } },
    { "kind": "scene",  "name": "沈黙" },
    { "kind": "pause",  "duration": 1.0 },
    { "kind": "pause",  "duration": 0.5 },
    { "kind": "speech", "speaker": "young_woman",  "text": "{surprised}何かあるんですか、わたし全然聞いてませんでした。" }
  ]
}
```

---

## 4. Annotation（Shortcode）

VDS では speech cue のテキスト中に `{shortcode}` 形式のインライン記法を記述することで、音声のスタイルや効果を制御できる。パーサが shortcode を上流モデルが認識する内部表現に展開してから合成に渡す。利用者は展開後の表現を意識する必要はない。

### 4.1 構文

- 形式: `{shortcode_name}`
- パターン: `\{([a-z_]+)\}` に適合する部分を展開対象とする
- speech cue のテキスト中の任意の位置に記述可能
- 1 つの cue に複数の shortcode を含めることができる
- 絵文字を直接テキストに書くこともできる（shortcode は省略形）

### 4.2 展開タイミング

shortcode の展開はパース時に行われる。テキスト形式・JSON 形式いずれでも、speech cue のテキストフィールドに対して適用される。パーサの出力（`VdsScript`）には展開済みの内部表現が格納される。

### 4.3 未知の shortcode

マッピングに存在しない shortcode はそのまま残される（エラーにならない）。

```
{unknown_code}テスト → {unknown_code}テスト（そのまま）
```

### 4.4 Shortcode 一覧

上流 [EMOJI_ANNOTATIONS.md](https://huggingface.co/Aratako/Irodori-TTS-500M/blob/main/EMOJI_ANNOTATIONS.md) 準拠。全 39 種。

| Shortcode | 効果 |
|-----------|------|
| `{angry}` | 怒って、不機嫌に |
| `{anxious}` | 不安そうに、心配げに |
| `{backchannel}` | 相槌 |
| `{cheerful}` | 明るく、朗らかに |
| `{chuckle}` | くすくす笑い、忍び笑い |
| `{cough}` | 咳、鼻すすり、くしゃみ |
| `{cry}` | すすり泣き、悲しげに |
| `{drunk}` | 酔って |
| `{echo}` | エコー、リバーブ |
| `{exasperated}` | 呆れて |
| `{fast}` | 早口、急いで |
| `{gasp}` | 息を呑む |
| `{gentle}` | やさしく、丁寧に |
| `{gulp}` | ごくり、嚥下音 |
| `{heavy_breath}` | 息切れ、荒い呼吸 |
| `{humming}` | 鼻歌 |
| `{joyful}` | 嬉しそうに、楽しげに |
| `{kiss}` | リップ音 |
| `{lick}` | 舐め音、咀嚼音、水音 |
| `{muffled}` | くぐもった声 |
| `{painful}` | 苦しげに |
| `{panic}` | 慌てて、動揺、どもり |
| `{pant}` | 荒い息、喘ぎ |
| `{pause}` | 間、沈黙 |
| `{phone}` | 電話越し、スピーカー越し |
| `{pleading}` | 懇願、お願い |
| `{relieved}` | 安堵、満足げに |
| `{scream}` | 叫び、悲鳴 |
| `{shy}` | 恥ずかしそうに |
| `{sigh}` | 吐息、ため息、寝息 |
| `{sleepy}` | 眠そうに、気だるく |
| `{slow}` | ゆっくり |
| `{surprised}` | 驚き、感嘆 |
| `{teasing}` | からかい、甘えた声 |
| `{trembling}` | 震え声、おずおずと |
| `{tsk}` | 舌打ち |
| `{whisper}` | ささやき、耳元で話す |
| `{wondering}` | 疑問、考え込んで |
| `{yawn}` | あくび |

### 4.5 使用例

```vds
chieri: {whisper}ねえ、聞こえる？{gasp}え、何それ？
# → ささやき声で「ねえ、聞こえる？」、息を呑んで「え、何それ？」

chieri: {angry}{scream}やめて！
# → 怒りと叫びの効果が重ねて適用される
```

---

## 5. セマンティクス

### 5.1 処理順序

cue リストを先頭から順に処理する。順序が再生順序を決定する。

### 5.2 Speech cue の処理

1. `speaker` エイリアスを `speakers` 定義から解決
2. テキスト中の `{shortcode}` を展開する（§4、パース時に実行済み）
3. 話者参照の種別に応じて合成経路を選ぶ:
   - `type: "lora"` → `speaker_id` として UUID を指定し合成
   - `type: "caption"` → caption 文字列を指定し VoiceDesign 経路で合成（サーバに `caption_hf_repo` の設定が必要）
4. サンプリングパラメータを解決（§5.6）
5. 合成結果の音声を出力

### 5.3 Gap（暗黙無音）

連続する speech cue の間に、`defaults.gap`（未指定時 `1.0` 秒）の無音が自動挿入される。

- **speech → speech**: gap が挿入される
- **speech → pause → speech**: gap は挿入されない（pause が間を制御する）
- **speech → scene → speech**: scene はスキップされるため、実質 speech → speech として gap が挿入される
- `gap: 0` で無効化

### 5.4 Pause cue の処理

指定秒数の無音を挿入する。

- 連続する pause は**加算**される（例: `pause 1.0` + `pause 0.5` = 1.5 秒の無音）
- 連続 pause に対しては警告が出る
- 先頭・末尾の pause も許容（再生前後の無音になる）

### 5.5 Scene cue の処理

v1 ではメタデータとして保持されるのみで、合成・再生には影響しない。パーサの出力に含まれるが、ランタイムはスキップする。

### 5.6 パラメータ解決順序

speech cue のサンプリングパラメータは、以下の順序でマージされる（後が優先）:

1. **サーバ内部デフォルト** — `num_steps=40`, `cfg_scale_text=3.0`, `cfg_scale_speaker=5.0`
2. **話者 LoRA の defaults** — LoRA ファイルのメタデータに埋め込まれた値
3. **VDS `defaults`** — 台本レベルの既定値
4. **cue `options`** — 個別 cue の上書き

各レイヤーで値が指定されていないキーは、下位レイヤーの値がそのまま残る。

### 5.7 合成失敗時

- 個別 cue の合成が失敗した場合、その cue はスキップし、処理を継続する
- スキップされた cue の前後の pause/gap はそのまま維持される
- 全 cue が失敗した場合はエラー

---

## 6. 出力形式

### 6.1 コンテンツネゴシエーション

サーバは HTTP `Accept` ヘッダに基づき 2 つの出力形式を提供する。

| Accept ヘッダ | Content-Type | 形式 |
|--------------|-------------|------|
| `audio/pcm`（デフォルト） | `audio/pcm` | Raw PCM16 mono |
| `audio/wav` | `audio/wav` | WAV ファイル（PCM16 subtype） |

`Accept` ヘッダに `audio/wav` が含まれない場合は PCM がデフォルトとなる。

### 6.2 PCM 形式

- **エンコーディング:** 符号付き 16-bit 整数（int16）、リトルエンディアン
- **チャンネル:** モノラル（1ch）
- **サンプルレート:** 24000 Hz（レスポンスヘッダ `X-TTS-Sample-Rate` で通知）
- **ヘッダなし:** WAV ヘッダ等は含まれない、純粋な PCM バイト列

無音（pause / gap）は `\x00\x00` を `duration * sample_rate` 個繰り返すことで表現する。

### 6.3 WAV 形式

標準的な WAV ファイル（RIFF ヘッダ付き）。PCM16 subtype、モノラル、24000 Hz。

### 6.4 ドラマモードの出力

- **PCM モード:** `StreamingResponse` による逐次配信。各 cue の合成完了時点で即座にバイト列を yield する。低レイテンシで再生開始できる。
- **WAV モード:** 全 cue の合成完了後に結合し、単一の WAV ファイルとして返却する。

### 6.5 レスポンスヘッダ

| ヘッダ | 説明 | モード |
|--------|------|--------|
| `X-TTS-Sample-Rate` | サンプルレート（整数、`"24000"`） | 共通 |
| `X-TTS-Used-Seed` | 実際に使用されたシード値 | 単一 cue |
| `X-TTS-Speaker-Id` | 話者 UUID | 単一 cue（LoRA） |
| `X-TTS-Speaker-Name` | URL エンコードされた話者表示名 | 単一 cue（LoRA） |
| `X-TTS-Cue-Count` | speech cue の総数 | ドラマモード |

### 6.6 API エンドポイント

#### `POST /synth`

統合エンドポイント。リクエストボディの内容に応じて単一 cue モードまたはドラマモードで動作する。

**単一 cue モード（LoRA）:** `speaker_id` + `text` を指定。

**単一 cue モード（Caption）:** `caption` + `text` を指定（`speaker_id` と排他）。

**ドラマモード:** `script` フィールドに VDS-JSON オブジェクトを指定。`speaker_id` / `text` / `caption` は無視される。

#### `POST /synth/vds`

`.vds` テキストファイルのアップロードエンドポイント。`multipart/form-data` で `file` フィールドにテキストファイルを添付する。UTF-8（BOM あり/なし）をサポート。パース後の処理は `/synth` のドラマモードと同一。

---

## 7. バリデーション

### 7.1 パースエラー（致命的）

以下はパース時に検出し、処理を中止する。テキスト形式では行番号を付与する。

| 条件 |
|------|
| `@version` が省略されている |
| `version` が `1` 以外 |
| 未定義のエイリアスを cue で使用 |
| 不明なディレクティブ |
| cue オプションに未知キー |
| cue オプション値が数値でない |
| `(pause)` の秒数が非正または非数値 |
| cue のテキストが空 |
| `@speaker` / `@defaults` が最初の cue より後に出現 |
| 同一エイリアスの再定義 |
| エイリアスが `[A-Za-z_][A-Za-z0-9_-]*` に適合しない |
| JSON: 各オブジェクトに未知フィールドがある |
| JSON: `speakers` / `cues` の型が不正 |
| JSON: `SpeakerRef` の `type` が `"lora"` / `"caption"` 以外 |

### 7.2 警告（処理は継続）

| 条件 |
|------|
| cue テキストが 120 文字超（30 秒を超える可能性） |
| 連続する pause（意図しない重複の可能性） |
| 定義済みだが未使用のエイリアス |
| `defaults` に未知キー（将来互換） |
| speech cue 数が 100 を超えている |

### 7.3 実行時エラー

| 条件 | HTTP ステータス | 挙動 |
|------|----------------|------|
| UUID がサーバの話者一覧に存在しない | 404 | ドラマモードでは事前検証で検出 |
| caption 話者を caption 未構成のサーバに投入 | 501 | `caption_hf_repo` が設定されていない |
| caption 文字列がサーバ側の上限超過 | 500 | サーバエラーをそのまま伝搬 |
| `speaker_id` と `caption` を同時に指定 | 422 | 排他制約違反 |
| `speaker_id` も `caption` も未指定（単一 cue モード） | 422 | いずれか必須 |
| `text` 未指定（単一 cue モード） | 422 | 必須フィールド |
| ドラマモードで speech cue が 0 件 | 422 | 合成対象なし |

---

## 8. 制限事項と将来拡張

以下は v1 では**対象外**とし、`@version: 2` 以降で検討する。

| 項目 | 備考 |
|------|------|
| 日本語エイリアス | Unicode 識別子の正規化問題を回避するため ASCII 限定 |
| 継続行（複数行 cue） | 1 cue = 1 行。長いテキストは書き手が分割する |
| 自動テキスト分割 | 30 秒超テキストの文境界自動分割 |
| BGM / 効果音 | ミキシング・ダッキング |
| 並列話者 | 2 人が重ねて話す表現 |
| 速度・ピッチ調整 | 上流モデルにパラメータなし |
| SSML 風プロソディ | 上流モデルに相当機能なし |
| シーン単位の制御 | `@scene` は v1 でメタデータ導入済み。シーン単位のエクスポート・defaults 上書きは v2 |
| プリセット | `@preset name = key=value, ...` 形式の名前付きオプション束 |
