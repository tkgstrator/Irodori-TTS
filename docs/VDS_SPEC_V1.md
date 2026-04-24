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
4. [セマンティクス](#4-セマンティクス)
5. [バリデーション](#5-バリデーション)
6. [制限事項と将来拡張](#6-制限事項と将来拡張)

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
chieri: おはよう、つむぎ。
# ↑↓ speech→speech 間にデフォルト gap（0.3秒）の無音
young_woman: おはようございます、ちえりさん。

(pause 0.8)
# ↑ 明示的 pause があるので gap は挿入されない

chieri [seed=42, cfg_scale_text=3.5]: 今日はね、特別な日なんだ。

@scene: 沈黙
(pause 1.0)
(pause 0.5)
# ↑ 連続 pause は加算される（1.5秒の無音）。警告あり

young_woman: 何かあるんですか、わたし全然聞いてませんでした。
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
  text: string;               // 非空
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
    { "kind": "speech", "speaker": "chieri",      "text": "おはよう、つむぎ。" },
    { "kind": "speech", "speaker": "young_woman",  "text": "おはようございます、ちえりさん。" },
    { "kind": "pause",  "duration": 0.8 },
    { "kind": "speech", "speaker": "chieri",      "text": "今日はね、特別な日なんだ。",
      "options": { "seed": 42, "cfg_scale_text": 3.5 } },
    { "kind": "scene",  "name": "沈黙" },
    { "kind": "pause",  "duration": 1.0 },
    { "kind": "pause",  "duration": 0.5 },
    { "kind": "speech", "speaker": "young_woman",  "text": "何かあるんですか、わたし全然聞いてませんでした。" }
  ]
}
```

---

## 4. セマンティクス

### 4.1 処理順序

cue リストを先頭から順に処理する。順序が再生順序を決定する。

### 4.2 Speech cue の処理

1. `speaker` エイリアスを `speakers` 定義から解決
2. 話者参照の種別に応じて合成経路を選ぶ:
   - `type: "lora"` → `speaker_id` として UUID を指定し合成
   - `type: "caption"` → caption 文字列を指定し合成（**現行 API 未対応、実行時エラー**）
3. サンプリングパラメータを解決（§4.5）
4. 合成結果の音声を出力

### 4.3 Gap（暗黙無音）

連続する speech cue の間に、`defaults.gap`（未指定時 `1.0` 秒）の無音が自動挿入される。

- **speech → speech**: gap が挿入される
- **speech → pause → speech**: gap は挿入されない（pause が間を制御する）
- **speech → scene → speech**: scene はスキップされるため、実質 speech → speech として gap が挿入される
- `gap: 0` で無効化

### 4.4 Pause cue の処理

指定秒数の無音を挿入する。

- 連続する pause は**加算**される（例: `pause 1.0` + `pause 0.5` = 1.5 秒の無音）
- 連続 pause に対しては警告が出る
- 先頭・末尾の pause も許容（再生前後の無音になる）

### 4.5 パラメータ解決順序

speech cue のサンプリングパラメータは、以下の順序でマージされる（後が優先）:

1. **サーバ内部デフォルト** — `num_steps=40`, `cfg_scale_text=3.0`, `cfg_scale_speaker=5.0`
2. **話者 LoRA の defaults** — LoRA ファイルのメタデータに埋め込まれた値
3. **VDS `defaults`** — 台本レベルの既定値
4. **cue `options`** — 個別 cue の上書き

各レイヤーで値が指定されていないキーは、下位レイヤーの値がそのまま残る。

### 4.6 Scene cue の処理

v1 ではメタデータとして保持されるのみで、合成・再生には影響しない。パーサの出力に含まれるが、ランタイムはスキップする。

### 4.7 合成失敗時

- 個別 cue の合成が失敗した場合、その cue はスキップし、処理を継続する
- スキップされた cue の前後の pause/gap はそのまま維持される
- 全 cue が失敗した場合はエラー

---

## 5. バリデーション

### 5.1 パースエラー（致命的）

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

### 5.2 警告（処理は継続）

| 条件 |
|------|
| cue テキストが 120 文字超（30 秒を超える可能性） |
| 連続する pause（意図しない重複の可能性） |
| 定義済みだが未使用のエイリアス |
| `defaults` に未知キー（将来互換） |
| speech cue 数が 100 を超えている |

### 5.3 実行時エラー

| 条件 | 挙動 |
|------|------|
| UUID がサーバの話者一覧に存在しない | HTTP 404。ドラマモードでは事前検証で検出 |
| caption 話者を caption 未対応 API に投入 | HTTP 501 |
| caption 文字列がサーバ側の上限超過 | サーバエラーをそのまま伝搬 |

---

## 6. 制限事項と将来拡張

以下は v1 では**対象外**とし、`@version: 2` 以降で検討する。

| 項目 | 備考 |
|------|------|
| 日本語エイリアス | Unicode 識別子の正規化問題を回避するため ASCII 限定 |
| 継続行（複数行 cue） | 1 cue = 1 行。長いテキストは書き手が分割する |
| 自動テキスト分割 | 30 秒超テキストの文境界自動分割 |
| 単一結合 WAV エクスポート | v1 はストリーミング再生が基本。API レベルでは `Accept: audio/wav` で対応済み |
| BGM / 効果音 | ミキシング・ダッキング |
| 並列話者 | 2 人が重ねて話す表現 |
| 速度・ピッチ調整 | 上流モデルにパラメータなし |
| SSML 風プロソディ | 上流モデルに相当機能なし |
| シーン単位の制御 | `@scene` は v1 でメタデータ導入済み。シーン単位のエクスポート・defaults 上書きは v2 |
| プリセット | `@preset name = key=value, ...` 形式の名前付きオプション束 |
| 絵文字 shortcode 展開 | `{whisper}` → 👂 のようなインライン記法。マッピング定義方式は v2 で決定 |
