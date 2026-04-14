---
name: tts-clean
description: Clean Whisper-transcribed metadata.jsonl for Irodori-TTS training (heuristic reject + LLM rewrite pass).
---

# tts-clean

Whisper large-v3 で字起こしされた `metadata_wts.jsonl` を、Irodori-TTS の LoRA 学習に使える品質まで整えるための独立スキル。`tts-preprocess` のステップ4を切り出したもので、既に字起こし済みのデータセットに対して単独で走らせられる。

## Invocation behavior

スキル起動時は、たとえ引数があっても **必ず最初に以下を確認** する:

1. **入力ファイル** — 対象の `metadata_wts.jsonl`（例: `data/ema/metadata_wts.jsonl`）。`file_name` + `text` の JSONL 形式であること。
2. **データセットルート** — 出力先。通常は入力ファイルと同じディレクトリ(例: `data/ema/`)。
3. **話者** — `data/<speaker>/config.yaml` が存在すること。新規話者なら先に作成する必要がある（`speaker.id/name`、`cleaning.first_person`、`cleaning.addressing` を定義）。
4. **ヒューリスティックパラメータ**（デフォルト提示 → 変更の有無を確認）:
   - `--min-chars 3`
   - `--rep-threshold 0.5`
   - `--ascii-threshold 0.3`
5. **LLM パスの実行可否** — ヒューリスティックのみで止めるのか、LLM リライトまで進めるのか。

回答が揃ったらプランを要約して確認を取ってから実行する。

## 前提

- 入力 `metadata_wts.jsonl` は **絶対に改変しない**（Whisper の生出力を保持）。
- クリーニング結果は別ファイルとして書き出し、差分を承認してから最終 `metadata.jsonl` を生成する。
- 作業ディレクトリ: `/home/vscode/app`。スクリプトは `scripts/` 配下。

## パイプライン

### Step A — ヒューリスティックフィルタ

`scripts/preprocess/filter_metadata_voice.py` を使い、機械的に落とせるレコードを除外する。**テキストの書き換えは一切しない。**

落とす条件:

- 正規化後テキストが `--min-chars` 未満
- 同一文字の占有率が `--rep-threshold` 以上（`ああああ` / `はぁっはぁっ` など）
- ASCII 比率が `--ascii-threshold` 以上（英語誤認識）
- 非言語音のみ（`あはは` / `ふぅー` など hiragana+記号のみ）
- テキスト完全重複

出力:

- `<root>/metadata_filtered.jsonl` — 残ったレコード
- `<root>/metadata_rejected.jsonl` — 落としたレコード + 理由

**non-verbal で落とした件は LLM パスで再評価する。** 短い正当な発話（`うん。`/`はい。`/`えっ？`）が混ざっていることがある。

### Step B — LLM リライトパス

Sonnet 4.6 を Agent サブエージェントで呼び、汎用プロンプト `.claude/skills/tts-preprocess/voice_cleaning_prompt.md` と当該話者の `data/<speaker>/config.yaml` を合わせて渡す。`metadata_filtered.jsonl` を ~150 レコード単位のバッチに分けて並列処理する。

汎用プロンプトは話者非依存で、次の判定原則だけをカバーする:

1. バッチ通読 → 文脈ベースの誤字修正
2. 一人称の音響距離判定（config の `cleaning.first_person` を当てはめる）
3. 呼称ヒント（config の `cleaning.addressing`: chan/san/kun/yobisute）
4. 句読点・文末記号（`、` / `。` / `？` / `！`）の補完
5. 低確度レコードの `suspect:` フラグ

**話者固有情報（正規名、一人称、呼称規約）は config.yaml が単一のソース・オブ・トゥルース**。汎用プロンプトには特定作品・特定キャラの情報を書かない。新規話者を処理する前に `data/<speaker>/config.yaml` が存在することを確認し、無ければユーザーに一人称・呼称を聞いて作成してから dispatch する。

各サブエージェントの起動時には:
- 汎用プロンプト `voice_cleaning_prompt.md` の全文
- 対象バッチファイル
- 出力 diff ファイル名
- **対象話者の `data/<speaker>/config.yaml` の全文**（yaml そのまま貼る）

を渡す。

各サブエージェントは **変更のあったレコードのみ** を `{file_name, original, cleaned, reason}` 形式で出力する。結果は集約して:

- `<root>/metadata_llm_diff.jsonl`

に書き出す。

### Step C — 差分レビュー → 最終適用

`metadata_llm_diff.jsonl` をユーザーに提示して承認を取る。具体的には:

- 変更件数 / 総件数
- 代表的な変更例（先頭 20 件程度）
- 理由別の内訳（名前正規化 / 句読点 / 誤字修正 …）

承認後、`metadata_filtered.jsonl` に diff を適用して `<root>/metadata.jsonl` を生成する。`metadata_wts.jsonl` はそのまま残す。

### Step D — レポート

最終的に以下を報告:

- heuristic: kept=X, rejected=Y（理由内訳）
- llm: changed=A / total=B
- final: `<root>/metadata.jsonl` のレコード数

次のステップ（`prepare_manifest.py` で DACVAE latent を用意 → LoRA 学習）を案内する。

## Logging

各ステップの stdout/stderr を `<root>/clean.log` に `tee -a` で追記する。ステップ境界には `=== step: <name> ===` のヘッダ行を出力する。

```
uv run --no-sync python scripts/preprocess/filter_metadata_voice.py ... 2>&1 | tee -a data/ema/clean.log
```

LLM パスはサブエージェント実行なので、集約結果と各バッチの報告(`batch_NNN: N changed out of M`)をログに残す。

## Out of scope

- 字起こし自体（`transcribe_dir.py`）— 別途 `tts-preprocess` で
- DACVAE latent 生成・LoRA 学習 — ユーザー指示で別途実行
- `voice_cleaning_prompt.md` の大幅改訂 — 話者追記以外は別タスク