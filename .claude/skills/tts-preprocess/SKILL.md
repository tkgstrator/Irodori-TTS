---
name: tts-preprocess
description: Interactive preprocessing pipeline for Irodori-TTS LoRA training data (trim, normalize, filter, transcribe, clean).
---

# tts-preprocess

Prepares audio data for Irodori-TTS LoRA training. Runs an interactive pipeline:

1. **trim** — remove leading/trailing silence with ffmpeg
2. **normalize** — loudness normalization via ffmpeg `loudnorm`
3. **filter** — drop clips outside min/max duration, sequence-rename
4. **transcribe** — faster-whisper large-v3 on each clip
5. **clean** — heuristic cleanup (too-short text, non-verbal vocalizations, English-mistranscriptions, duplicates)

## Invocation behavior

When the skill is invoked, **always start by asking the user** what to process. Even when the invocation includes arguments, explicitly confirm the following before doing any work:

1. **Input source** — one of:
   - An existing flat directory of audio files (e.g. `data/_raw_ema/`)
   - A zip file with a filter pattern (e.g. `data/Voice.zip` matching `Voice/*Ema*.ogg` excluding `*EmaFake*`)
2. **Output directory** — dataset root (e.g. `data/ema`). Audio files are written under `<output_dir>/wavs/` (kept as `.ogg`; the `wavs/` name is conventional). Metadata files go directly under `<output_dir>/`.
3. **Parameters** — present these defaults and ask whether to change any:
   - `--min-seconds 1.5`
   - `--max-seconds 30.0`
   - `--silence-db -40`
   - `--normalize-lufs -23`
   - `--whisper-model large-v3`
   - `--workers 12`

Once the answers are collected, summarize the plan and ask for confirmation before running anything.

## Pipeline details

Working directory: `/home/vscode/app`. Scripts live under `scripts/`.

### Step 1 — stage input

- If the input is a zip: extract the matching entries (flat, with `unzip -j`) into `data/_<name>_raw/`.
- If the input is an existing directory: use it in place.

### Step 2 — trim + normalize + filter

Use `scripts/preprocess/preprocess_audio.py`. Internally it applies this ffmpeg chain:

```
silenceremove=1:0:<silence-db>dB,
areverse,silenceremove=1:0:<silence-db>dB,areverse,
loudnorm=I=<lufs>:TP=-1.5:LRA=11
```

`silenceremove=1:0:...` removes a single leading silent period of any length — this is the correct form that does not cut mid-speech (unlike `stop_periods=1:stop_silence=...` which also stops on internal silences). The reverse/trim/reverse pair handles the trailing silence.

Output: sequence-renamed files `<output_dir>/wavs/00000.ogg`, `00001.ogg`, ... and `<output_dir>/_source_map.tsv` recording the original filename and post-trim duration for each kept clip.

### Step 3 — transcribe

Use `scripts/preprocess/transcribe_dir.py` against `<output_dir>/wavs/`. It transcribes every ogg with faster-whisper large-v3, word-level timestamps, and inserts `、` / `。` based on inter-word gaps.

Output: `<output_dir>/metadata_wts.jsonl` with records `{"file_name": "00000.ogg", "text": "..."}` (file names are relative to `<output_dir>/wavs/`).

### Step 4 — clean

Cleaning is a **two-pass** process. The heuristic filter never rewrites text; the LLM pass handles all context-sensitive rewrites.

**Step 4a — heuristic filter** (`scripts/preprocess/filter_metadata_voice.py`): drops records matching any of:

- Text shorter than 3 characters after normalization
- Repeated-character ratio >= 0.5 (e.g. `ああああ`, `はぁっはぁっ`)
- ASCII-letter ratio >= 0.3 (English misrecognition)
- Non-verbal-only hiragana pattern
- Exact-duplicate text

Writes `<output_dir>/metadata_filtered.jsonl` and `<output_dir>/metadata_rejected.jsonl`. Also re-evaluate non-verbal rejects with the LLM pass before dropping them — some are legitimate short utterances.

**Step 4b — LLM cleaning pass** (Sonnet 4.6, via Agent sub-agents): dispatches batches of ~150 records with two inputs merged at dispatch time:

1. **`.claude/skills/tts-preprocess/voice_cleaning_prompt.md`** — the speaker-agnostic, dataset-agnostic prompt. Covers the acoustic-distance principle for first-person normalization, addressing hints, contextual Japanese error correction, punctuation fixes, and the `suspect:` flag convention. **Do not put speaker- or work-specific facts in this file.**
2. **`data/<speaker>/config.yaml`** — the per-speaker config. Provides canonical `cleaning.first_person` and `cleaning.addressing` (one of `chan` / `san` / `kun` / `yobisute`). This is the single source of truth for speaker-specific info.

Before running the LLM pass on a new speaker, ensure `data/<speaker>/config.yaml` exists. If not, ask the user for the canonical first-person and addressing convention and create it before dispatching agents.

Preserve the original Whisper output as `metadata_wts.jsonl` (never modify). The LLM pass writes a diff file `metadata_llm_diff.jsonl` with `{file_name, original, cleaned, reason}`; show the diff to the user for approval before applying it to produce the final `metadata.jsonl`.

### Step 5 — report

Report counts after each step:

- extract: N files
- preprocess: kept=X, short=Y, long=Z, err=W
- transcribe: N files
- clean: kept=X, rejected=Y

At the end, tell the user what comes next: run `prepare_manifest.py` to encode DACVAE latents, then launch LoRA training.

## Logging

**Always redirect each step's stdout/stderr into `<output_dir>/preprocess.log`** so the user can tail progress in another terminal. Use `tee -a` (append) so every step contributes to the same file. Example:

```
uv run --no-sync python scripts/preprocess/preprocess_audio.py ... 2>&1 | tee -a data/ema/preprocess.log
uv run --no-sync python scripts/preprocess/transcribe_dir.py  ... 2>&1 | tee -a data/ema/preprocess.log
uv run --no-sync python scripts/clean/clean_metadata_auto.py ... 2>&1 | tee -a data/ema/preprocess.log
```

Also emit a header line before each step (e.g. `=== step: transcribe ===`) into the log so the sections are visually separated.

## Out of scope

- `prepare_manifest.py` (DACVAE latent encoding) — run separately on user request
- LoRA training itself — run separately