---
name: tts-train
description: Interactive LoRA training pipeline for Irodori-TTS speaker adapters (manifest prep, DACVAE latents, training launch).
---

# tts-train

Trains a speaker LoRA adapter for Irodori-TTS from a preprocessed dataset (the output of `tts-preprocess`). Runs an interactive pipeline:

1. **verify dataset** — ensure `metadata.jsonl` and audio files exist
2. **encode latents** — run `prepare_manifest.py` to encode DACVAE latents and emit `train_manifest.jsonl`
3. **write training config** — materialize a per-speaker YAML derived from `configs/train_500m_v2_speaker_lora.yaml`
4. **launch training** — start `train.py` with resume support and checkpoint retention

## Invocation behavior

When the skill is invoked, **always start by asking the user** what to train. Even when arguments are supplied, explicitly confirm:

1. **Dataset directory** — directory containing `metadata.jsonl` + audio files (e.g. `data/ema`).
2. **Speaker name** — short identifier used for output directories and run names (e.g. `ema`).
3. **Base model checkpoint** — default `models/Irodori-TTS-500M-v2/model.safetensors`. Confirm it exists; otherwise ask whether to use HF (`Aratako/Irodori-TTS-500M-v2`) or a different path.
4. **Output directory** — default `outputs/<speaker>_lora`.
5. **Config overrides** — present these defaults (from `configs/train_500m_v2_speaker_lora.yaml`) and ask whether any should change:
   - `--max-steps 5000`
   - `--batch-size 8`
   - `--gradient-accumulation-steps 4`
   - `--lr 1e-4`
   - `--lora-r 32`, `--lora-alpha 64`
   - `--save-every 500`
   - `--valid-ratio 0.01`
   - `--wandb-project irodori-tts-speaker-lora`
6. **Resume** — ask whether to resume from an existing checkpoint.

Summarize the plan and get explicit confirmation before launching anything long-running.

## Pipeline details

Working directory: `/home/vscode/app`.

### Step 1 — verify dataset

Check that `<dataset_dir>/metadata.jsonl` exists and references ogg files under `<dataset_dir>/wavs/` that resolve. Report the number of records. Abort early if missing or inconsistent.

### Step 2 — encode DACVAE latents

Run `prepare_manifest.py` against the dataset. Audio lives in `<dataset_dir>/wavs/`, metadata in `<dataset_dir>/metadata.jsonl`. The script encodes each clip into `<dataset_dir>/latents/*.pt` and writes `<dataset_dir>/train_manifest.jsonl` with `{text, latent_path, num_frames}` records (so `latent_path` is `latents/<id>.pt` relative to `<dataset_dir>`).

This step is **GPU-bound** and can take several minutes for thousands of clips. Run it in the foreground and report progress. Skip it if the manifest already exists unless the user explicitly asks to rebuild.

### Step 3 — write training config

Start from `configs/train_500m_v2_speaker_lora.yaml` and write a derived file `configs/train_500m_v2_<speaker>_lora.yaml` with any user-specified overrides (run name, max_steps, lora params, etc.). Do not modify the base config in place.

### Step 4 — launch training

Invoke `train.py` with flags matching the derived config:

```
uv run python train.py \
  --config configs/train_500m_v2_<speaker>_lora.yaml \
  --manifest <dataset_dir>/train_manifest.jsonl \
  --latent-root <dataset_dir> \
  --output-dir outputs/<speaker>_lora \
  --base-checkpoint <base_checkpoint> \
  [--resume <checkpoint_path>]
```

For long runs, start the process in the background and tail the log. Report step/loss at reasonable intervals (e.g. every few hundred steps or on user request).

**Always redirect stdout/stderr into `<output_dir>/train.log`** (e.g. `outputs/ema_lora/train.log`) using `tee -a` so both the user and the assistant can tail it. Include a header line before each launch (e.g. `=== launch: 2026-04-13 03:05 ===`) for readability. The `prepare_manifest.py` step should log into the same file (or a sibling `<dataset_dir>/preprocess.log` if the user prefers); prefer keeping latent-encoding logs with the dataset and training logs with the output dir.

**Early stopping policy (user preference)**: watch the validation loss reported at each `valid_every` step. On prior cherry LoRA runs, the best-sounding checkpoint was the one with the lowest `val_loss`, and quality degraded as training continued past that point (overfitting). Once val_loss stops improving for 2+ validation windows, stop training and use the best `checkpoint_best_val_loss_*` entry instead of running to `max_steps`. Prefer stopping early over completing the configured schedule.

### Step 5 — post-train

When training finishes (or the user stops it), list the checkpoints under `outputs/<speaker>_lora/` including any `checkpoint_best_val_loss_*` entries and recommend next actions:

- Register the adapter in `configs/runtime.yaml` under `speakers:` with a fresh UUID and restart the server
- Synthesize samples with the new adapter to sanity-check quality
- Convert to merged safetensors only if needed (the server loads base + LoRA directly)

## Out of scope

- Data cleaning and transcription — handled by `tts-preprocess`
- Server deployment — handled manually by editing `configs/runtime.yaml` and restarting
- Distillation, quantization, or other model surgery