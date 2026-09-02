---
name: tts-train
description: Interactive LoRA training pipeline for Irodori-TTS speaker adapters (manifest prep, DACVAE latents, training launch).
---

# tts-train

Trains a speaker LoRA adapter for Irodori-TTS from a preprocessed dataset (the output of `tts-preprocess`). Runs an interactive pipeline:

1. **verify dataset** — ensure `metadata.jsonl` and audio files exist
2. **encode latents** — run `prepare_manifest.py` to encode DACVAE latents and emit `manifest.jsonl`
3. **pick the training config** — normally the shared `configs/train_v4_small_lora.yaml`; no per-speaker YAML is needed
4. **launch training** — start `train.py` with resume support and checkpoint retention

## Which base model / config generation

The config and the base checkpoint are a **pair**, and they must be chosen together:

| Generation | Config | Base checkpoint |
|---|---|---|
| **v4 (default)** | `configs/train_v4_small_lora.yaml` | `Aratako/Irodori-TTS-v4.1-Small` |
| v3 (legacy) | `configs/train_500m_v3/lora/default.yaml` | `Aratako/Irodori-TTS-500M-v3` |
| v2 (legacy) | `configs/train_500m_v2/lora/default.yaml` | `Aratako/Irodori-TTS-500M-v2` |

Why the pairing matters: the config's `model:` section **is** the architecture. `train.py --help` exposes no flag for any of those fields — not `model_dim`, not `text_encoder_type`, not `use_caption_condition`, not `duration_architecture` — so the config file is the only way to get the architecture right. Pairing the wrong config with a checkpoint either fails to load outright or loads as something other than what was intended. Never mix a row.

Note the path shape differs between rows. v2 and v3 each have a per-generation directory with `lora/default.yaml` inside it; v4's LoRA config is the single top-level file `configs/train_v4_small_lora.yaml`. There is no `configs/train_v4_small/lora/` directory — do not go looking for one.

Two things to know about the v4 row:

- The file's header comment names `Irodori-TTS-v4-Small`, but all 39 of its `model:` fields match the **v4.1** checkpoint exactly. v4 and v4.1 have byte-identical architecture; the v4.1 release replaced and retrained only the duration predictor weights, and the v4-Small model card itself recommends using v4.1 instead. So the config is correct for v4.1 despite the comment. Use v4.1.
- The executable launchers (`docker/train/entrypoint.sh`, `scripts/train/train_multi_speaker.sh`, `scripts/train/stream_pipeline.sh`) still default to the v3 config and the v3 checkpoint. All three honor `CONFIG` and `BASE_CKPT` environment overrides, so driving v4 through a launcher means setting both; otherwise invoke `train.py` directly as in step 4.

A LoRA adapter is only valid against the generation it was trained on. If the user asks for v2 or v3, confirm they also intend to run it against the matching base at inference time — `configs/runtime.yaml` decides which base the server loads, so check it before promising an adapter is servable. `docs/TRAINING.md` is still written against the v2 flow; treat it as out of date on anything version-specific.

## What is different when training on v4

Compared with `configs/train_500m_v3/lora/default.yaml`, these are the differences that change what the operator does or sees. The rest is tuning, not behavior.

1. **Caption conditioning is on** — `use_caption_condition: true` with `caption_tokenizer_repo` / `caption_dim: 512` / `caption_add_bos: true`, and `max_caption_len: 512`, `caption_condition_dropout: 0.1`. v3 had no caption branch at all. See "Do captions and speaker ids need to be in the dataset?" below before promising anything about it.
2. **The text encoder is a pretrained ModernBERT** — `text_encoder_type: pretrained`, `text_tokenizer_repo: sbintuitions/modernbert-ja-310m` pinned at revision `77675fc96a7e445e982e2ba90246b816efc74ec6`, with `pretrained_projector_type: residual_mlp`. v3 trained its own text stack (`text_layers: 10`, `text_heads: 8`). Two operator-visible consequences. First, `train.py` fetches that tokenizer and its `AutoConfig` from HuggingFace during startup, so the repo must be reachable or already in the HF cache. Second, the backbone *weights* are not fetched — when `--init-checkpoint` points at a checkpoint that carries a pretrained encoder, `train.py` takes the weights and the encoder config from that checkpoint instead. So `--init-checkpoint` is not optional in practice on v4. The backbone gets its own `pretrained_text_encoder_learning_rate: 1e-5`, separate from `learning_rate`.
3. **There is no automatic early stopping.** The v3 config set `early_stop_enabled: true` with a patience backstop. The v4 config sets no `early_stop_*` key at all, and the dataclass default in `irodori_tts/config.py` is `early_stop_enabled: false`. On v4 nothing will stop the run before `max_steps`, so the manual policy in step 4 is the only stopping mechanism. This makes watching `val_loss` more important than it was on v3, not less.
4. **There is no sample generation.** The v3 config carried a `sample_generation:` block with `enabled: true`; the v4 config has no such block, and `SampleGenerationConfig.enabled` defaults to `false`. A v4 run therefore writes **no** `samples/` directory, and no CLI flag can turn it on — `train.py --help` exposes nothing for it. To get audio samples during training, copy the config and add a `sample_generation:` block; prompts are still auto-filled from `<dataset_dir>/config.yaml`'s `sample_texts`, or picked from the manifest, when `prompts:` is omitted.
5. **W&B is off by default.** v3 had `wandb_enabled: true` and `wandb_project: ${WANDB_PROJECT:irodori-tts-v3}`, expanded from the environment at load time. The v4 config has `wandb_enabled: false`, an empty `wandb_project`, and no `${...}` expansion anywhere. Logging to W&B on v4 means passing `--wandb` and `--wandb-project <name>` explicitly; exporting `WANDB_PROJECT` alone no longer does anything.
6. **Long reference sampling** — `ref_min_seconds: 1.0`, `ref_max_seconds: 120.0`. Each step concatenates several same-speaker clips into one long reference. This engages only when the manifest carries `speaker_id`; see below.
7. Architecture differences with no operator-facing action, listed so they are not mistaken for something to tune: `speaker_patch_size: 4` (v3: 1), a dual-adaRN duration predictor (`duration_architecture: token_sum_dual_adarn_zero_no_aux` with `duration_caption_fusion: adarn_zero`, versus v3's single-adaRN speaker-only form), and `duration_loss_weight: 1.0` (v3: 0.1). Do not carry v3 values onto these.

### Do captions and speaker ids need to be in the dataset?

Determined by reading `prepare_manifest.py` and `irodori_tts/dataset.py`:

- **Captions are optional, and their absence is silent.** `prepare_manifest.py --caption-column <col>` copies a source column into the manifest under the key `caption`; the flag is optional, and the datasets `tts-preprocess` produces have no such column. When the key is absent, `_caption_candidates` returns empty, the sample gets `has_caption=False`, and the collator zeroes that row's caption mask. Nothing errors. The real consequence is that the caption branch receives no training signal from this data, so the adapter learns nothing caption-specific. That is perfectly fine for a plain speaker LoRA. **Say this to the user rather than letting them assume otherwise.** If they want the adapter to respond to captions, the source data needs a caption/style column and `--caption-column` has to be passed at manifest time.
- **`speaker_id` gates reference conditioning, and its absence is also silent.** `prepare_manifest.py --speaker-column <col>` is what writes `speaker_id`. Without it, `speaker_labeled_count` is 0, `speaker_group_offsets` stays `None`, and every sample trains with `has_speaker=False` — the speaker/reference branch and the `ref_min_seconds`/`ref_max_seconds` concat never engage. This was equally true on v3, so it is not a v4 regression, but v4 leans on that branch more. `tts-preprocess` emits no speaker column, so a dataset built by that skill has none.

On `speaker_id` the user's position is that it exists to tell speakers apart, not as a quality lever, so a single-speaker dataset does not need one and step 2 should not stop to ask. Default to omitting `--speaker-column` for single-speaker training.

Be aware that this is not purely a labelling choice, and say so if it becomes relevant. Omitting it means every sample trains with `has_speaker=False`, so the adapter learns the voice into its own weights with the reference branch inert. Supplying one constant value instead puts every clip in a single group and trains the reference and long-reference concat path. Both are coherent regimes; which yields the better adapter is an empirical question the code cannot answer. Raise it only if the user is chasing a quality problem or wants reference conditioning specifically.

## Invocation behavior

When the skill is invoked, **always start by asking the user** what to train. Even when arguments are supplied, explicitly confirm:

1. **Dataset directory** — directory containing `metadata.jsonl` + audio files (e.g. `data/ema`).
2. **Speaker name** — short identifier used for output directories and run names (e.g. `ema`).
3. **Base model checkpoint** — default `models/Irodori-TTS-v4.1-Small/model.safetensors`. Confirm it exists; otherwise ask whether to pull from HF (`Aratako/Irodori-TTS-v4.1-Small`) or use a different path. The launchers download it with `huggingface_hub.snapshot_download` restricted to `model.safetensors`, `*.json` and `*.md`; the `*.json` matters because the checkpoint's bundled tokenizer assets ride in it. See "Which base model / config generation" above before offering v3 or v2.
4. **Output directory** — default `outputs/<speaker>_lora`.
5. **Config overrides** — present these defaults (from `configs/train_v4_small_lora.yaml`) and ask whether any should change. The CLI flag on the left overrides the config value on the right:
   - `--max-steps 30000`
   - `--batch-size 40` with `--gradient-accumulation-steps 2`, i.e. an effective batch of 80. This pairing is tuned for large-VRAM cards. Ask the user what GPU this run lands on and lower `--batch-size` (raising `--gradient-accumulation-steps` to compensate) if it will not fit; do not assume it does.
   - `--train-mode rf`
   - `--lr 1e-4`
   - `--pretrained-text-encoder-learning-rate 1e-5` — separate LR for the ModernBERT backbone. The main scheduler multiplier applies to it too. Must be > 0 or `train.py` raises.
   - `--lora-r 16`, `--lora-alpha 32`
   - `--lora-target-modules diffusion_attn`
   - `--max-text-len 256`, `--max-caption-len 512`
   - `--duration-loss-weight 1.0`
   - `--ref-min-seconds 1.0`, `--ref-max-seconds 120.0`
   - `--save-every 1000`
   - `--checkpoint-best-n 5`
   - `--valid-ratio 0.0005`, `--valid-every 1000` (v3 used 200; v4 validates five times less often)
   - `--wandb` — required to enable W&B at all, since the v4 config has `wandb_enabled: false`.
   - `--wandb-project <name>` — the v4 config leaves this empty and does no `${VAR}` expansion, so exporting `WANDB_PROJECT` is **not** enough; pass the flag.
   - `--wandb-run-name <speaker>_lora_v4` — the launchers set this per speaker; the config leaves it empty.
6. **Resume** — ask whether to resume from an existing checkpoint.

Summarize the plan and get explicit confirmation before launching anything long-running.

## Pipeline details

Working directory: `/home/vscode/app`.

### Step 1 — verify dataset

Check that `<dataset_dir>/metadata.jsonl` exists and that the `file_name` values in it resolve under `<dataset_dir>/wavs/`. Despite the directory name, `tts-preprocess` writes **ogg** files into `wavs/`; that is the intended layout, not a mistake. Report the number of records. Abort early if missing or inconsistent.

The `metadata.jsonl` + sibling `wavs/` layout is load-bearing for the next step: `prepare_manifest.py` only auto-derives the audio root when the single local data file is literally named `metadata.jsonl` and a `wavs/` directory sits next to it.

### Step 2 — encode DACVAE latents

`prepare_manifest.py` reads through HuggingFace `datasets`, so a local dataset is fed to it as a JSON dataset via `--data-files`. It takes no dataset-directory argument:

```
uv run --no-sync python prepare_manifest.py \
  --dataset json \
  --data-files "train=<dataset_dir>/metadata.jsonl" \
  --split train \
  --audio-column audio --text-column text \
  --target-sample-rate 44100 \
  --output-manifest <dataset_dir>/manifest.jsonl \
  --latent-dir <dataset_dir>/latents \
  --device cuda
```

`--audio-column audio` is intentional even though `metadata.jsonl` has no `audio` column: the script detects the missing column, finds `file_name` plus the sibling `wavs/`, and synthesizes the audio paths itself.

Two optional flags decide what v4's extra conditioning branches get to see, and both must be set here — there is no way to add them later without re-running this step:

- `--caption-column <col>` writes a `caption` key into the manifest. Omit it and caption conditioning trains on nothing (silently).
- `--speaker-column <col>` writes `speaker_id`, which is what enables reference/long-reference conditioning. Omit it and that branch trains on nothing (silently). `--speaker-id-prefix` namespaces the value; it defaults to the dataset name.

Read "Do captions and speaker ids need to be in the dataset?" above and settle both with the user before running this command.

Output: `<dataset_dir>/latents/<written>_<source_idx>.pt` (e.g. `00000000_00000123.pt`) and `<dataset_dir>/manifest.jsonl` with `{text, latent_path, num_frames}` records. `latent_path` is stored relative to the **manifest's own directory** (e.g. `latents/00000000_00000123.pt`), and that is also how training resolves it — there is no latent-root flag or config key anywhere. Keep the manifest and the `latents/` directory as siblings, or move them together.

This step is **GPU-bound** and can take several minutes for thousands of clips. Run it in the foreground and report progress. Skip it if the manifest already exists unless the user explicitly asks to rebuild.

### Step 3 — pick the training config

For the v4 flow there is no per-speaker YAML: every speaker shares `configs/train_v4_small_lora.yaml`. Apply any user-requested overrides as CLI flags rather than by copying the YAML.

The one thing that cannot be done with a flag is sample generation, which the v4 config does not enable at all (see "What is different when training on v4"). If the user wants audio samples during the run, copy the config to a run-specific path and add a `sample_generation:` block there — leaving `prompts:` out lets `train.py` fill it from `<dataset_dir>/config.yaml`'s `sample_texts`, or auto-pick length-balanced prompts from the manifest when that file is absent. Do not edit the shared config in place.

Only if the user has explicitly chosen the legacy v2 flow: run `uv run python scripts/train/make_speaker_config.py <speaker>` to derive `configs/train_500m_v2/lora/<speaker>.yaml` from the v2 template (it defaults to reading `data/<speaker>/manifest.jsonl`; pass `--manifest` otherwise). This script is hardcoded to the v2 template and produces nothing usable for v3 or v4. Do not modify the base template in place.

### Step 4 — launch training

Invoke `train.py`. The base checkpoint goes through `--init-checkpoint` (there is no `--base-checkpoint`), and the launchers pass it on resume as well:

```
uv run --no-sync python train.py \
  --config configs/train_v4_small_lora.yaml \
  --manifest <dataset_dir>/manifest.jsonl \
  --output-dir outputs/<speaker>_lora \
  --init-checkpoint models/Irodori-TTS-v4.1-Small/model.safetensors \
  [--wandb --wandb-project <name> --wandb-run-name <speaker>_lora_v4] \
  [--resume <checkpoint_path>]
```

`--resume` restores full training state (optimizer, scheduler, dataloader position) from a checkpoint; `--init-checkpoint` only seeds weights and starts a fresh run. On v4 keep passing `--init-checkpoint` on resume as well, the way the launchers do: the pretrained encoder path reads the backbone weights and encoder config from it, and resuming a pretrained-encoder run without the checkpoint's `model_config` metadata is rejected outright. Pin the run to one GPU with `CUDA_VISIBLE_DEVICES=<n>` when other work shares the machine.

For long runs, start the process in the background and tail the log. Report step/loss at reasonable intervals (e.g. every few hundred steps or on user request).

**Always redirect stdout/stderr into `<output_dir>/train.log`** (e.g. `outputs/ema_lora/train.log`) using `tee -a` so both the user and the assistant can tail it. Include a header line before each launch (e.g. `=== launch: 2026-04-13 03:05 ===`) for readability. The `prepare_manifest.py` step should log into the same file (or a sibling `<dataset_dir>/preprocess.log` if the user prefers); prefer keeping latent-encoding logs with the dataset and training logs with the output dir.

**Early stopping policy (user preference)**: watch the validation loss reported at each `valid_every` step. On prior cherry LoRA runs, the best-sounding checkpoint was the one with the lowest `val_loss`, and quality degraded as training continued past that point (overfitting). Once val_loss stops improving for 2+ validation windows, stop training and use the best `checkpoint_best_val_loss_*` entry instead of running to `max_steps`. Prefer stopping early over completing the configured schedule.

On v4 this is the **only** stopping mechanism. Unlike the v3 config, `configs/train_v4_small_lora.yaml` sets no `early_stop_*` keys, and the code default is `early_stop_enabled: false` — there is no automatic backstop to fall back on, so a run left alone will go all the way to `max_steps`. (The v3 config did enable one, with `early_stop_patience: 10` and `early_stop_min_delta: 0.001`, but it was far more permissive than the policy above, so even on v3 the instruction was to stop by hand.) Note also that v4 validates every 1000 steps rather than v3's 200, so each validation window is five times longer in wall-clock terms; account for that before declaring "two windows without improvement". The validation split size is clamped to `[50, 100]` samples regardless of how small `valid_ratio` is (`VALID_MIN_COUNT` / `VALID_MAX_COUNT` in `irodori_tts/training/sampler.py`), so `val_loss` stays meaningful even at the config's `0.0005`.

### Step 5 — post-train

When training finishes (or the user stops it), list the checkpoints under `outputs/<speaker>_lora/`. For LoRA runs, checkpoints are **directories, not `.pt` files**:

- `checkpoint_<step:07d>` — periodic (e.g. `checkpoint_0002000`)
- `checkpoint_best_val_loss_<step:07d>_<val_loss:.6f>` — best-N by validation loss (e.g. `checkpoint_best_val_loss_0002400_0.312100`)
- `checkpoint_final` — final step

With the stock v4 config there is **no** `outputs/<speaker>_lora/samples/` directory, because `sample_generation` is not enabled (see "What is different when training on v4"). Checkpoints have to be A/B'd by ear after the fact with `infer_with_adapter.py` below, or by adding a `sample_generation:` block to a copied config before the run starts. Only v3 runs get samples for free.

Recommended next actions:

- Export the chosen checkpoint to a single self-describing `.safetensors` and drop it in `models/LoRA/`:

  ```
  uv run python scripts/lora/export_lora_to_safetensors.py \
    --input outputs/<speaker>_lora/checkpoint_best_val_loss_0002400_0.312100 \
    --output models/LoRA/<speaker>.safetensors \
    --defaults '{"num_steps": 40, "cfg_scale_text": 3.0, "cfg_scale_speaker": 5.0}'
  ```

  The server auto-discovers every `*.safetensors` under `lora_dir` (`models/LoRA`) at startup — name, UUID and inference defaults ride along in the file's own metadata, so **no entry in `configs/runtime.yaml` is needed**. The display name comes from `data/<speaker>/config.yaml`'s `speaker.label`; `--name` is only a fallback when that is absent, and `--uuid` is derived deterministically from the output filename when omitted. Restart the server to pick the file up, then confirm via `GET /speakers`.
- A manual `speakers:` list in `configs/runtime.yaml` (`uuid` / `name` / `adapter` / `defaults` / `category_id` / `category_label`) is still honored and is appended after auto-discovery, but it is the legacy path. Use it only if the user asks for it explicitly.
- Sanity-check quality before exporting with `PYTHONPATH=. uv run python scripts/lora/infer_with_adapter.py --base <base>.safetensors --adapter outputs/<speaker>_lora/<checkpoint_dir> --text "..." --no-ref --output sample.wav` (this script needs `PYTHONPATH=.`). `--base` must be the same generation the adapter was trained on. Be aware this script has **no `--caption` flag**, so it cannot exercise v4's caption branch; use it for text and reference checks only.
- Merged safetensors are not needed; the server loads base + LoRA directly.

## Out of scope

- Data cleaning and transcription — handled by `tts-preprocess`
- Server deployment — handled manually by dropping the export into `models/LoRA/` and restarting
- Distillation, quantization, or other model surgery
