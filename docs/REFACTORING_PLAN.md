# Refactoring Plan

Status: proposed (2026-09-02). Behavior-preserving refactoring of the largest modules. Each step is one PR and must leave CI green.

## Background and constraints

This repository is a fork of Aratako/Irodori-TTS. `pyproject.toml` declares a fork policy: files inherited from upstream keep baseline lint rules only and are excluded from `ruff format`, so that upstream merges stay cheap. A file may only be restructured once it has genuinely diverged in this fork.

Classification by `git diff upstream/main...HEAD`:

- Fork-owned (free to split): `server.py` (entire file), `irodori_tts/vds/`, `irodori_tts/wandb_client.py`, `irodori_tts/training_samples.py`, `tests/`, most of `scripts/`.
- Substantially diverged (worth splitting): `train.py` (plus 678 lines), `irodori_tts/inference_runtime.py` (plus 184), `irodori_tts/config.py` (plus 102), `irodori_tts/lora.py` (plus 78).
- Near-identical to upstream (do not touch): `gradio_app.py`, `gradio_app_voicedesign.py`, `infer.py`, `convert_checkpoint_to_safetensors.py`, `quantize_checkpoint.py`, `irodori_tts/dataset.py`, `irodori_tts/model.py` (plus 12 lines only), `irodori_tts/codec.py`, `irodori_tts/rf.py`.

The two gradio apps share about 80 percent of their code, and `convert_checkpoint_to_safetensors.py` carries private copies of several `train.py` helpers. This duplication is intentional: it keeps upstream merges cheap. Leave it in place.

CI runs exactly three checks (`.github/workflows/ci.yaml`): `ruff check`, `ruff format --check`, `pytest`. mypy is not in CI. Existing tests cover only `vds/parser` and `wandb_client`; train, model, inference_runtime, server, and dataset have no tests.

Entry points that must not change:

- `train.py` CLI flags (consumed by `docker/train/entrypoint.sh` and `configs/*.yaml`).
- `python server.py` startup (`docker/runtime/entrypoint.sh`) and the API routes `/health`, `/speakers`, `/synth`, `/synth/vds` including PCM streaming (contracts documented in `docs/SERVER.md` and `docs/LLM_API_REFERENCE.md`).
- Checkpoint formats: `.pt` payload keys, safetensors LoRA adapter metadata (the embedded wandb run uuid is read on resume), and the legacy-checkpoint upgrade path (`_check_model_config_compatibility`, `_upgrade_speaker_in_proj`).

## Target layout

```
irodori_tts/
  training/                  # extracted from train.py, strict ruff compliant
    __init__.py
    losses.py                # echo_style_masked_mse, utterance_mean_masked_mse, compute_rf_loss
    checkpointing.py         # save_checkpoint, list_periodic_checkpoints,
                             # enforce_periodic_checkpoint_limit,
                             # list/prune/maybe_save_best_val_loss_* helpers,
                             # periodic/best/final checkpoint path helpers,
                             # dataloader state collection and rank selection,
                             # checkpoint payload loading
    speaker_prompts.py       # speaker yaml loading, speaker name and id resolution,
                             # base model name resolution, prompt building from
                             # speaker config and manifest autopick,
                             # LoRA safetensors metadata build and injection
    model_init.py            # tokenizer builders, backbone dim validation,
                             # initialize_*_from_pretrained, partial state loading,
                             # config compatibility checks, checkpoint format probes,
                             # speaker in_proj upgrade, parameter classification,
                             # grad clearing and freezing helpers, resume LoRA config
    distributed.py           # resolve_dist_env, setup_distributed, reduce_mean, reduce_sum,
                             # batch device moves, cuda prefetch
    sampler.py               # LengthGroupedSampler, split_train_valid_indices
    duration_metrics.py      # duration_condition_group totals, metrics, log suffix, wandb metrics
    validation.py            # run_validation
    cli_args.py              # argparse construction (about 390 lines), cli_provided,
                             # per-field LoRA CLI explicitness
  server/                    # extracted from server.py, fork-owned so unconditionally safe
    __init__.py
    schemas.py               # Vds request models, SynthRequest, defaults merging
    config.py                # SpeakerSpec, ServerConfig, load_config, LoRA dir discovery,
                             # checkpoint resolution, display name resolution
    registry.py              # RuntimeRegistry
    audio.py                 # fade application
    synthesis.py             # synthesis path lifted out of the build_app closure:
                             # single and cue synthesis, drama rendering, PCM streaming
                             # and framing, request validation, route handler bodies
train.py                     # thin orchestrator, target about 1500 lines (main assembly plus training loop)
server.py                    # thin shim (build_app plus main, uvicorn startup) kept for docker compatibility
```

`model.py`, `dataset.py`, both gradio apps, `infer.py`, `convert_checkpoint_to_safetensors.py`, and `quantize_checkpoint.py` stay where they are.

## Steps

Each step is one PR. Run the full verification list (below) before merging.

### Step 0: characterization tests (first, near-zero risk)

Add tests for the pure functions that are about to move. CPU-only, lightweight.

- `tests/test_train_checkpoint_pruning.py`: filename conventions and retention counts for `list_periodic_checkpoints`, `enforce_periodic_checkpoint_limit`, `prune_best_val_loss_checkpoints`, `maybe_save_best_val_loss_checkpoint`, exercised on `tmp_path`.
- `tests/test_train_metrics.py`: `duration_condition_group_totals` and `duration_condition_group_metrics`, `split_train_valid_indices`, `LengthGroupedSampler` determinism with a fixed seed.
- `tests/test_server_config.py`: `load_config`, `_discover_lora_dir`, `_merge_defaults`, `_apply_fade`; FastAPI `TestClient` with `build_app(cfg, eager_load=False)` covering `/health` and schema validation errors.
- `tests/test_inference_runtime_utils.py`: `resolve_cfg_scales`, `find_flattening_point`, `_coerce_latent_shape`, `RuntimeKey`.

Roughly 600 to 900 lines of tests. These test files are independent and can be written in parallel.

### Step 1: split server.py (fork-owned, safest)

Move the symbols listed above into `irodori_tts/server/`. Break the large `build_app` closure (roughly lines 594 to 1207) into route handler functions, then move those and the rest of the synthesis path into `synthesis.py`. `server.py` remains as a shim so `python server.py` and its CLI keep working. Route definitions are unchanged. New files must satisfy strict ruff since they are outside the ignore list.

About 1100 lines moved plus lint compliance work. Done: `server.py` is 110 lines, holding `build_app`, the four route definitions and `main`.

### Step 2a: extract checkpointing and speaker prompts from train.py

Create `training/checkpointing.py` (about 550 lines) and `training/speaker_prompts.py` (about 190 lines); replace with imports in `train.py`.

Caution: do not change a single key in the `save_checkpoint` payload (`model_config`, `train_config`, `base_init`, `text_encoder_config`, dataloader state) or in the LoRA safetensors adapter metadata (the wandb run uuid there is read on resume, around `train.py` line 3302).

### Step 2b: extract model initialization and checkpoint compatibility

Create `training/model_init.py` (about 650 lines). This is the backward-compatibility cluster (`_check_model_config_compatibility`, `_upgrade_speaker_in_proj`, and friends); move code verbatim, no cleanups.

The private copies in `convert_checkpoint_to_safetensors.py` stay as they are (that file is upstream-pristine). If it ever diverges, consolidate by importing from `training/model_init.py` then.

### Step 2c: extract distributed, sampler, losses, validation

Create `training/distributed.py`, `training/sampler.py`, `training/duration_metrics.py`, `training/losses.py`, `training/validation.py` (about 750 lines total). The Step 0 tests carry over to the new modules unchanged.

### Step 3: dismantle train.py main()

Move argparse construction (about 390 lines) to `training/cli_args.py`. Capture `python train.py --help` output before and after and diff it; it must be identical. Split the remaining `main()` into functions (config resolution, wandb setup, dataset and loader construction, model with LoRA and resume, training loop) within the same file.

This brings `train.py` from 4818 to roughly 1500 lines. Once done, remove `train.py` from the `ruff format` exclude and per-file-ignores in `pyproject.toml` and apply `ruff format`, declaring it diverged per the fork policy. The format diff is large, so make this its own PR.

### Step 4 (optional, decide later): inference_runtime.py

The file has diverged enough to qualify, but shares a lot with upstream, so splitting it raises merge cost. If done: stage the `synthesize()` function (roughly lines 1214 to 1642, about 430 lines) into phases, and move the checkpoint loading group (`_load_checkpoint_from_pt`, `_load_checkpoint_from_safetensors`, `download_hf_checkpoint`) into an `inference_io.py`. Public names used by the gradio apps, `infer.py`, and `server.py` (`RuntimeKey`, `SamplingRequest`, `get_cached_runtime`, `save_wav`, `resolve_cfg_scales`, and the rest) must be re-exported from `inference_runtime` unchanged.

### Explicitly out of scope

Splitting `model.py`, deduplicating the gradio apps, and touching `infer.py`, `convert_checkpoint_to_safetensors.py`, `quantize_checkpoint.py`, `prepare_manifest.py`, or `dataset.py`. These are upstream-pristine or barely diverged; merge cost exceeds duplication cost, and none of them have tests.

## Parallelization

| Work | Owned files | Parallelism |
|---|---|---|
| Step 0 test files | new files under `tests/` | parallel with each other |
| Step 1 (server) | `server.py`, `irodori_tts/server/` | parallel with Step 2 |
| Steps 2a, 2b, 2c, 3 | `train.py`, `irodori_tts/training/` | single worker, sequential |
| Step 4 | `irodori_tts/inference_runtime.py` | parallel with Steps 1 to 3; if it touches server.py imports, after Step 1 |

Contention: `pyproject.toml` (ignore list edits) is touched by several steps; keep those edits as a small trailing change in each PR and resolve by merge order. The public API of `irodori_tts/__init__.py` (`ModelConfig`, `TrainConfig`, `TextToLatentRFDiT`, and the rest) is unchanged.

## Risks

1. Test blind spots: the training loop, LoRA resume, and DDP paths cannot be tested automatically (GPU and multi-process). Mitigate with verbatim moves, the `--help` diff, and a short CPU smoke run if a dry-run mode is feasible.
2. Checkpoint compatibility: payload keys, safetensors metadata, and the legacy upgrade path directly affect loading existing artifacts. Do not change key names or the order of compatibility checks.
3. Public entry points: train CLI, server startup and API routes, gradio UI.
4. Strict ruff on moved code: new files are outside the ignore list, so verbatim moves will surface violations (PTH, TRY, SLF, and so on). Budget lint compliance at 10 to 20 percent of moved lines; keep behavior-changing rewrites to a minimum.
5. Upstream merges: once a file is split, future upstream merges for it become manual. This is why the plan only touches fork-owned or already-diverged files.

## Verification after each step

Same as CI:

```
uvx ruff@0.16.5 check .
uvx ruff@0.16.5 format --check .
uv run pytest -q
```

Additionally: after Steps 1 and 4, `python server.py --help` and a `TestClient` smoke of `/health` and `/speakers`; after Steps 2 and 3, diff `python train.py --help` against the pre-change output; import smoke via `uv run python -c "import server, train"`.
