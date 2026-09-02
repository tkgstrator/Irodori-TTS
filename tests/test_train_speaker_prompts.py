"""Tests for sample-prompt auto-selection from a manifest.

``_autopick_prompts_from_manifest`` promises determinism, so the fallback path
must not order equal-length texts through a set.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from irodori_tts.training.speaker_prompts import _autopick_prompts_from_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]

# All 5 characters long, so every text ties on the length sort and only the
# tiebreak decides the order. All shorter than min_len, forcing the fallback.
SHORT_TEXTS = ["kappa", "sigma", "delta", "gamma", "theta", "omega", "alpha", "betaa"]

_SELECT_SNIPPET = """
import sys
from irodori_tts.training.speaker_prompts import _autopick_prompts_from_manifest

picks = _autopick_prompts_from_manifest(sys.argv[1])
print("|".join(p.text for p in picks))
"""


def _write_manifest(path: Path, texts: list[str]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return path


def _select_in_subprocess(manifest: Path, hash_seed: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _SELECT_SNIPPET, str(manifest)],
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(REPO_ROOT)},
        text=True,
    )
    return result.stdout.strip()


def test_fallback_selection_is_stable_across_hash_seeds(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl", SHORT_TEXTS)
    first = _select_in_subprocess(manifest, "0")
    second = _select_in_subprocess(manifest, "12345")
    assert first == second
    assert first != ""


def test_fallback_breaks_length_ties_by_text(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl", SHORT_TEXTS)
    picks = [p.text for p in _autopick_prompts_from_manifest(manifest)]
    assert picks == sorted(picks)


def test_fallback_deduplicates(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl", ["aaa", "aaa", "bbb"])
    picks = [p.text for p in _autopick_prompts_from_manifest(manifest)]
    assert picks == ["aaa", "bbb"]


def test_in_range_texts_keep_manifest_order_for_ties(tmp_path: Path) -> None:
    texts = ["zzzzzzzzzzzz", "aaaaaaaaaaaa", "mmmmmmmmmmmm"]
    manifest = _write_manifest(tmp_path / "manifest.jsonl", texts)
    picks = [p.text for p in _autopick_prompts_from_manifest(manifest)]
    assert picks == texts


def test_missing_manifest_returns_nothing(tmp_path: Path) -> None:
    assert _autopick_prompts_from_manifest(tmp_path / "absent.jsonl") == []
    assert _autopick_prompts_from_manifest(None) == []
