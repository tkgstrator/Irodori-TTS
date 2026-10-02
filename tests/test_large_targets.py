"""Prepared Large speaker inventory must match the published adapter IDs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.train import large_targets


def _prepared(root: Path, speaker: str, clips: int = 2) -> Path:
    folder = root / speaker
    (folder / "latents").mkdir(parents=True)
    records = []
    for index in range(clips):
        name = f"{index:08d}.pt"
        (folder / "latents" / name).write_bytes(b"latent")
        records.append({"text": "こんにちは", "latent_path": f"latents/{name}", "num_frames": 20})
    manifest = folder / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


def test_published_targets_are_sorted_and_reject_duplicates() -> None:
    paths = [
        "v3/ayaka.safetensors",
        "v4.1-small/vtuber/vtuber_cherry.safetensors",
        "v4.1-small/genshin_impact/gi_ayaka.safetensors",
        "README.md",
    ]
    assert large_targets.published_targets(paths) == [
        ("gi_ayaka", "genshin_impact"),
        ("vtuber_cherry", "vtuber"),
    ]
    with pytest.raises(ValueError, match="duplicate_ids=1"):
        large_targets.published_targets([*paths, paths[2]])
    with pytest.raises(ValueError, match="malformed=1"):
        large_targets.published_targets(["v4.1-small/vtuber/gi_ayaka.safetensors"])


def test_inventory_prefers_direct_data_and_explicit_legacy_alias(tmp_path: Path) -> None:
    v4 = tmp_path / "v4"
    legacy = tmp_path / "legacy"
    direct = _prepared(v4, "gi_ayaka")
    _prepared(legacy, "ayaka", clips=1)
    alias = _prepared(legacy, "vivi", clips=3)
    inventory = large_targets.build_inventory(
        [
            "v4.1-small/vtuber/vtuber_vivi.safetensors",
            "v4.1-small/genshin_impact/gi_ayaka.safetensors",
        ],
        v4_data_root=v4,
        legacy_data_root=legacy,
        revision="abc123",
    )
    assert inventory["source_revision"] == "abc123"
    assert inventory["entries"] == [
        {"id": "gi_ayaka", "category": "genshin_impact", "manifest_path": str(direct), "clips": 2},
        {"id": "vtuber_vivi", "category": "vtuber", "manifest_path": str(alias), "clips": 3},
    ]


def test_inventory_rejects_missing_data_with_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="targets=1 valid=0 invalid=1"):
        large_targets.build_inventory(
            ["v4.1-small/honkai_star_rail/hsr_bailu.safetensors"],
            v4_data_root=tmp_path / "v4",
            legacy_data_root=tmp_path / "legacy",
            revision="sha",
        )


@pytest.mark.parametrize("broken", ["missing", "duplicate", "traversal", "count"])
def test_manifest_validation_rejects_incomplete_latents(tmp_path: Path, broken: str) -> None:
    manifest = _prepared(tmp_path, "gi_ayaka")
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if broken == "missing":
        (manifest.parent / "latents" / "00000001.pt").unlink()
    elif broken == "duplicate":
        lines[1] = lines[0]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif broken == "traversal":
        record = json.loads(lines[1])
        record["latent_path"] = "latents/../bad.pt"
        lines[1] = json.dumps(record)
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        (manifest.parent / "latents" / "extra.pt").write_bytes(b"latent")
    with pytest.raises(ValueError, match="gi_ayaka"):
        large_targets.build_inventory(
            ["v4.1-small/genshin_impact/gi_ayaka.safetensors"],
            v4_data_root=tmp_path,
            legacy_data_root=tmp_path / "legacy",
            revision="sha",
        )


def test_main_pins_hub_revision_and_never_overwrites_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepared(tmp_path / "v4", "gi_ayaka", clips=1)
    output = tmp_path / "targets.json"

    class FakeApi:
        def __init__(self, token: str | None) -> None:
            assert token == "test-token"

        def repo_info(self, repo_id: str, *, repo_type: str, revision: str) -> SimpleNamespace:
            assert repo_id == large_targets.REPO_ID
            assert repo_type == "model"
            assert revision == "main"
            return SimpleNamespace(sha="pinned-sha")

        def list_repo_files(self, repo_id: str, *, repo_type: str, revision: str) -> list[str]:
            assert repo_id == large_targets.REPO_ID
            assert repo_type == "model"
            assert revision == "pinned-sha"
            return ["v4.1-small/genshin_impact/gi_ayaka.safetensors"]

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "large_targets.py",
            "--v4-data-root",
            str(tmp_path / "v4"),
            "--legacy-data-root",
            str(tmp_path / "legacy"),
            "--output",
            str(output),
        ],
    )
    large_targets.main()
    assert json.loads(output.read_text(encoding="utf-8"))["source_revision"] == "pinned-sha"
    large_targets.main()
    assert "unchanged:" in capsys.readouterr().out
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        large_targets.main()
