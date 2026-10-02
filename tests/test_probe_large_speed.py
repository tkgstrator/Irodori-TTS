"""Exercise the comparison runner without SSH, Docker, or a GPU."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_suite(tmp_path: Path, helper: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    scripts = tmp_path / "scripts" / "train"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/train/probe_large_speed.sh", scripts / "probe_large_speed.sh")
    (scripts / "probe_large_batch.sh").write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$PROBE_TEST_CALLS"\n' + helper,
        encoding="utf-8",
    )
    calls = tmp_path / "calls.txt"
    result = subprocess.run(
        ["bash", str(scripts / "probe_large_speed.sh"), "3", "test-node"],
        env={**os.environ, "PROBE_TEST_CALLS": str(calls)},
        capture_output=True,
        text=True,
        check=False,
    )
    return result, calls.read_text(encoding="utf-8").splitlines()


def test_suite_runs_sequential_single_change_comparisons(tmp_path: Path) -> None:
    result, calls = _run_suite(tmp_path, "exit 0\n")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 6
    assert all("3 test-node --steps 30" in call for call in calls)
    assert "--persistent-workers false --workers 16 --checkpoint true" in calls[0]
    assert "--persistent-workers true --workers 16 --checkpoint true" in calls[1]
    assert "--workers 8 --checkpoint true" in calls[2]
    assert "--workers 4 --checkpoint true" in calls[3]
    assert "--workers 8 --checkpoint false" in calls[4]
    assert calls[5].startswith("40 3 test-node")
    assert "--accumulation 2" in calls[5]
    summaries = list((tmp_path / "outputs/batch-probes").glob("*/results.tsv"))
    assert len(summaries) == 1
    assert len(summaries[0].read_text(encoding="utf-8").splitlines()) == 7


def test_non_oom_failure_stops_comparisons(tmp_path: Path) -> None:
    result, calls = _run_suite(tmp_path, "echo 'unauthorized' >&2\nexit 7\n")
    assert result.returncode == 7
    assert len(calls) == 1
    assert "Non-OOM failure" in result.stderr


def test_oom_is_recorded_and_next_comparison_runs(tmp_path: Path) -> None:
    result, calls = _run_suite(tmp_path, "echo 'torch.OutOfMemoryError' >&2\nexit 1\n")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 6
    summary = next((tmp_path / "outputs/batch-probes").glob("*/results.tsv"))
    assert all("\t1\t" in row for row in summary.read_text(encoding="utf-8").splitlines()[1:])
