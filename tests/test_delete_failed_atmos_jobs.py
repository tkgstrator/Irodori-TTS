"""Failed-job cleanup never removes an active or differently scoped run."""

from __future__ import annotations

import io
import json

import httpx
import pytest

from scripts.train import delete_failed_atmos_jobs as cleanup

IDS = [
    "95172f44-0e76-4038-bb76-dc30708c03ad",
    "6c8cac87-d084-4ad9-a171-01502a392234",
]


def _configure(monkeypatch, jobs, *, delete_status=204):
    deleted = []

    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-only-secret"
        if request.method == "GET":
            return httpx.Response(200, json=jobs[request.url.path.rsplit("/", 1)[-1]])
        deleted.append(request.url.path)
        return httpx.Response(delete_status)

    client = httpx.Client(
        base_url=cleanup.API_URL,
        headers={"Authorization": "Bearer test-only-secret"},
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setenv("ATMOS_TOKEN", "test-only-secret")
    monkeypatch.setattr(cleanup.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(cleanup.sys, "stdin", io.StringIO(json.dumps(IDS)))
    return deleted


def test_only_currently_failed_large_jobs_are_deleted(monkeypatch, capsys):
    deleted = _configure(
        monkeypatch,
        {
            IDS[0]: {"status": "failed", "name": "gi_aether_lora_v4_large"},
            IDS[1]: {"status": "running", "name": "gi_aino_lora_v4_large"},
        },
    )
    cleanup.main()
    assert len(deleted) == 1
    assert deleted[0].endswith(IDS[0])
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert summary["deleted"] == [IDS[0]]
    assert summary["skipped"][0]["status"] == "running"
    assert "test-only-secret" not in output


def test_deletion_error_stops_further_deletes(monkeypatch, capsys):
    deleted = _configure(
        monkeypatch,
        {value: {"status": "failed", "name": "gi_one_lora_v4_large"} for value in IDS},
        delete_status=403,
    )
    with pytest.raises(SystemExit):
        cleanup.main()
    assert len(deleted) == 1
    assert json.loads(capsys.readouterr().out)["errors"][0]["http_status"] == 403


def test_finished_and_other_model_jobs_are_kept(monkeypatch, capsys):
    deleted = _configure(
        monkeypatch,
        {
            IDS[0]: {"status": "finished", "name": "gi_aether_lora_v4_large"},
            IDS[1]: {"status": "failed", "name": "gi_aether_lora_v4_small"},
        },
    )
    cleanup.main()
    assert deleted == []
    assert len(json.loads(capsys.readouterr().out)["skipped"]) == 2
