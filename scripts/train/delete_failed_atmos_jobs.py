#!/usr/bin/env python3
"""Delete only currently failed jobs from the V4 Large Atmos project.

Run inside a training container, passing a previously inspected list of job IDs
on stdin. Credentials come from the container environment and are never printed.
"""

from __future__ import annotations

import json
import os
import sys
from uuid import UUID

import httpx

PROJECT = "6c8cac87-d084-4ad9-a171-01502a392234"
API_URL = "https://atmos-staging.qleap.jp"


def main() -> None:
    ids = json.load(sys.stdin)
    if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
        raise ValueError("Expected a JSON list of job IDs")
    for value in ids:
        UUID(value)
    token = os.environ["ATMOS_TOKEN"]
    summary: dict[str, list] = {"deleted": [], "skipped": [], "errors": []}
    with httpx.Client(
        base_url=API_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
        follow_redirects=False,
    ) as client:
        for job_id in ids:
            path = f"/api/projects/{PROJECT}/jobs/{job_id}"
            try:
                response = client.get(path)
                response.raise_for_status()
                job = response.json()
                if job["status"] != "failed":
                    summary["skipped"].append({"id": job_id, "status": job["status"]})
                    continue
                if not (job.get("name") or "").endswith("_v4_large"):
                    summary["skipped"].append({"id": job_id, "reason": "different model"})
                    continue
                response = client.delete(path)
                if response.status_code != 204:
                    summary["errors"].append({"id": job_id, "http_status": response.status_code})
                    break
                summary["deleted"].append(job_id)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                summary["errors"].append({"id": job_id, "error": type(exc).__name__})
                break
    print(json.dumps(summary, ensure_ascii=False))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
