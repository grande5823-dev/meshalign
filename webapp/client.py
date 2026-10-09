"""Thin Python client for the meshalign console API.

One function per endpoint; includes the job-profile helpers
``start_run``, ``get_run(wait=)``, ``stream_events``, ``cancel_run``
(style guide §15.4). Standard library only.

    from client import MeshalignClient
    c = MeshalignClient("http://localhost:8766")
    run = c.start_run(ref="mech_parts/Connector.stl", test="mech_parts/Connector_test.stl")
    final = c.get_run(run["run_id"], wait=120)          # long-poll to terminal
    print(final["metrics"]["mae_um"])
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator


class MeshalignClient:
    def __init__(self, base_url: str = "http://localhost:8766"):
        self.base = base_url.rstrip("/")

    # -- plumbing -----------------------------------------------------------
    def _request(self, path: str, method: str = "GET", body: dict | None = None,
                 timeout: float = 60) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json",
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read().decode())["error"]
                raise RuntimeError(f"{e.code} {err['code']}: {err['message']}") from None
            except (json.JSONDecodeError, KeyError):
                raise RuntimeError(f"{e.code} {e.reason}") from None

    def _stream(self, path: str, timeout: float | None = None) -> Iterator[dict]:
        """Yield parsed SSE events: {id, event, data}."""
        req = urllib.request.Request(self.base + path,
                                     headers={"Accept": "text/event-stream"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            eid, etype = None, None
            for raw in resp:
                line = raw.decode().rstrip("\n")
                if line.startswith("id: "):
                    eid = int(line[4:])
                elif line.startswith("event: "):
                    etype = line[7:]
                elif line.startswith("data: ") and etype:
                    yield {"id": eid, "event": etype,
                           "data": json.loads(line[6:])}
                    eid, etype = None, None
                elif line == "" and etype:  # some events carry no data
                    yield {"id": eid, "event": etype, "data": {}}
                    eid, etype = None, None

    # -- meta ---------------------------------------------------------------
    def health(self) -> dict:
        """Server liveness + the active run id, if any."""
        return self._request("/api/health")

    def schema(self) -> dict:
        """Parameter schema for start_run (types, defaults, ranges)."""
        return self._request("/api/schema")

    # -- datasets -----------------------------------------------------------
    def datasets(self) -> list[dict]:
        """Available reference/test geometry files."""
        return self._request("/api/datasets")

    def upload(self, path: str, folder: str | None = None) -> dict:
        """Upload a local mesh/point-cloud file (optionally into an upload
        sub-folder, used to pair files for ``batch``); returns its dataset row."""
        import os
        import uuid
        boundary = uuid.uuid4().hex
        name = urllib.parse.quote(os.path.basename(path))
        with open(path, "rb") as f:
            payload = f.read()
        parts = b""
        if folder:
            parts = (f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="folder"\r\n\r\n{folder}\r\n').encode()
        body = parts + (f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="file"; filename="{name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + payload + \
               f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            self.base + "/api/datasets/upload", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode())

    # -- job profile (§4) ----------------------------------------------------
    def start_run(self, ref: str, test: str, mode: str = "default",
                  use_icp: bool = True, metrics: bool = True,
                  area_weighted: bool = True, seed: int = 0,
                  client_token: str | None = None) -> dict:
        """Start an alignment run; returns 202 {run_id}. Raises RuntimeError
        (409 run_active) if a run is already active."""
        body = {"params": {"ref": ref, "test": test, "mode": mode,
                           "use_icp": use_icp, "metrics": metrics,
                           "area_weighted": area_weighted, "seed": seed}}
        if client_token:
            body["client_token"] = client_token
        return self._request("/api/runs", method="POST", body=body)

    def commands(self) -> dict:
        """The command registry (run, measure, register, chamfer, ensemble,
        batch) with every parameter, plus the dataset folders usable by batch."""
        return self._request("/api/commands")

    def start(self, command: str, client_token: str | None = None, **params) -> dict:
        """Start any command, e.g. ``start("register", ref=..., test=..., path="global")``,
        ``start("chamfer", a=..., b=...)``, ``start("ensemble", meshes=[...])``,
        ``start("batch", refs="uploads/refs", tests="uploads/tests")``."""
        body = {"command": command, "params": params}
        if client_token:
            body["client_token"] = client_token
        return self._request("/api/runs", method="POST", body=body)

    def get_run(self, run_id: str, wait: float = 0) -> dict:
        """Run status/metrics/params/meta; ``wait`` long-polls for a terminal
        state (max 60 s per call — loop for longer waits)."""
        return self._request(f"/api/runs/{run_id}?wait={min(wait, 60)}")

    def wait_terminal(self, run_id: str, timeout_s: float = 3600) -> dict:
        """Block until the run reaches a terminal state, then return it."""
        deadline = time.monotonic() + timeout_s
        while True:
            rec = self.get_run(run_id, wait=60)
            if rec.get("status") in ("succeeded", "failed", "cancelled"):
                return rec
            if time.monotonic() > deadline:
                raise TimeoutError(f"run {run_id} still not terminal")

    def list_runs(self, limit: int = 50) -> list[dict]:
        """Past runs, newest first."""
        return self._request(f"/api/runs?limit={limit}")

    def stream_events(self, run_id: str, last_event_id: int = 0,
                      timeout: float | None = None) -> Iterator[dict]:
        """Yield SSE events (status/progress/log/metric/done) for a run."""
        return self._stream(
            f"/api/runs/{run_id}/events?lastEventId={last_event_id}",
            timeout=timeout)

    def cancel_run(self, run_id: str) -> dict:
        """Cancel the active run; terminal runs report their status."""
        return self._request(f"/api/runs/{run_id}/cancel", method="POST")

    # -- results -------------------------------------------------------------
    def artifacts(self, run_id: str) -> list[dict]:
        """Artifact rows: {name, type, size_bytes, url}."""
        return self._request(f"/api/runs/{run_id}/artifacts")

    def artifact_url(self, run_id: str, name: str) -> str:
        """Direct URL for one artifact (aligned mesh, CSVs, histogram…)."""
        return self.base + f"/api/runs/{run_id}/artifacts/{name}"

    def view(self, run_id: str) -> dict:
        """Viewer payload: summary + mesh/error/histogram URLs + stats."""
        return self._request(f"/api/runs/{run_id}/view")

    def export_url(self, run_id: str) -> str:
        """URL that downloads the whole run directory as a zip."""
        return self.base + f"/api/runs/{run_id}/export"


if __name__ == "__main__":
    import sys
    c = MeshalignClient(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8766")
    print("health:", c.health())
    for row in c.datasets()[:5]:
        print(f'  dataset: {row["id"]:40s} {row["kind"]:6s} {row["size_bytes"]}')
    print("run with: c.start_run(ref=..., test=...)")
