"""API tests for the meshalign console webapp.

Runs real (small) alignments on the bundled demo Connector pair, so each run
takes ~5 s. Uses a temp runs dir; nothing touches webapp/runs.

Run from webapp/:
    pytest tests/test_api.py
"""
from __future__ import annotations

import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEMO_REF = "mech_parts/Connector.stl"
DEMO_TEST = "mech_parts/Connector_test.stl"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MESHALIGN_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MESHALIGN_UPLOADS_DIR", str(tmp_path / "uploads"))
    import app as appmod
    import importlib
    importlib.reload(appmod)
    # fresh manager against the temp dir
    appmod.manager = appmod.runner_mod.RunManager(str(tmp_path / "runs"))
    with TestClient(appmod.app) as c:
        yield c


def _wait_terminal(client, run_id, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = client.get(f"/api/runs/{run_id}").json()
        if rec.get("status") in ("succeeded", "failed", "cancelled"):
            return rec
        time.sleep(0.5)
    raise TimeoutError(f"run {run_id} not terminal")


def test_health_and_schema(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "active_run" in r.json()

    r = client.get("/api/schema")
    props = r.json()["properties"]
    assert {"ref", "test", "mode", "metrics", "seed", "relative", "histogram",
            "per_sample", "per_vertex_mae"} <= set(props)
    assert props["mode"]["enum"] == ["fast", "default", "accuracy"]
    assert props["seed"]["default"] == 0
    assert props["ref"]["type"] == "string"


def test_datasets_listed(client):
    r = client.get("/api/datasets")
    ids = [d["id"] for d in r.json()]
    assert DEMO_REF in ids and DEMO_TEST in ids
    by_id = {d["id"]: d for d in r.json()}
    assert by_id[DEMO_REF]["kind"] == "mesh"


def test_start_validation(client):
    # unknown dataset -> 422 with the field named
    r = client.post("/api/runs", json={"params": {
        "ref": "demo/does-not-exist.stl", "test": DEMO_TEST}})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "unknown_dataset"
    assert err["details"]["field"] == "ref"

    # bad enum / negative seed -> 422 per-field validation errors
    r = client.post("/api/runs", json={"params": {
        "ref": DEMO_REF, "test": DEMO_TEST, "mode": "turbo"}})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"

    r = client.post("/api/runs", json={"params": {
        "ref": DEMO_REF, "test": DEMO_TEST, "seed": -1}})
    assert r.status_code == 422
    fields = r.json()["error"]["details"]["fields"]
    assert any(f["field"].endswith("seed") for f in fields)

    # path traversal in a dataset id is refused
    r = client.post("/api/runs", json={"params": {
        "ref": "../webapp/app.py", "test": DEMO_TEST}})
    assert r.status_code == 422


def test_run_happy_path_and_artifacts(client):
    r = client.post("/api/runs", json={"params": {
        "ref": DEMO_REF, "test": DEMO_TEST, "mode": "fast",
        "metrics": True, "seed": 0}})
    assert r.status_code == 202
    rid = r.json()["run_id"]

    rec = _wait_terminal(client, rid)
    assert rec["status"] == "succeeded", rec.get("message")
    m = rec["metrics"]
    assert 0 <= m["mae_um"] < 50           # noise-free pair: ~0 um
    assert m["iou"] is not None and m["iou"] > 0.9
    assert rec["params"]["seed"] == 0
    assert rec["params_hash"]
    assert rec["duration_s"] > 0
    assert rec["git_commit"]

    # artifacts include canonical outputs + viewer files
    arts = {a["name"] for a in client.get(f"/api/runs/{rid}/artifacts").json()}
    assert {"params.json", "meta.json", "metrics.json", "log.txt",
            "pair_aligned.stl", "pair_surf_mae_um.csv",
            "viewer/ref.ply", "viewer/aligned.ply",
            "viewer/errors_ref.f32", "viewer/errors_test.f32",
            "viewer/hist.json", "viewer/summary.json"} <= arts

    # viewer payload: geometry + error stats + histogram
    v = client.get(f"/api/runs/{rid}/view").json()
    n_ref = v["view"]["ref"]["n_points"]
    assert n_ref > 0
    assert v["view"]["errors_test"]["count"] == n_ref
    assert v["view"]["hist"]["url"]
    assert v["transform_4x4"][3] == [0.0, 0.0, 0.0, 1.0]
    assert v["headline"]["mae_um"] == pytest.approx(m["mae_um"], abs=0.01)

    # a mesh artifact downloads with the right type
    r = client.get(f"/api/runs/{rid}/artifacts/viewer/aligned.ply")
    assert r.status_code == 200
    assert r.headers["content-type"] == "model/ply"
    assert r.content[:3] == b"ply"

    # export zip
    r = client.get(f"/api/runs/{rid}/export")
    assert r.status_code == 200
    assert r.content[:2] == b"PK"

    # runs list carries key metrics
    runs = client.get("/api/runs").json()
    assert runs[0]["run_id"] == rid
    assert runs[0]["key_metrics"]["mae_um"] == pytest.approx(m["mae_um"], abs=0.01)


def test_concurrent_start_409_and_idempotent_token(client):
    r = client.post("/api/runs", json={"params": {
        "ref": DEMO_REF, "test": DEMO_TEST, "mode": "default"},
        "client_token": "tok-1"})
    assert r.status_code == 202
    rid = r.json()["run_id"]

    # a second run while active -> 409 with the active id
    r = client.post("/api/runs", json={"params": {
        "ref": DEMO_REF, "test": DEMO_TEST}})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "run_active"
    assert r.json()["error"]["details"]["active_run_id"] == rid

    # same token returns the SAME run (idempotent), not a 409
    r = client.post("/api/runs", json={"params": {
        "ref": DEMO_REF, "test": DEMO_TEST}, "client_token": "tok-1"})
    assert r.status_code == 202
    assert r.json()["run_id"] == rid

    rec = _wait_terminal(client, rid)
    assert rec["status"] in ("succeeded", "cancelled")


def test_cancel(client):
    r = client.post("/api/runs", json={"params": {
        "ref": DEMO_REF, "test": DEMO_TEST, "mode": "default"}})
    rid = r.json()["run_id"]
    r = client.post(f"/api/runs/{rid}/cancel")
    assert r.status_code == 200
    rec = _wait_terminal(client, rid, timeout=60)
    # cancel lands unless the run finished first (it's a ~5 s job)
    assert rec["status"] in ("cancelled", "succeeded")
    if rec["status"] == "cancelled":
        assert rec["message"] == "cancelled by user"
    # cancelling a terminal run reports the unchanged status
    r = client.post(f"/api/runs/{rid}/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == rec["status"]


def test_unknown_run_404(client):
    r = client.get("/api/runs/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "run_not_found"


def test_upload_and_use(client):
    # upload the demo ref as if it were a user file
    with open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "..", "data", "mech_parts",
            "Connector.stl"), "rb") as f:
        r = client.post("/api/datasets/upload", files={
            "file": ("my_scan.stl", f, "model/stl")})
    assert r.status_code == 200
    row = r.json()
    assert row["id"].startswith("uploads/")
    assert row["kind"] == "mesh"

    # the uploaded dataset is listable and runnable
    ids = [d["id"] for d in client.get("/api/datasets").json()]
    assert row["id"] in ids
    r = client.post("/api/runs", json={"params": {
        "ref": DEMO_REF, "test": row["id"], "mode": "fast"}})
    assert r.status_code == 202
    rec = _wait_terminal(client, r.json()["run_id"])
    assert rec["status"] == "succeeded", rec.get("message")

    # bad extension refused
    r = client.post("/api/datasets/upload", files={
        "file": ("evil.exe", b"MZ...", "application/octet-stream")})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_upload"


def test_sse_stream_and_wait_longpoll(client):
    import json as _json
    r = client.post("/api/runs", json={"params": {
        "ref": DEMO_REF, "test": DEMO_TEST, "mode": "fast"}})
    rid = r.json()["run_id"]
    # long-poll returns promptly when the run finishes
    rec = client.get(f"/api/runs/{rid}?wait=60").json()
    assert rec["status"] in ("succeeded", "failed", "cancelled")
    # SSE replay for a terminal run: log events then a done event
    with client.stream("GET", f"/api/runs/{rid}/events") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in resp.iter_lines():
            line = line.decode() if isinstance(line, bytes) else line
            if line.startswith("event: "):
                events.append(line[7:])
            if events and events[-1] == "done":
                break
    assert "log" in events and events[-1] == "done"


def test_restart_safety(tmp_path):
    # a run dir left in "running" is marked failed by a fresh manager
    runs = tmp_path / "runs" / "20260101-000000-dead00"
    runs.mkdir(parents=True)
    (runs / "meta.json").write_text(
        '{"run_id": "20260101-000000-dead00", "status": "running"}')
    import runner
    mgr = runner.RunManager(str(tmp_path / "runs"))
    rec = mgr.get_run("20260101-000000-dead00")
    assert rec["status"] == "failed"
    assert rec["message"] == "server restarted during run"


def test_spa_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"meshalign console" in r.content
    r = client.get("/whatever/page")
    assert r.status_code == 200
    assert b"meshalign console" in r.content
    r = client.get("/api/nope")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# every CLI command is reachable through the API
# ---------------------------------------------------------------------------

def _run(client, command, params, timeout=240):
    r = client.post("/api/runs", json={"command": command, "params": params})
    assert r.status_code == 202, r.text
    rec = _wait_terminal(client, r.json()["run_id"], timeout=timeout)
    assert rec["status"] == "succeeded", rec
    return rec


def test_commands_registry_matches_cli(client):
    from meshalign.cli import build_parser
    body = client.get("/api/commands").json()
    cmds = body["commands"]
    assert set(cmds) == {"run", "measure", "register", "chamfer", "ensemble", "batch"}
    sub = [a for a in build_parser()._actions if a.dest == "cmd"][0].choices
    assert set(sub) == set(cmds)
    # every CLI option (except -h and server-controlled output paths) has a form field
    for name, parser in sub.items():
        cli_dests = {a.dest for a in parser._actions if a.dest not in ("help",)}
        form = {p["name"] for p in cmds[name]["params"]}
        assert cli_dests - form <= {"out"}, (name, cli_dests - form)
    assert any(g["id"] == "mech_parts" for g in body["groups"])


def test_register_command(client):
    rec = _run(client, "register", {"ref": DEMO_REF, "test": DEMO_TEST, "mode": "fast",
                                    "path": "local"})
    rid = rec["run_id"]
    names = {a["name"] for a in client.get(f"/api/runs/{rid}/artifacts").json()}
    assert {"aligned.stl", "viewer/aligned.ply", "viewer/ref.ply", "result.json"} <= names
    assert len(rec["result"]["transform_4x4"]) == 4
    assert "rotation_deg" in rec["metrics"]


def test_measure_and_run_flags(client):
    rec = _run(client, "measure", {"ref": DEMO_REF, "test": DEMO_TEST, "mode": "fast",
                                   "metrics": False, "histogram": False,
                                   "per_sample": False, "relative": True,
                                   "per_vertex_mae": True})
    names = {a["name"] for a in client.get(f"/api/runs/{rec['run_id']}/artifacts").json()}
    assert "pair_dft_summary.csv" in names
    assert "pair_surf_histogram.png" not in names
    assert not any(n.startswith("pair_surf_errors_mm.") for n in names)
    assert "pair_metrics.json" not in names
    assert rec["params"]["command"] == "measure"
    # legacy use_icp=False on `run` maps to measure
    r = client.post("/api/runs", json={"params": {
        "ref": DEMO_REF, "test": DEMO_TEST, "mode": "fast", "use_icp": False,
        "metrics": False}})
    assert r.status_code == 202
    assert _wait_terminal(client, r.json()["run_id"])["params"]["command"] == "measure"


def test_chamfer_command(client):
    rec = _run(client, "chamfer", {"a": DEMO_REF, "b": DEMO_TEST})
    assert rec["result"]["chamfer"] > 0
    assert rec["metrics"]["chamfer_mm"] == rec["result"]["chamfer"]


def test_ensemble_command(client):
    rec = _run(client, "ensemble", {"meshes": [DEMO_REF, DEMO_TEST]})
    assert len(rec["result"]["rows"]) == 2
    names = {a["name"] for a in client.get(f"/api/runs/{rec['run_id']}/artifacts").json()}
    assert "ensemble.csv" in names
    r = client.post("/api/runs", json={"command": "ensemble",
                                       "params": {"meshes": [DEMO_REF]}})
    assert r.status_code == 422


def _upload(client, fname, src, folder):
    with open(src, "rb") as f:
        r = client.post("/api/datasets/upload", files={"file": (fname, f, "model/stl")},
                        data={"folder": folder})
    assert r.status_code == 200, r.text
    return r.json()


def test_batch_command_with_upload_folders(client):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "..", "data", "mech_parts")
    a = _upload(client, "part.stl", os.path.join(root, "Connector.stl"), "refs")
    b = _upload(client, "part.stl", os.path.join(root, "Connector_test.stl"), "tests")
    assert a["id"] == "uploads/refs/part.stl" and a["group"] == "uploads/refs"
    groups = {g["id"] for g in client.get("/api/commands").json()["groups"]}
    assert {"uploads/refs", "uploads/tests"} <= groups
    rec = _run(client, "batch", {"refs": "uploads/refs", "tests": "uploads/tests",
                                 "mode": "fast", "jobs": 1})
    assert rec["result"]["n_ok"] == 1 and rec["metrics"]["n_pairs"] == 1
    names = {x["name"] for x in client.get(f"/api/runs/{rec['run_id']}/artifacts").json()}
    assert "batch_summary.csv" in names
    r = client.post("/api/runs", json={"command": "batch", "params": {
        "refs": "uploads/nope", "tests": "uploads/tests"}})
    assert r.status_code == 422 and r.json()["error"]["details"]["field"] == "refs"
    r = client.post("/api/runs", json={"command": "batch", "params": {
        "refs": "../..", "tests": "uploads/tests"}})
    assert r.status_code == 422


def test_unknown_command_and_extra_param_422(client):
    r = client.post("/api/runs", json={"command": "explode", "params": {}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_command"
    r = client.post("/api/runs", json={"command": "chamfer", "params": {
        "a": DEMO_REF, "b": DEMO_TEST, "bogus": 1}})
    assert r.status_code == 422


def test_export_requires_finished_run(client):
    r = client.post("/api/runs", json={"params": {"ref": DEMO_REF, "test": DEMO_TEST,
                                                  "mode": "accuracy"}})
    rid = r.json()["run_id"]
    assert client.get(f"/api/runs/{rid}/export").status_code == 409
    client.post(f"/api/runs/{rid}/cancel")
    _wait_terminal(client, rid)
    assert client.get(f"/api/runs/{rid}/export").status_code == 200


def test_client_token_survives_restart(client, tmp_path):
    import app as appmod
    r = client.post("/api/runs", json={"client_token": "tok-1", "command": "chamfer",
                                       "params": {"a": DEMO_REF, "b": DEMO_TEST}})
    rid = r.json()["run_id"]
    _wait_terminal(client, rid)
    appmod.manager = appmod.runner_mod.RunManager(str(tmp_path / "runs"))
    r2 = client.post("/api/runs", json={"client_token": "tok-1", "command": "chamfer",
                                        "params": {"a": DEMO_REF, "b": DEMO_TEST}})
    assert r2.json()["run_id"] == rid


def test_history_labels_per_command(client):
    _run(client, "chamfer", {"a": DEMO_REF, "b": DEMO_TEST})
    row = client.get("/api/runs").json()[0]
    assert row["key_params"]["command"] == "chamfer"
    assert "Connector" in row["key_params"]["label"]
    assert row["key_metrics"]["chamfer_mm"] > 0
