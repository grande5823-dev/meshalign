"""meshalign console — FastAPI app (job profile).

Run an alignment on a reference/test pair, watch it stream, and inspect the
result: 3D error map, error histogram, metrics, provenance, artifacts.

Humans use the browser UI at ``/``; agents use the HTTP API under ``/api``
(discoverable via ``/openapi.json``, wrapped by ``client.py``).

Run:  uvicorn app:app --port 8766   (from the webapp/ directory)
Env:  MESHALIGN_RUNS_DIR      where run outputs persist (default webapp/runs)
      MESHALIGN_UPLOADS_DIR   where uploads land (default webapp/uploads)
"""
from __future__ import annotations

import json
import os
import zipfile
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Path, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import datasets as datasets_mod
import runner as runner_mod
from runner import ActiveRunError, RunManager

from meshalign import commands as commands_mod  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = "1.0.0"

app = FastAPI(
    title="meshalign console",
    version=VERSION,
    description="Align a reference/test mesh pair and inspect the result.",
)

manager = RunManager()


# ---------------------------------------------------------------------------
# Error shape
# ---------------------------------------------------------------------------

class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = {}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        self.status, self.code, self.message = status, code, message
        self.details = details or {}


@app.exception_handler(ApiError)
async def _api_error_handler(_req: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status,
                         content={"error": {"code": exc.code,
                                            "message": exc.message,
                                            "details": exc.details}})


@app.exception_handler(HTTPException)
async def _http_error_handler(_req: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                         content={"error": {"code": "http_error",
                                            "message": str(exc.detail),
                                            "details": {}}})


@app.exception_handler(RequestValidationError)
async def _validation_handler(_req: Request, exc: RequestValidationError):
    fields = []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", []) if str(x) != "body"]
        fields.append({"field": ".".join(loc) or "body",
                       "message": err.get("msg", "invalid value"),
                       "type": err.get("type", "value_error")})
    return JSONResponse(status_code=422,
                         content={"error": {"code": "validation_error",
                                            "message": "invalid request",
                                            "details": {"fields": fields}}})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StartRun(BaseModel):
    command: str = Field("run", description="Command: " + ", ".join(commands_mod.COMMANDS))
    params: dict = Field(default_factory=dict,
                         description="Command parameters; see GET /api/commands")
    client_token: str | None = Field(
        None, description="Idempotency token: retrying with the same token "
                          "returns the same run")


class RunSummary(BaseModel):
    run_id: str
    status: str
    started_at: str | None = None
    duration_s: float | None = None
    params_hash: str | None = None
    key_params: dict
    key_metrics: dict


class Health(BaseModel):
    ok: bool
    version: str
    active_run: str | None = None


class DatasetRow(BaseModel):
    id: str
    name: str
    kind: str
    size_bytes: int
    group: str


class ArtifactRow(BaseModel):
    name: str
    type: str
    size_bytes: int
    url: str


MIME = {".csv": "text/csv", ".png": "image/png", ".json": "application/json",
        ".stl": "model/stl", ".ply": "model/ply", ".f32": "application/octet-stream",
        ".txt": "text/plain"}


def _artifact_rows(run_id: str) -> list[ArtifactRow]:
    root = manager._run_dir(run_id)
    rows = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            ext = os.path.splitext(fn)[1].lower()
            rows.append(ArtifactRow(
                name=rel, type=MIME.get(ext, "application/octet-stream"),
                size_bytes=os.path.getsize(full),
                url=f"/api/runs/{run_id}/artifacts/{rel}"))
    rows.sort(key=lambda r: r.name)
    return rows


def _safe_artifact_path(run_id: str, name: str) -> str:
    root = os.path.realpath(manager._run_dir(run_id))
    full = os.path.realpath(os.path.join(root, name))
    if not full.startswith(root + os.sep) or not os.path.isfile(full):
        raise ApiError(404, "artifact_not_found", f"no artifact {name!r}",
                       {"run_id": run_id})
    return full


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=Health, operation_id="get_health",
         tags=["meta"])
def get_health() -> Health:
    """Liveness + the active run id (open mid-run pages reconnect to it)."""
    return Health(ok=True, version=VERSION, active_run=manager.active_run_id())


@app.get("/api/commands", operation_id="get_commands", tags=["meta"])
def get_commands() -> dict:
    """Every command (run, measure, register, chamfer, ensemble, batch) with
    its parameters: type, default, choices, help. The CLI, this API and the
    UI form are all generated from this one registry."""
    return {"commands": commands_mod.web_schema(),
            "groups": datasets_mod.list_groups()}


@app.get("/api/schema", operation_id="get_schema", tags=["meta"])
def get_schema() -> dict:
    """Parameter list for ``run`` (kept for older clients; see
    GET /api/commands for all commands)."""
    tmap = {"dataset": "string", "datasets": "array", "group": "string", "bool": "boolean",
            "int": "integer", "float": "number", "enum": "string", "str": "string",
            "path": "string"}
    props = {}
    for p in commands_mod.web_schema()["run"]["params"]:
        q = {k: v for k, v in p.items() if k != "name"}
        q["kind"] = p["type"]
        q["type"] = tmap.get(p["type"], "string")
        if "choices" in q:
            q["enum"] = q.pop("choices")
        props[p["name"]] = q
    return {"type": "object", "required": ["ref", "test"], "properties": props}


@app.get("/api/datasets", response_model=list[DatasetRow],
         operation_id="list_datasets", tags=["datasets"])
def list_datasets() -> list[DatasetRow]:
    """Geometry files available to align: repo data dirs + uploads."""
    return [DatasetRow(**d) for d in datasets_mod.list_datasets()]


@app.post("/api/datasets/upload", response_model=DatasetRow,
          operation_id="upload_dataset", tags=["datasets"])
def upload_dataset(file: UploadFile = File(description=(
        "Geometry file: .stl .obj .ply .pcd .xyz .pts .asc .txt, up to 500 MB")),
        folder: str | None = Form(None, description=(
            "Optional upload sub-folder (batch pairs files by folder)"))) -> DatasetRow:
    """Upload a mesh/point-cloud to use as an input."""
    try:
        row = datasets_mod.save_upload(file.filename, file.file, folder=folder)
    except ValueError as e:
        raise ApiError(422, "invalid_upload", str(e), {})
    return DatasetRow(**row)


def _resolve_inputs(command: str, p: dict) -> dict:
    """Map dataset/group ids in ``p`` to absolute paths (``<name>_path``,
    ``<name>_paths``, ``<name>_dir``); 422 ``unknown_dataset`` on a bad id."""
    full = dict(p)
    for spec in commands_mod.COMMANDS[command]["params"]:
        name, t = spec["name"], spec["type"]
        v = p.get(name)
        if v is None:
            continue
        try:
            if t == "dataset":
                full[name + "_path"] = datasets_mod.resolve(v)
            elif t == "datasets":
                full[name + "_paths"] = [datasets_mod.resolve(x) for x in v]
            elif t == "group":
                full[name + "_dir"] = datasets_mod.resolve_group(v)
        except KeyError as e:
            raise ApiError(422, "unknown_dataset",
                           f"dataset {e.args[0]!r} not found",
                           {"field": name, "available": "GET /api/datasets"})
    return full


@app.post("/api/runs", status_code=202, operation_id="start_run",
          tags=["runs"])
def start_run(body: StartRun) -> dict:
    """Start a command run (default ``run``). Returns 202 {run_id} immediately;
    poll GET /api/runs/{id} or stream /api/runs/{id}/events. 409 if a run is
    already active. Same client_token returns the same run."""
    command = body.command
    if command not in commands_mod.COMMANDS:
        raise ApiError(422, "unknown_command", f"unknown command {command!r}",
                       {"available": list(commands_mod.COMMANDS)})
    raw = dict(body.params)
    # backwards compatibility with the pre-registry run form
    if command == "run" and raw.pop("use_icp", True) is False:
        command = "measure"
    if "area_weighted" in raw:
        raw["per_vertex_mae"] = not raw.pop("area_weighted")
    try:
        p = commands_mod.validate(command, raw, web=True)
    except commands_mod.ParamError as e:
        raise ApiError(422, "validation_error", "invalid request",
                       {"fields": e.problems})
    p = {k: v for k, v in p.items() if v is not None}
    full = _resolve_inputs(command, p)
    full["command"] = command
    if command in ("run", "measure"):
        full["use_icp"] = command == "run"
    try:
        return manager.start(full, client_token=body.client_token)
    except ActiveRunError as e:
        raise ApiError(409, "run_active",
                       "another run is already active",
                       {"active_run_id": e.active_run_id})


@app.get("/api/runs", response_model=list[RunSummary],
         operation_id="list_runs", tags=["runs"])
def list_runs(limit: int = Query(50, ge=1, le=200)) -> list[RunSummary]:
    """Past runs, newest first: status, duration, key params + metrics."""
    return [RunSummary(**r) for r in manager.list_runs(limit=limit)]


@app.get("/api/runs/{run_id}", operation_id="get_run", tags=["runs"])
def get_run(run_id: str = Path(description="Run id"),
            wait: float = Query(0, ge=0, le=60,
                                description="Long-poll up to this many seconds "
                                            "for a terminal state")) -> dict:
    """One run's status/progress/metrics/params/meta. ``?wait=<s>`` blocks
    until the run is terminal or the timeout (max 60 s)."""
    import time as _time
    deadline = _time.monotonic() + (wait or 0)
    while True:
        rec = manager.get_run(run_id)
        if rec is None:
            raise ApiError(404, "run_not_found", f"no run {run_id!r}",
                           {"run_id": run_id})
        if rec.get("status") in runner_mod.TERMINAL or _time.monotonic() >= deadline:
            return rec
        _time.sleep(0.25)


@app.get("/api/runs/{run_id}/events", operation_id="stream_events",
         tags=["runs"])
async def stream_events(run_id: str = Path(description="Run id"),
                         request: Request = None,
                         last_event_id: int = Query(
                             0, alias="lastEventId", ge=0,
                             description="Resume after this SSE event id")) \
        -> StreamingResponse:
    """SSE stream for a run: status, progress, log, metric, done events with
    increasing ids (resumable via Last-Event-ID / lastEventId)."""

    async def gen():
        yield "retry: 2000\n\n"
        for eid, etype, data in manager.subscribe(run_id, last_event_id):
            payload = json.dumps(data)
            yield f"id: {eid}\nevent: {etype}\ndata: {payload}\n\n"
            if etype == "done":
                return

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/runs/{run_id}/cancel", operation_id="cancel_run",
          tags=["runs"])
def cancel_run(run_id: str = Path(description="Run id")) -> dict:
    """Cancel the active run (terminal runs report their status unchanged)."""
    try:
        return manager.cancel(run_id)
    except KeyError:
        raise ApiError(404, "run_not_found", f"no run {run_id!r}",
                       {"run_id": run_id})


@app.get("/api/runs/{run_id}/artifacts", response_model=list[ArtifactRow],
         operation_id="list_artifacts", tags=["runs"])
def list_artifacts(run_id: str = Path(description="Run id")) -> list[ArtifactRow]:
    """Everything the run wrote: CSVs, aligned mesh, histogram, viewer files."""
    if manager.get_run(run_id) is None:
        raise ApiError(404, "run_not_found", f"no run {run_id!r}",
                       {"run_id": run_id})
    return _artifact_rows(run_id)


@app.get("/api/runs/{run_id}/artifacts/{name:path}",
         operation_id="get_artifact", tags=["runs"], response_class=FileResponse)
def get_artifact(run_id: str = Path(description="Run id"),
                 name: str = Path(description="Artifact path within the run")) \
        -> FileResponse:
    """Download one artifact file (CSV, mesh, PNG, JSON, error arrays)."""
    path = _safe_artifact_path(run_id, name)
    ext = os.path.splitext(path)[1].lower()
    return FileResponse(path, media_type=MIME.get(ext, "application/octet-stream"),
                        filename=os.path.basename(path))


@app.get("/api/runs/{run_id}/export", operation_id="export_run",
         tags=["runs"], response_class=FileResponse)
def export_run(run_id: str = Path(description="Run id")) -> FileResponse:
    """Zip of the whole run directory (params, artifacts, viewer files, log).
    409 while the run is still active (the zip would be partial)."""
    import tempfile
    from starlette.background import BackgroundTask
    rec = manager.get_run(run_id)
    if rec is None:
        raise ApiError(404, "run_not_found", f"no run {run_id!r}",
                       {"run_id": run_id})
    if rec.get("status") not in runner_mod.TERMINAL:
        raise ApiError(409, "run_active", "run has not finished yet",
                       {"run_id": run_id})
    root = manager._run_dir(run_id)
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                zf.write(full, os.path.relpath(full, root))
    return FileResponse(tmp.name, media_type="application/zip",
                        filename=f"meshalign_run_{run_id}.zip",
                        background=BackgroundTask(os.unlink, tmp.name))


@app.get("/api/runs/{run_id}/view", operation_id="get_view",
         tags=["runs"])
def get_view(run_id: str = Path(description="Run id")) -> dict:
    """Viewer payload for the UI/agents: summary.json + viewer file URLs with
    sizes (mesh geometry, error arrays, histogram, stats)."""
    if manager.get_run(run_id) is None:
        raise ApiError(404, "run_not_found", f"no run {run_id!r}",
                       {"run_id": run_id})
    summary_path = os.path.join(manager._run_dir(run_id), "viewer", "summary.json")
    if not os.path.isfile(summary_path):
        raise ApiError(409, "view_not_ready",
                       "this run has no viewer payload yet",
                       {"hint": "wait for status=succeeded"})
    with open(summary_path) as f:
        summary = json.load(f)
    # absolutize viewer URLs
    def fix(node):
        if isinstance(node, dict):
            return {k: (f"/api/runs/{run_id}/artifacts/{v}"
                        if k in ("url", "stats_url") and isinstance(v, str)
                        and not v.startswith("/")
                        else fix(v))
                    for k, v in node.items()}
        if isinstance(node, list):
            return [fix(x) for x in node]
        return node
    return fix(summary)


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")),
          name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/{spa_path:path}", include_in_schema=False)
def spa(spa_path: str):
    if spa_path.startswith("api/"):
        raise ApiError(404, "not_found", f"unknown API path /{spa_path}")
    candidate = os.path.realpath(os.path.join(APP_DIR, "static", spa_path))
    static_root = os.path.realpath(os.path.join(APP_DIR, "static"))
    if (candidate.startswith(static_root + os.sep) and os.path.isfile(candidate)
            and ".." not in spa_path):
        return FileResponse(candidate)
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))
