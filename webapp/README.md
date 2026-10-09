# meshalign console (webapp)

Job-profile web app for the meshalign package: pick (or upload) a reference
and a test mesh/cloud, run the alignment in the background, and inspect the
result — 3D error map (signed per-vertex surface error on the aligned pair),
error histogram, metric cards, canonical CSV artifacts, and full provenance.

- **Branch:** `main`
- **Profiles:** core + **job** (alignments take seconds-to-minutes and run in
  a subprocess; §3–4 of the lab webapp style guide)
- Humans use the browser UI at `/`; agents use the HTTP API
  (`/openapi.json`, `client.py`).


## Run

```bash
cd webapp
uvicorn app:app --host 0.0.0.0 --port 8766
```

Configuration (environment variables):

| Var | Default | Meaning |
|---|---|---|
| `MESHALIGN_RUNS_DIR` | `webapp/runs` | persistent run outputs (one dir per run) |
| `MESHALIGN_UPLOADS_DIR` | `webapp/uploads` | user-uploaded geometry files |

Point both at local scratch for speed, e.g.
`MESHALIGN_RUNS_DIR=/scratch/afm8/meshalign_webapp/runs`.

Python 3.11+ with `fastapi`, `uvicorn`, `python-multipart`, plus the
meshalign package deps (`open3d`, `numpy`, `matplotlib`; `pandas`/`scipy`
for the metrics bundle). The app imports `meshalign` from the repo's
`src/` automatically; a `pip install -e .` also works.

## What a run does

`POST /api/runs` with `{command, params}` where `command` is one of
`run`, `measure`, `register`, `chamfer`, `ensemble`, `batch` -- exactly the
CLI subcommands, with the same options (`GET /api/commands` lists every
parameter; the CLI parsers, the API and the UI form are all generated from
`meshalign.commands.COMMANDS`, so a new CLI flag shows up everywhere). The
`run` command:

1. loads both inputs (`load_any`: STL/OBJ meshes, PLY/PCD/XYZ clouds),
2. registers with the package's public routing aligner `ma.register`
   (the same path `run_pair` uses; identity-bay + yaw-seed arbitration),
3. measures via `ma.run_pair(use_icp=False)` on the pre-aligned pair, so
   all canonical artifacts are produced unchanged (per-vertex CSVs, aligned
   STL/PLY, histogram PNG, metrics bundle, T1 area-weighted stats),
4. writes viewer files (binary PLY meshes, per-vertex error arrays,
   histogram bins, summary JSON) for the 3D/histogram views.

Each run persists to `runs/{run_id}/`:
`params.json`, `meta.json` (params hash, seed, git commit, host, times),
`metrics.json`, `log.txt`, `pair_*` artifacts, `viewer/*`.
The server marks any run left `running` by a previous server process as
`failed` ("server restarted during run") on startup.

## Endpoints

| Method | Path | operation_id | What |
|---|---|---|---|
| GET | `/api/health` | `get_health` | liveness + `active_run` |
| GET | `/api/schema` | `get_schema` | parameter schema (type/default/unit/range) |
| GET | `/api/datasets` | `list_datasets` | repo `data/` files + uploads |
| POST | `/api/datasets/upload` | `upload_dataset` | upload a mesh/cloud (≤500 MB) |
| POST | `/api/runs` | `start_run` | start a run → 202 `{run_id}`; 409 if active |
| GET | `/api/runs` | `list_runs` | run history with key metrics |
| GET | `/api/runs/{id}?wait=<s>` | `get_run` | status/progress/metrics; long-poll |
| GET | `/api/runs/{id}/events` | `stream_events` | SSE: status/progress/log/metric/done |
| POST | `/api/runs/{id}/cancel` | `cancel_run` | cancel the active run |
| GET | `/api/runs/{id}/artifacts` | `list_artifacts` | everything the run wrote |
| GET | `/api/runs/{id}/artifacts/{name}` | `get_artifact` | download one file |
| GET | `/api/runs/{id}/export` | `export_run` | zip of the whole run dir |
| GET | `/api/commands` | `get_commands` | command registry + dataset folders (UI/agent form source) |
| GET | `/api/runs/{id}/view` | `get_view` | viewer payload (meshes, errors, histogram) |

All non-2xx responses are `{"error": {"code", "message", "details"}}`;
invalid params → 422 with per-field messages. `client_token` on
`POST /api/runs` makes starts idempotent.

## Agent example (client.py)

```python
from client import MeshalignClient

c = MeshalignClient("http://localhost:8766")
run = c.start_run(ref="mech_parts/Connector.stl", test="mech_parts/Connector_test.stl",
                  mode="default")
# any other CLI command:
c.start("register", ref="mech_parts/Connector.stl", test="mech_parts/Connector_test.stl", path="global")
c.start("chamfer", a="mech_parts/Connector.stl", b="mech_parts/Connector_test.stl")
c.start("ensemble", meshes=["mech_parts/Connector.stl", "mech_parts/Connector_test.stl"])
c.upload("ref.stl", folder="refs"); c.upload("ref.stl", folder="tests")   # batch pairs files by name
c.start("batch", refs="uploads/refs", tests="uploads/tests", mode="fast")
final = c.wait_terminal(run["run_id"])          # long-polls to terminal
print(final["metrics"]["mae_um"], "um MAE")
arts = c.artifacts(final["run_id"])
print(c.artifact_url(final["run_id"], "pair_aligned.stl"))
# stream live progress:
for ev in c.stream_events(final["run_id"]):
    print(ev["event"], ev["data"])
```

## Tests

```bash
pytest tests/test_api.py   # real runs: every command, 409, cancel, 422, restart safety
pytest tests/test_ui.py    # Playwright: full UI lifecycle incl. 3D canvas + histogram
```

## Files

- `app.py` — FastAPI routes (thin; logic lives below)
- `runner.py` — run state machine, subprocess management, SSE event bus
- `alignment.py` — the job: meshalign calls + viewer artifact writers
- `datasets.py` — dataset listing/upload + path resolution
- `static/index.html` — the UI (single file, vanilla JS + three.js + Plotly)
- `client.py` — one function per endpoint + job helpers
- `tests/` — pytest suites

## Compliance note (webapp-design-and-api-style-guide.md)

Profiles used: **core + job**. Live control (§5) does not apply — nothing
real-time or hardware.

Deviations:

1. **§3 cancellation** is subprocess-terminate, not a checked token: Open3D
   registration is native code that cannot be interrupted mid-call, so Stop
   SIGTERMs the job process (artifacts written so far are kept, the run is
   marked `cancelled`). The child does flush its event queue before exit.
2. **§3 progress** is stage-based (load 0.05 → register 0.15 → measure 0.7 →
   viewer 0.9 → 1.0) plus streamed log lines; registration itself is a single
   opaque native call, so there is no sub-progress within it.
3. **§4 `?wait=`** long-polls in 0.25 s steps and holds the request for up to
   60 s (the endpoint runs in Starlette's threadpool, so the event loop is
   never blocked).
4. **§9 tables "Compare"** is a metrics side-by-side table (run columns,
   delta vs first) rather than a plot overlay — alignment results are meshes
   and scalars, not time series; overlaying runs in 3D is not meaningful.
5. **§9 3D "decimated preview first"** is adapted: meshes over 500k faces
   start in points mode (with a notice; Toggle points loads the full mesh) —
   the full-resolution geometry is fetched once and rendered from GPU memory,
   and per-vertex error colors must stay index-aligned with the geometry.
6. **Vendored `static/vendor/plotly.min.js`, `three.module.js`,
   `OrbitControls.js`** instead of CDN tags: the serv box serves the lab on
   networks where external CDNs are unreachable; still zero build step.
7. **§15 test_ui** requires `playwright install chromium`; tests
   `importorskip` so a bare pytest run without browsers still passes
   (verified here with Chromium 153 — all UI tests pass).
8. The job subprocess is `multiprocessing` spawn: running the server via
   `python app.py`-style unguarded mains is unsupported (uvicorn /
   pytest / TestClient entry points are all safe).

Everything else (§1 API-first with operation_ids, §2 stack, §3 run dir
layout + restart safety + seeds, §4 endpoint names + SSE event types +
`client_token`, §6 job layout + tabs, §6a URL state, §7 monochrome tokens,
§8 RdBu signed-error colormap + units + tabular-nums, §9 components +
status pill + metric cards + log console, §10 state behavior table +
mid-run reconnect, §11 data-testids, §12 provenance + Copy params / Re-run /
Export, §13 empty/loading/error states, §14 do/don't) is followed.
