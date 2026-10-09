"""Run manager for the meshalign console webapp (job profile, style guide §3-4).

One active run at a time. A run is a state machine
``queued -> running -> succeeded | failed | cancelled``. The alignment itself
executes in a spawned subprocess (``alignment.run_alignment``) so Open3D work
never blocks the event loop and cancellation can hard-terminate native code;
the child streams (type, data) events home over a multiprocessing Queue which
this module turns into the run's persisted event log + SSE feed.

Every run persists to ``runs/{run_id}/``:
  params.json, meta.json, metrics.json, log.txt, viewer/, pair_* artifacts.

Importable and testable without a server: ``RunManager`` is plain Python.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import queue as queue_mod
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone

RUNS_DEFAULT = os.environ.get(
    "MESHALIGN_RUNS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))

TERMINAL = ("succeeded", "failed", "cancelled")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _kill_tree(proc) -> None:
    """Terminate a job process AND its children (batch pool workers)."""
    if proc is None or proc.pid is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=15)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        if proc.is_alive():
            proc.kill()
    except Exception:
        pass


def run_label(params: dict) -> str:
    """Short human label for history rows: what this run was about."""
    c = params.get("command", "run")
    base = lambda k: os.path.basename(str(params.get(k) or ""))
    if c in ("run", "measure", "register"):
        return f"{base('ref')} → {base('test')}"
    if c == "chamfer":
        return f"{base('a')} ↔ {base('b')}"
    if c == "ensemble":
        return f"{len(params.get('meshes') or [])} meshes"
    if c == "batch":
        return f"{params.get('refs')} vs {params.get('tests')}"
    return c


def _git_commit() -> str | None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                               capture_output=True, text=True,
                               timeout=10).stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Child process entry
# ---------------------------------------------------------------------------

def _child_main(run_dir: str, params: dict, evq) -> None:
    """Spawned-process entry: redirect stdout/stderr into the event stream,
    then run the job. The parent may terminate this process to cancel."""
    sys_path_hack = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    import sys
    if sys_path_hack not in sys.path:
        sys.path.insert(0, sys_path_hack)
    import alignment
    import jobs
    from alignment import EmitLog
    if os.name != "nt":
        os.setsid()    # own process group: cancel can kill batch pool workers too
    sys.stdout = EmitLog(evq, "INFO")
    sys.stderr = EmitLog(evq, "ERROR")
    try:
        jobs.run_job(params, run_dir, evq)
        evq.put(("status", {"status": "succeeded"}))
    except Exception as e:
        import traceback
        evq.put(("log", {"level": "ERROR", "msg": f"{type(e).__name__}: {e}"}))
        evq.put(("log", {"level": "ERROR", "msg": traceback.format_exc()}))
        evq.put(("status", {"status": "failed", "message": f"{type(e).__name__}: {e}"}))
    finally:
        # flush the queue feeder so no buffered events are lost at exit
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            evq.close()
            evq.join_thread()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RunManager
# ---------------------------------------------------------------------------

class RunManager:
    def __init__(self, runs_dir: str | None = None):
        self.runs_dir = runs_dir or RUNS_DEFAULT
        os.makedirs(self.runs_dir, exist_ok=True)
        self._lock = threading.RLock()  # re-entrant: start() broadcasts under it
        self._active: dict | None = None          # in-memory active-run record
        self._client_tokens: dict[str, str] = {}  # token -> run_id
        self._subscribers: dict[str, set] = {}    # run_id -> per-subscriber queues
        self._restart_safety()
        self._load_tokens()

    # -- persistence --------------------------------------------------------

    def _run_dir(self, run_id: str) -> str:
        return os.path.join(self.runs_dir, run_id)

    def _read_json(self, run_id: str, name: str) -> dict | None:
        path = os.path.join(self._run_dir(run_id), name)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _restart_safety(self):
        """Any run recorded as running when the server died is marked failed."""
        for run_id in os.listdir(self.runs_dir):
            meta = self._read_json(run_id, "meta.json")
            if meta and meta.get("status") not in TERMINAL:
                meta["status"] = "failed"
                meta["message"] = "server restarted during run"
                meta["end_time"] = utcnow()
                self._write_meta(run_id, meta)

    _MAX_TOKENS = 1000

    def _load_tokens(self):
        """Rebuild the idempotency map from persisted meta (newest wins)."""
        for run_id in sorted(os.listdir(self.runs_dir))[-self._MAX_TOKENS:]:
            meta = self._read_json(run_id, "meta.json")
            if meta and meta.get("client_token"):
                self._client_tokens[meta["client_token"]] = run_id

    def _remember_token(self, token: str, run_id: str):
        self._client_tokens[token] = run_id
        while len(self._client_tokens) > self._MAX_TOKENS:
            self._client_tokens.pop(next(iter(self._client_tokens)))

    def _write_meta(self, run_id: str, meta: dict):
        os.makedirs(self._run_dir(run_id), exist_ok=True)
        with open(os.path.join(self._run_dir(run_id), "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    def _append_log_line(self, run_id: str, line: str):
        os.makedirs(self._run_dir(run_id), exist_ok=True)
        with open(os.path.join(self._run_dir(run_id), "log.txt"), "a") as f:
            f.write(line + "\n")

    # -- queries ------------------------------------------------------------

    def active_run_id(self) -> str | None:
        with self._lock:
            return self._active["run_id"] if self._active else None

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            if self._active and self._active["run_id"] == run_id:
                rec = {k: v for k, v in self._active.items()
                       if k not in ("proc", "evq", "events", "start_wall",
                                    "client_token")}
                rec["progress"] = self._active.get("progress", 0.0)
                rec["params"] = self._active["params"]
                return rec
        meta = self._read_json(run_id, "meta.json")
        if meta is None:
            return None
        params = self._read_json(run_id, "params.json") or {}
        rec = {"run_id": run_id, **meta, "params": params}
        if meta.get("status") == "succeeded":
            m = self._read_json(run_id, "metrics.json")
            if m:
                rec["metrics"] = m
            r = self._read_json(run_id, "result.json")
            if r is not None:
                rec["result"] = r
        return rec

    def list_runs(self, limit: int = 50) -> list[dict]:
        out = []
        for run_id in sorted(os.listdir(self.runs_dir), reverse=True):
            if run_id.startswith("."):
                continue
            rec = self.get_run(run_id)
            if rec is None:
                continue
            metrics = rec.get("metrics") or {}
            out.append({
                "run_id": run_id,
                "status": rec.get("status"),
                "started_at": rec.get("start_time"),
                "duration_s": rec.get("duration_s"),
                "params_hash": rec.get("params_hash"),
                "key_params": dict(
                    {k: rec.get("params", {}).get(k)
                     for k in ("ref", "test", "mode", "use_icp")},
                    command=rec.get("params", {}).get("command", "run"),
                    label=run_label(rec.get("params", {}))),
                "key_metrics": {k: metrics.get(k) for k in (
                    "mae_um", "p95_um", "iou", "chamfer_mm", "rotation_deg",
                    "n_ok", "n_pairs", "mean_mae_um", "ensemble_median_um")},
            })
            if len(out) >= limit:
                break
        return out

    # -- starting -----------------------------------------------------------

    def start(self, params: dict, client_token: str | None = None) -> dict:
        with self._lock:
            # idempotency first: a retry with the same token returns the same run
            # even while that run is still active
            if client_token and client_token in self._client_tokens:
                rid = self._client_tokens[client_token]
                rec = self.get_run(rid)
                if rec is not None:
                    return rec
            if self._active:
                raise ActiveRunError(self._active["run_id"])
            if client_token and client_token in self._client_tokens:
                rid = self._client_tokens[client_token]
                rec = self.get_run(rid)
                if rec:
                    return rec
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            meta = {
                "run_id": run_id,
                "status": "queued",
                "start_time": utcnow(),
                "end_time": None,
                "duration_s": None,
                "params_hash": params_hash(params),
                "seed": params.get("seed", 0),
                "git_commit": _git_commit(),
                "hostname": socket.gethostname(),
                "message": None,
                "command": params.get("command", "run"),
                "client_token": client_token,
            }
            os.makedirs(self._run_dir(run_id), exist_ok=True)
            with open(os.path.join(self._run_dir(run_id), "params.json"), "w") as f:
                json.dump(params, f, indent=1)
            self._write_meta(run_id, meta)
            if client_token:
                self._remember_token(client_token, run_id)

            ctx = multiprocessing.get_context("spawn")
            evq = ctx.Queue()
            proc = ctx.Process(target=_child_main,
                               args=(self._run_dir(run_id), params, evq),
                               # batch spawns a worker pool; daemonic processes may not have children
                               daemon=params.get("command") != "batch")
            proc.start()
            self._active = {
                "run_id": run_id, "status": "running",
                "started_at": meta["start_time"], "params": params,
                "params_hash": meta["params_hash"], "seed": meta["seed"],
                "progress": 0.0, "message": "starting",
                "proc": proc, "evq": evq, "events": [],
                "next_event_id": 1, "client_token": client_token,
                "start_wall": time.monotonic(),
            }
            threading.Thread(target=self._pump, args=(run_id,), daemon=True).start()
            self._broadcast(run_id, "status", {"status": "running",
                                               "message": "starting"})
            return {"run_id": run_id, "status": "running"}

    # -- event pump ----------------------------------------------------------

    def _pump(self, run_id: str):
        """Drain the child event queue into the run record + SSE subscribers."""
        with self._lock:
            rec = self._active
        if rec is None or rec["run_id"] != run_id:
            return
        evq, proc = rec["evq"], rec["proc"]
        while True:
            try:
                item = evq.get(timeout=0.5)
            except queue_mod.Empty:
                item = None
            with self._lock:
                rec = self._active
            if rec is None or rec["run_id"] != run_id:
                return  # run was cancelled + finalized
            if proc.exitcode is not None and item is None:
                # child exited without a terminal status event (feeder race
                # or hard crash): decide from what it left behind
                finished = os.path.isfile(
                    os.path.join(self._run_dir(run_id), "viewer", "summary.json"))
                if proc.exitcode == 0 and finished:
                    self._finalize(run_id, status="succeeded")
                else:
                    self._finalize(run_id, status="failed",
                                   message=f"job process exited (code {proc.exitcode})")
                return
            if item is None:
                continue
            etype, data = item
            if etype == "status":
                if data.get("status") == "succeeded":
                    self._finalize(run_id, status="succeeded")
                    return
                if data.get("status") == "failed":
                    self._finalize(run_id, status="failed",
                                   message=data.get("message", "job failed"))
                    return
                rec["status"] = data.get("status", rec["status"])
                rec["message"] = data.get("message", rec["message"])
            elif etype == "progress":
                rec["progress"] = float(data.get("progress", 0) or 0)
            elif etype == "log":
                line = f"[{utcnow()}] {data.get('level','INFO'):5s} {data.get('msg','')}"
                self._append_log_line(run_id, line)
            elif etype == "metric":
                pass
            self._record_and_broadcast(run_id, etype, data)

    def _record_and_broadcast(self, run_id: str, etype: str, data: dict):
        with self._lock:
            rec = self._active
            if rec is None or rec["run_id"] != run_id:
                return
            eid = rec["next_event_id"]
            rec["next_event_id"] += 1
            rec["events"].append({"id": eid, "event": etype,
                                  "data": data, "ts": utcnow()})
            subs = list(self._subscribers.get(run_id, ()))
        for q in subs:
            q.put((eid, etype, data))

    def _broadcast(self, run_id: str, etype: str, data: dict):
        self._record_and_broadcast(run_id, etype, data)

    def _finalize(self, run_id: str, status: str, message: str | None = None):
        with self._lock:
            rec = self._active
            if rec is None or rec["run_id"] != run_id:
                return
            proc = rec.get("proc")
            duration = round(time.monotonic() - rec["start_wall"], 3)
            # persist the terminal state BEFORE broadcasting done, so a client
            # that immediately GETs the run sees the final record + metrics
            meta = self._read_json(run_id, "meta.json") or {}
            meta.update({"status": status, "end_time": utcnow(),
                         "duration_s": duration, "message": message})
            self._write_meta(run_id, meta)
            self._active = None
            eid = rec["next_event_id"]
            rec["next_event_id"] += 1
            final = {"run_id": run_id, "status": status, "message": message,
                     "duration_s": duration}
            rec["events"].append({"id": eid, "event": "done", "data": final,
                                  "ts": utcnow()})
            subs = list(self._subscribers.get(run_id, ()))
            for q in subs:
                q.put((eid, "done", final))
        if proc is not None and proc.is_alive():
            _kill_tree(proc)
            proc.join(timeout=5)

    # -- cancel --------------------------------------------------------------

    def cancel(self, run_id: str) -> dict:
        with self._lock:
            rec = self._active
            is_active = rec is not None and rec["run_id"] == run_id
        if is_active:
            self._append_log_line(run_id, f"[{utcnow()}] INFO  cancel requested")
            self._finalize(run_id, status="cancelled", message="cancelled by user")
            return {"run_id": run_id, "status": "cancelled"}
        meta = self._read_json(run_id, "meta.json")
        if meta is None:
            raise KeyError(run_id)
        return {"run_id": run_id, "status": meta.get("status")}

    # -- SSE -----------------------------------------------------------------

    def subscribe(self, run_id: str, last_event_id: int = 0):
        """Yield (id, event, data) tuples; replays from last_event_id, then
        follows the active run live, then replays the terminal `done` event."""
        q = queue_mod.Queue()
        with self._lock:
            rec = self._active
            if rec is not None and rec["run_id"] == run_id:
                for ev in rec["events"]:
                    if ev["id"] > last_event_id:
                        q.put((ev["id"], ev["event"], ev["data"]))
                self._subscribers.setdefault(run_id, set()).add(q)
                live = True
            else:
                live = False
        if live:
            try:
                while True:
                    item = q.get(timeout=30)
                    yield item
                    if item[1] == "done":
                        return
            finally:
                with self._lock:
                    self._subscribers.get(run_id, set()).discard(q)
        else:
            # terminal run: replay persisted log as log events + a done event
            meta = self._read_json(run_id, "meta.json")
            if meta is None:
                return
            eid = 0
            log_path = os.path.join(self._run_dir(run_id), "log.txt")
            if os.path.isfile(log_path):
                with open(log_path) as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        eid += 1
                        if eid > last_event_id:
                            yield eid, "log", {"level": "INFO", "msg": line}
            eid += 1
            final = {"run_id": run_id, "status": meta.get("status"),
                     "message": meta.get("message"),
                     "duration_s": meta.get("duration_s")}
            yield eid, "done", final


class ActiveRunError(Exception):
    def __init__(self, active_run_id: str):
        self.active_run_id = active_run_id
        super().__init__(f"a run is already active: {active_run_id}")
