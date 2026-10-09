"""Job entry points for every console command (runs inside the job subprocess).

``run`` / ``measure``  -> ``alignment.run_alignment`` (3D viewer + metrics)
``register``           -> ``run_register`` (aligned mesh + viewer, no errors)
``chamfer`` / ``ensemble`` / ``batch`` -> ``run_generic`` (structured result.json)

Everything delegates to ``meshalign.commands.execute`` so the web UI, the HTTP
API and the CLI run the exact same code.
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np

import alignment
from alignment import (_cloud_arrays, _json_dump, _mesh_arrays, write_binary_ply)


def _write_aligned_viewer(ref, test, T, ref_kind, test_kind, viewer_dir):
    """ref.ply + aligned.ply (no error arrays) -> the ``view`` summary block."""
    aligned = copy.deepcopy(test)
    aligned.transform(T)
    if test_kind == "mesh":
        aligned.compute_vertex_normals()
    rv, rn, rf = _mesh_arrays(ref) if ref_kind == "mesh" else _cloud_arrays(ref)
    tv, tn, tf = _mesh_arrays(aligned) if test_kind == "mesh" else _cloud_arrays(aligned)
    write_binary_ply(os.path.join(viewer_dir, "ref.ply"), rv, rn, rf)
    write_binary_ply(os.path.join(viewer_dir, "aligned.ply"), tv, tn, tf)
    return {
        "ref": {"url": "viewer/ref.ply", "kind": ref_kind, "n_points": int(len(rv)),
                "n_faces": int(len(rf)) if rf is not None else 0, "unit": "mm"},
        "aligned": {"url": "viewer/aligned.ply", "kind": test_kind, "n_points": int(len(tv)),
                    "n_faces": int(len(tf)) if tf is not None else 0, "unit": "mm"}}


def run_register(params: dict, run_dir: str, evq) -> dict:
    """``meshalign register``: registration only. Saves the aligned mesh/cloud
    as ``aligned.stl|ply`` and a viewer payload without error arrays."""
    import meshalign as ma
    from meshalign import commands

    t_start = time.time()
    viewer_dir = os.path.join(run_dir, "viewer")
    os.makedirs(viewer_dir, exist_ok=True)
    ref_kind, ref = ma.load_any(params["ref_path"])
    test_kind, test = ma.load_any(params["test_path"], estimate_normals=True)
    evq.put(("progress", {"progress": 0.1}))
    evq.put(("log", {"level": "INFO",
                     "msg": f"registering (path={params.get('path', 'auto')}, "
                            f"mode={params.get('mode', 'default')}, seed={params.get('seed', 0)})"}))
    ext = ".ply" if test_kind == "cloud" else ".stl"
    write = params.get("write", True) is not False
    res = commands.execute("register", {
        "ref": params["ref_path"], "test": params["test_path"],
        "path": params.get("path", "auto"), "mode": params.get("mode", "default"),
        "seed": params.get("seed", 0), "write": write,
        "out": os.path.join(run_dir, "aligned" + ext)})
    T = np.asarray(res["transform_4x4"])
    evq.put(("progress", {"progress": 0.8}))
    view = _write_aligned_viewer(ref, test, T, ref_kind, test_kind, viewer_dir)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    trans = float(np.linalg.norm(T[:3, 3]))
    evq.put(("log", {"level": "INFO",
                     "msg": f"registered: rotation {ang:.3f} deg, translation {trans:.3f} mm"}))
    summary = {
        "command": "register",
        "inputs": {"ref": {"name": os.path.basename(params["ref_path"]), "kind": ref_kind},
                   "test": {"name": os.path.basename(params["test_path"]), "kind": test_kind}},
        "transform_4x4": [[round(float(x), 6) for x in row] for row in T],
        "headline": {"rotation_deg": round(ang, 4), "translation_mm": round(trans, 4)},
        "t1_diagnostics": {}, "metrics_bundle": {},
        "timings_s": {"total": round(time.time() - t_start, 3)},
        "meshalign_version": getattr(ma, "__version__", None),
        "view": view,
    }
    _json_dump(summary, os.path.join(viewer_dir, "summary.json"))
    _json_dump(dict(summary["headline"], duration_s=summary["timings_s"]["total"]),
               os.path.join(run_dir, "metrics.json"))
    _json_dump({"transform_4x4": summary["transform_4x4"],
                "aligned": os.path.basename(res["aligned_path"]) if res.get("aligned_path") else None},
               os.path.join(run_dir, "result.json"))
    evq.put(("progress", {"progress": 1.0}))
    return summary


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return _jsonable(o.item())
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def run_generic(params: dict, run_dir: str, evq) -> dict:
    """chamfer / ensemble / batch: no 3D viewer, a structured ``result.json``."""
    from meshalign import commands

    command = params["command"]
    t_start = time.time()
    if command == "chamfer":
        p = {"a": params["a_path"], "b": params["b_path"],
             "sample": params.get("sample", 12000), "seed": params.get("seed", 0)}
    elif command == "ensemble":
        p = {"meshes": params["meshes_paths"], "hub_index": params.get("hub_index"),
             "seed": params.get("seed", 0), "out": os.path.join(run_dir, "ensemble.csv")}
    elif command == "batch":
        p = {"refs": params["refs_dir"], "tests": params["tests_dir"], "out": run_dir,
             "jobs": params.get("jobs"), "mode": params.get("mode", "default"),
             "icp": params.get("icp", True), "absolute": params.get("absolute", False),
             "pattern": params.get("pattern", "*"), "seed_base": params.get("seed_base", 0)}
    else:
        raise ValueError(f"unknown command {command!r}")
    evq.put(("progress", {"progress": 0.05}))
    evq.put(("log", {"level": "INFO", "msg": f"running {command}"}))

    def progress(frac, msg):
        evq.put(("progress", {"progress": round(0.05 + 0.9 * frac, 4)}))
        evq.put(("log", {"level": "INFO", "msg": msg}))

    res = _jsonable(commands.execute(command, p, progress=progress))
    flat = {}
    if command == "chamfer":
        flat = {"chamfer_mm": res["chamfer"]}
    elif command == "ensemble":
        rows = res["rows"]
        flat = {"ensemble_median_um": res["median_um"], "ensemble_mad_um": res["mad_um"],
                "n_outliers": sum(1 for r in rows if r.get("ensemble_outlier")),
                "n_meshes": len(rows)}
    elif command == "batch":
        maes = [r["mae_um"] for r in res["results"] if r.get("ok")]
        flat = {"n_ok": res["n_ok"], "n_pairs": res["n_pairs"],
                "mean_mae_um": round(sum(maes) / len(maes), 4) if maes else None}
    flat["duration_s"] = round(time.time() - t_start, 3)
    _json_dump(res, os.path.join(run_dir, "result.json"))
    _json_dump(flat, os.path.join(run_dir, "metrics.json"))
    evq.put(("progress", {"progress": 1.0}))
    return res


def run_job(params: dict, run_dir: str, evq):
    """Dispatch one job by ``params["command"]`` (default ``run``)."""
    command = params.get("command", "run")
    if command in ("run", "measure"):
        return alignment.run_alignment(params, run_dir, evq)
    if command == "register":
        return run_register(params, run_dir, evq)
    return run_generic(params, run_dir, evq)
