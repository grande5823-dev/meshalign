"""meshalign webapp domain: run one alignment job and build viewer artifacts.

This module runs INSIDE the job subprocess (spawned by ``runner.py``); it is
also importable standalone for tests. All heavy lifting (registration,
measurement, metrics) is delegated to the ``meshalign`` package API — this
module adds: staged progress events, stdout capture as log events, and the
webapp viewer artifacts (binary PLY meshes, per-vertex error arrays, histogram
bins, summary JSON).

Job function contract (style guide §15.2): takes ``params`` (a plain dict),
``run_dir`` (output directory) and ``evq`` (a queue-like ``emit(type, data)``
channel), and a ``cancelled`` callable checked between stages (cooperative;
the parent may also hard-terminate the process).
"""
from __future__ import annotations

import copy
import io
import json
import os
import struct
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

MESH_EXTS = (".stl", ".obj")
CLOUD_EXTS = (".ply", ".pcd", ".xyz", ".pts", ".asc", ".txt")
ALL_EXTS = MESH_EXTS + CLOUD_EXTS

# symmetric color limits: cover p1..p99 of |signed error|, floored at 1 um
CLIM_QUANTILES = (0.01, 0.99)
CLIM_MIN_MM = 1e-3


class EmitLog(io.TextIOBase):
    """File-like stdout/stderr replacement that forwards lines as log events."""

    def __init__(self, evq, level="INFO"):
        self.evq = evq
        self.level = level
        self._buf = ""

    def writable(self):
        return True

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.evq.put(("log", {"level": self.level, "msg": line.rstrip()}))
        return len(s)

    def flush(self):
        if self._buf.strip():
            self.evq.put(("log", {"level": self.level, "msg": self._buf.rstrip()}))
            self._buf = ""


def _json_dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def write_binary_ply(path, vertices, normals=None, faces=None):
    """Minimal binary PLY writer: float32 xyz [+ float32 normals] [+ uint32
    triangle indices]. three.js PLYLoader-compatible; we also parse this exact
    format client-side, so no third-party parser quirks."""
    n = len(vertices)
    props = [("property", "float", "x"), ("property", "float", "y"),
             ("property", "float", "z")]
    if normals is not None:
        props += [("property", "float", "nx"), ("property", "float", "ny"),
                  ("property", "float", "nz")]
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {n}"]
    for _, t, name in props:
        header.append(f"property {t} {name}")
    if faces is not None and len(faces):
        header.append(f"element face {len(faces)}")
        header.append("property list uchar int vertex_indices")
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        v = np.asarray(vertices, dtype="<f4")
        parts = [v]
        if normals is not None:
            parts.append(np.asarray(normals, dtype="<f4"))
        arr = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        arr.tofile(f)
        if faces is not None and len(faces):
            tri = np.asarray(faces, dtype="<i4")
            # PLY face records are packed per-face: [uchar count][int idx]*3
            buf = np.zeros((len(tri), 13), dtype=np.uint8)
            buf[:, 0] = 3
            buf[:, 1:].view("<i4")[:] = tri  # last axis contiguous -> view ok
            buf.tofile(f)


def read_binary_ply(path):
    """Parser for the exact format written above (for tests)."""
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    n_vertex = int([l for l in header.splitlines()
                    if l.startswith("element vertex")][0].split()[2])
    face_line = [l for l in header.splitlines() if l.startswith("element face")]
    n_face = int(face_line[0].split()[2]) if face_line else 0
    has_normals = "property float nx" in header
    v_itemsize = 3 + (3 if has_normals else 0)
    verts = np.frombuffer(data, dtype="<f4", count=n_vertex * v_itemsize,
                          offset=end).reshape(n_vertex, v_itemsize)
    vertices = verts[:, :3].copy()
    normals = verts[:, 3:].copy() if has_normals else None
    faces = None
    if n_face:
        off = end + n_vertex * v_itemsize * 4
        face_data = data[off:]
        faces = []
        i = 0
        for _ in range(n_face):
            cnt = face_data[i]; i += 1
            idx = struct.unpack_from("<iii", face_data, i); i += 12
            faces.append(idx)
        faces = np.asarray(faces, dtype=np.int64)
    return vertices, normals, faces


def _mesh_arrays(obj):
    """(vertices, normals, faces) for a mesh object; normals None if absent."""
    verts = np.asarray(obj.vertices)
    faces = np.asarray(obj.triangles) if len(obj.triangles) else None
    normals = np.asarray(obj.vertex_normals) if obj.has_vertex_normals() else None
    return verts, normals, faces


def _cloud_arrays(obj):
    verts = np.asarray(obj.points)
    normals = np.asarray(obj.normals) if obj.has_normals() else None
    return verts, normals, None


def _write_f32(path, values):
    np.asarray(values, dtype="<f4").tofile(path)


def _err_stats(errs_mm):
    """Symmetric color limits + headline stats for one error array."""
    e = np.abs(np.asarray(errs_mm, dtype=np.float64))
    if len(e) == 0:
        return {"count": 0, "clim_mm": CLIM_MIN_MM, "p50_abs_mm": None,
                "p95_abs_mm": None, "max_abs_mm": None, "mean_signed_mm": None}
    lo, hi = np.quantile(e, CLIM_QUANTILES)
    clim = max(float(hi), CLIM_MIN_MM)
    return {"count": int(len(e)), "clim_mm": clim,
            "p50_abs_mm": float(np.median(e)),
            "p95_abs_mm": float(np.quantile(e, 0.95)),
            "max_abs_mm": float(e.max()),
            "mean_signed_mm": float(np.mean(np.asarray(errs_mm, dtype=np.float64)))}


def _histogram(errs_mm, bins=80):
    e = np.asarray(errs_mm, dtype=np.float64)
    if len(e) == 0:
        return {"bins": bins, "counts": [], "edges": [], "stats": _err_stats(e)}
    clim = _err_stats(e)["clim_mm"]
    counts, edges = np.histogram(np.clip(e, -clim, clim), bins=bins,
                                 range=(-clim, clim))
    return {"bins": int(bins), "counts": [int(c) for c in counts],
            "edges": [round(float(x), 6) for x in edges],
            "clim_mm": clim, "stats": _err_stats(e)}


def run_alignment(params: dict, run_dir: str, evq) -> dict:
    """Execute one alignment job. Returns the summary dict (also written to
    ``run_dir/viewer/summary.json``). Emits ("progress"|"log"|"metric", data)
    events to ``evq``.

    Stages: load → register (optional) → measure (meshalign run_pair on the
    pre-aligned pair, so the exact pipeline artifacts are produced) → viewer
    artifacts → summary. Registration reuses the public ``meshalign.register``
    (the same routing aligner ``run_pair`` itself calls), then calls
    ``run_pair(use_icp=False)`` on the aligned inputs so the canonical
    CSVs/histogram/metrics bundle are written unchanged.
    """
    import meshalign as ma  # imported late: only inside the job process

    t_start = time.time()
    os.makedirs(run_dir, exist_ok=True)
    viewer_dir = os.path.join(run_dir, "viewer")
    os.makedirs(viewer_dir, exist_ok=True)

    ref_path = params["ref_path"]
    test_path = params["test_path"]
    mode = params.get("mode", "default")
    command = params.get("command", "run")
    use_icp = command != "measure"
    metrics_on = bool(params.get("metrics", True))
    per_vertex_mae = bool(params.get("per_vertex_mae", False))
    relative = bool(params.get("relative", False))
    want_hist = params.get("histogram")
    want_hist = True if want_hist is None else bool(want_hist)
    keep_per_sample = params.get("per_sample")
    keep_per_sample = True if keep_per_sample is None else bool(keep_per_sample)
    seed = int(params.get("seed", 0))

    evq.put(("log", {"level": "INFO",
                     "msg": f"loading ref={os.path.basename(ref_path)} "
                            f"test={os.path.basename(test_path)}"}))
    evq.put(("progress", {"progress": 0.05}))
    t0 = time.time()
    ref_kind, ref = ma.load_any(ref_path)
    test_kind, test = ma.load_any(test_path, estimate_normals=True)
    t_load = time.time() - t0
    evq.put(("log", {"level": "INFO",
                     "msg": f"loaded: ref={ref_kind} ({len(ref.vertices) if ref_kind=='mesh' else len(ref.points)} pts), "
                            f"test={test_kind} ({len(test.vertices) if test_kind=='mesh' else len(test.points)} pts)"}))

    # ---- register -----------------------------------------------------------
    T = np.eye(4)
    t0 = time.time()
    if use_icp:
        evq.put(("status", {"status": "running", "message": "registering"}))
        evq.put(("progress", {"progress": 0.15}))
        evq.put(("log", {"level": "INFO", "msg": f"registering (mode={mode}, seed={seed})"}))
        T = ma.register(ref, test, seed=seed, mode=mode)
        aligned = copy.deepcopy(test)
        aligned.transform(T)
        if test_kind == "mesh" and not aligned.has_vertex_normals():
            aligned.compute_vertex_normals()
        # persist the aligned input so run_pair measures exactly this pose
        ext = ".ply" if test_kind == "cloud" else ".stl"
        aligned_input = os.path.join(run_dir, f"aligned_input{ext}")
        if test_kind == "cloud":
            ma.save_point_cloud(aligned_input, aligned)
        else:
            import open3d as o3d
            o3d.io.write_triangle_mesh(aligned_input, aligned)
        ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        evq.put(("log", {"level": "INFO",
                         "msg": f"registered: rotation {ang:.3f} deg, "
                                f"translation {np.linalg.norm(T[:3, 3]):.3f} mm"}))
    else:
        aligned = copy.deepcopy(test)
        if test_kind == "mesh" and not aligned.has_vertex_normals():
            aligned.compute_vertex_normals()
        aligned_input = test_path
    t_register = time.time() - t0

    # ---- measure (canonical pipeline artifacts) -----------------------------
    evq.put(("progress", {"progress": 0.7}))
    evq.put(("log", {"level": "INFO", "msg": "measuring surface error (run_pair)"}))
    t0 = time.time()
    pair_fn = ma.run_pair_relative if relative else ma.run_pair
    pair_out = pair_fn(
        ref_path, aligned_input, run_dir, use_icp=False, prefix="pair",
        seed=seed, histogram=want_hist, save_per_sample=True, metrics=metrics_on,
        mode=mode, area_weighted=False if per_vertex_mae else None)
    mae_um, p95_um = pair_out[0], pair_out[1]
    t_measure = time.time() - t0
    evq.put(("metric", {"name": "mae_um", "value": round(mae_um, 4)}))
    evq.put(("metric", {"name": "p95_um", "value": round(p95_um, 4)}))

    # ---- gather errors per side ---------------------------------------------
    # run_pair's per-vertex errors attach to:
    #   mesh/mesh   -> ref vertices      (colorable side: ref)
    #   mesh/cloud  -> test points        (colorable side: test)
    #   cloud/mesh  -> ref points         (colorable side: ref)
    #   cloud/cloud -> test points, NN     (colorable side: test)
    errs_path = os.path.join(run_dir, "pair_surf_errors_mm.csv")
    errs = np.loadtxt(errs_path, delimiter=",", skiprows=1) if os.path.exists(errs_path) \
        else np.array([], dtype=np.float64)

    errs_ref = errs_test = None
    if ref_kind == "mesh" and test_kind == "mesh":
        errs_ref = errs
        # extra pass: test vertices -> ref surface (signed by test normals)
        evq.put(("log", {"level": "INFO", "msg": "computing test-side error map"}))
        e_test, _ = ma.compute_vertex_error_and_delta_nearest(aligned, ref)
        errs_test = np.asarray(e_test, dtype=np.float64)
    elif ref_kind == "mesh" and test_kind == "cloud":
        errs_test = errs
    elif ref_kind == "cloud" and test_kind == "mesh":
        errs_ref = errs
    else:
        errs_test = errs

    # ---- viewer artifacts ----------------------------------------------------
    evq.put(("progress", {"progress": 0.9}))
    evq.put(("log", {"level": "INFO", "msg": "writing viewer artifacts"}))
    t0 = time.time()

    if ref_kind == "mesh":
        rv, rn, rf = _mesh_arrays(ref)
    else:
        rv, rn, rf = _cloud_arrays(ref)
    if test_kind == "mesh":
        tv, tn, tf = _mesh_arrays(aligned)
    else:
        tv, tn, tf = _cloud_arrays(aligned)

    write_binary_ply(os.path.join(viewer_dir, "ref.ply"), rv, rn, rf)
    write_binary_ply(os.path.join(viewer_dir, "aligned.ply"), tv, tn, tf)

    view = {"ref": {"url": "viewer/ref.ply", "kind": ref_kind,
                    "n_points": int(len(rv)),
                    "n_faces": int(len(rf)) if rf is not None else 0,
                    "unit": "mm"},
            "aligned": {"url": "viewer/aligned.ply", "kind": test_kind,
                        "n_points": int(len(tv)),
                        "n_faces": int(len(tf)) if tf is not None else 0,
                        "unit": "mm"}}

    for side, arr in (("ref", errs_ref), ("test", errs_test)):
        if arr is None or len(arr) == 0:
            continue
        stats = _err_stats(arr)
        _write_f32(os.path.join(viewer_dir, f"errors_{side}.f32"), arr)
        _json_dump(stats, os.path.join(viewer_dir, f"errors_{side}.json"))
        view[f"errors_{side}"] = {"url": f"viewer/errors_{side}.f32",
                                  "stats_url": f"viewer/errors_{side}.json",
                                  **stats}

    headline_errs = errs if len(errs) else (errs_test if errs_test is not None else errs_ref)
    if headline_errs is not None and len(headline_errs):
        hist = _histogram(headline_errs)
        _json_dump(hist, os.path.join(viewer_dir, "hist.json"))
        view["hist"] = {"url": "viewer/hist.json"}

    # ---- summary -------------------------------------------------------------
    import subprocess
    try:
        commit = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=10
                                 ).stdout.strip()
    except Exception:
        commit = None

    summary_mae_csv = os.path.join(run_dir, "pair_surf_mae_um.csv")
    mae_row = {}
    if os.path.exists(summary_mae_csv):
        with open(summary_mae_csv) as f:
            lines = [l.strip() for l in f if l.strip()]
        keys = lines[0].split(",")
        vals = lines[1].split(",")
        mae_row = dict(zip(keys, vals))

    bundle = {}
    bundle_path = os.path.join(run_dir, "pair_metrics.json")
    if metrics_on and os.path.exists(bundle_path):
        with open(bundle_path) as f:
            bundle = json.load(f)

    summary = {
        "inputs": {
            "ref": {"name": os.path.basename(ref_path), "kind": ref_kind,
                    "n_points": int(len(rv)),
                    "n_faces": int(len(rf)) if rf is not None else 0},
            "test": {"name": os.path.basename(test_path), "kind": test_kind,
                     "n_points": int(len(tv)),
                     "n_faces": int(len(tf)) if tf is not None else 0},
        },
        "transform_4x4": [[round(float(x), 6) for x in row] for row in T],
        "headline": {
            "mae_um": round(float(mae_um), 4), "p95_um": round(float(p95_um), 4),
            "char_length_um": float(mae_row.get("char_length_um", "nan") or "nan"),
            "rel_mae_pct": float(mae_row.get("rel_mae_pct", "nan") or "nan"),
            "rel_p95_pct": float(mae_row.get("rel_p95_pct", "nan") or "nan"),
        },
        "t1_diagnostics": {k: mae_row.get(k, "") for k in
                           ("E_vertex_um", "E_area_um", "bias_um",
                            "risk_bound_um", "cv_area", "kappa_tail")},
        "metrics_bundle": bundle,
        "timings_s": {"load": round(t_load, 3),
                      "register": round(t_register, 3) if use_icp else None,
                      "measure": round(t_measure, 3),
                      "viewer": round(time.time() - t0, 3),
                      "total": round(time.time() - t_start, 3)},
        "git_commit": commit,
        "meshalign_version": getattr(ma, "__version__", None),
        "command": command,
        "view": view,
    }
    _json_dump(summary, os.path.join(run_dir, "viewer", "summary.json"))

    # guide-named metrics.json (flat metric numbers for cards/agents)
    metrics_out = dict(summary["headline"])
    metrics_out.update({k: v for k, v in bundle.items()})
    metrics_out["duration_s"] = summary["timings_s"]["total"]
    _json_dump(metrics_out, os.path.join(run_dir, "metrics.json"))

    if not keep_per_sample:
        # per-vertex dumps were only needed to build the viewer arrays
        for fn in os.listdir(run_dir):
            if fn.startswith(("pair_surf_errors_mm.", "pair_surf_dft_xyz_um.")):
                os.remove(os.path.join(run_dir, fn))
    evq.put(("progress", {"progress": 1.0}))
    evq.put(("log", {"level": "INFO",
                     "msg": f"done in {summary['timings_s']['total']:.1f} s "
                            f"(MAE {mae_um:.1f} um, p95 {p95_um:.1f} um)"}))
    return summary
