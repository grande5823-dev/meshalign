"""Single source of truth for every meshalign command.

The argparse CLI (``cli.py``), the webapp's HTTP API / ``GET /api/commands``
and the browser form are ALL generated from ``COMMANDS``, and every surface
executes through ``execute()``. A flag added here therefore appears in the
CLI, the API schema and the UI form at once.

Param spec keys:
  name      python/JSON name (CLI flag is ``--name`` with ``_`` -> ``-``)
  type      dataset | datasets | group | bool | int | float | enum | str | path
  default   default value (None = unset / preset decides)
  help      one-line description (CLI help, API schema, UI tooltip)
  choices   for enum
  required  must be supplied
  positional  CLI positional argument (datasets / dataset)
  advanced  UI hides it under "Advanced"
  flags     extra CLI aliases mapping to a value: {"--global": "global"}
  cli_only  not shown in the web form (e.g. output paths the server controls)
"""
from __future__ import annotations

import copy
import os

from .config import DEFAULT_SEED

MODES = ["fast", "default", "accuracy"]


def _p(name, type, default=None, help="", **kw):
    d = dict(name=name, type=type, default=default, help=help)
    d.update(kw)
    return d


_SEED = _p("seed", "int", DEFAULT_SEED, "RNG seed (results are deterministic per seed)",
           advanced=True, min=0, max=2**31 - 1)
_MODE = _p("mode", "enum", "default",
           "Solver profile: fast (speed), default (tuned), accuracy (adds Go-ICP audit)",
           choices=MODES)
_REF = _p("ref", "dataset", None, "Reference mesh/point cloud", required=True, positional=True)
_TEST = _p("test", "dataset", None, "Test mesh/point cloud", required=True, positional=True)

_PIPELINE_OPTS = [
    _p("histogram", "bool", None, "Write the error histogram PNG (default: per mode)", advanced=True),
    _p("per_sample", "bool", None, "Write per-vertex error dumps (default: per mode)", advanced=True),
    _MODE,
    _p("metrics", "bool", False, "Also compute the Chamfer/Hausdorff/normal/voxel bundle",
       web_default=True),
    _p("relative", "bool", False, "Also write the size-normalized DFT summary"),
    _p("per_vertex_mae", "bool", False,
       "Report per-vertex MAE/p95 instead of the default area-weighted statistics",
       advanced=True),
    _SEED,
]

COMMANDS = {
    "run": dict(
        title="Run", summary="Register the test onto the reference, then measure surface error.",
        viewer=True,
        params=[_REF, _TEST,
                _p("out", "path", None, "Output directory", required=True, cli_only=True)]
               + copy.deepcopy(_PIPELINE_OPTS)),
    "measure": dict(
        title="Measure", summary="Measure surface error as-is (no registration).",
        viewer=True,
        params=[_REF, _TEST,
                _p("out", "path", None, "Output directory", required=True, cli_only=True)]
               + copy.deepcopy(_PIPELINE_OPTS)),
    "register": dict(
        title="Register", summary="Rigidly register TEST onto REF and save the aligned mesh.",
        viewer=True,
        params=[_REF, _TEST,
                _p("out", "path", None, "Output path for the aligned mesh (stl/ply)",
                   cli_only=True, short="-o"),
                _p("path", "enum", "auto",
                   "auto routes by identity fit; global = FPFH+RANSAC search; "
                   "local = bounded refine (input already close)",
                   choices=["auto", "global", "local"],
                   flags={"--global": "global", "--local": "local"}),
                _p("write", "bool", True, "Save the aligned mesh"),
                _MODE, copy.deepcopy(_SEED)]),
    "chamfer": dict(
        title="Chamfer", summary="Symmetric Chamfer distance between two meshes/point sets.",
        viewer=False,
        params=[_p("a", "dataset", None, "First mesh/point cloud", required=True, positional=True),
                _p("b", "dataset", None, "Second mesh/point cloud", required=True, positional=True),
                _p("sample", "int", 12000, "Points sampled per side", min=100, max=1_000_000),
                _p("seed", "int", 0, "RNG seed", advanced=True, min=0, max=2**31 - 1)]),
    "ensemble": dict(
        title="Ensemble", summary="No-reference consistency of 2+ meshes of the same object.",
        viewer=False,
        params=[_p("meshes", "datasets", None, "Two or more candidate meshes",
                   required=True, positional=True, min_items=2),
                _p("hub_index", "int", None, "Index of the hub mesh (default: Chamfer medoid)",
                   advanced=True, min=0),
                _p("out", "path", None, "Optional CSV output path", cli_only=True),
                _p("seed", "int", 0, "RNG seed", advanced=True, min=0, max=2**31 - 1)]),
    "batch": dict(
        title="Batch", summary="Parallel align + measure over filename-matched ref/test folders.",
        viewer=False,
        params=[_p("refs", "group", None, "Reference folder", required=True),
                _p("tests", "group", None, "Test folder", required=True),
                _p("out", "path", None, "Output directory", required=True, cli_only=True),
                _p("jobs", "int", None, "Worker processes (default min(32, cpus))",
                   advanced=True, min=1, max=64),
                _MODE,
                _p("icp", "bool", True, "Register before measuring (--no-icp measures as-is)"),
                _p("absolute", "bool", False, "Plain run_pair instead of size-normalized relative",
                   advanced=True),
                _p("pattern", "str", "*", "Filename glob within both folders", advanced=True),
                _p("seed_base", "int", DEFAULT_SEED, "Base seed (pair i uses base+i)",
                   advanced=True, min=0, max=2**31 - 1)]),
}


def spec(name):
    if name not in COMMANDS:
        raise ValueError(f"unknown command {name!r} (choose from {tuple(COMMANDS)})")
    return COMMANDS[name]


def web_schema():
    """(see below) defaults shown in the UI use ``web_default`` when present."""
    return _web_schema()


def _web_schema():
    """JSON-serializable registry for ``GET /api/commands`` (UI form source)."""
    out = {}
    for name, c in COMMANDS.items():
        params = []
        for p in c["params"]:
            if p.get("cli_only"):
                continue
            q = {k: v for k, v in p.items() if k not in ("short", "web_default")}
            if "web_default" in p:
                q["default"] = p["web_default"]
            params.append(q)
        out[name] = dict(title=c["title"], summary=c["summary"], viewer=c["viewer"],
                         params=params)
    return out


def validate(name, params, strict=True, web=False):
    """Fill defaults and type-check ``params`` against the registry.

    Returns the normalized dict. Raises ``ParamError`` with a list of
    ``{"field", "message"}`` problems (API maps it to a 422).
    """
    c = spec(name)
    known = {p["name"]: p for p in c["params"]}
    problems = []
    out = {}
    for k in params:
        if k not in known and strict:
            problems.append({"field": k, "message": "unknown parameter"})
    for p in c["params"]:
        k = p["name"]
        v = params.get(k, p.get("web_default", p["default"]) if web else p["default"])
        if v is None or v == "":
            if p.get("required") and not p.get("cli_only"):
                problems.append({"field": k, "message": "required"})
            out[k] = None if v == "" else v
            continue
        t = p["type"]
        try:
            if t == "bool":
                if not isinstance(v, bool):
                    raise ValueError("must be true/false")
            elif t in ("int", "float"):
                if isinstance(v, bool):
                    raise ValueError(f"must be a number")
                v = int(v) if t == "int" else float(v)
                if "min" in p and v < p["min"]:
                    raise ValueError(f"must be >= {p['min']}")
                if "max" in p and v > p["max"]:
                    raise ValueError(f"must be <= {p['max']}")
            elif t == "enum":
                if v not in p["choices"]:
                    raise ValueError(f"must be one of {p['choices']}")
            elif t == "datasets":
                if not isinstance(v, (list, tuple)):
                    raise ValueError("must be a list")
                if len(v) < p.get("min_items", 1):
                    raise ValueError(f"need at least {p.get('min_items', 1)} items")
                v = [str(x) for x in v]
            else:  # dataset | group | str | path
                v = str(v)
        except (TypeError, ValueError) as e:
            problems.append({"field": k, "message": str(e)})
            continue
        out[k] = v
    if problems:
        raise ParamError(problems)
    return out


class ParamError(ValueError):
    def __init__(self, problems):
        self.problems = problems
        super().__init__("; ".join(f"{p['field']}: {p['message']}" for p in problems))


# ---------------------------------------------------------------------------
# Execution (shared by the CLI and the webapp job runner)
# ---------------------------------------------------------------------------

def _register(P):
    import copy as _copy
    import numpy as np
    import open3d as o3d
    from . import align
    from .io import load_any

    ref_kind, ref = load_any(P["ref"])
    test_kind, test = load_any(P["test"])
    seed, mode = P["seed"], P["mode"]
    if P["path"] == "local":
        src, dst = align._icp_sample(ref, test, 25000, seed)
        if test_kind == "mesh":
            scene = align._test_scene(test)
            base = align._mean_ref_to_test(scene, np.asarray(dst.points), np.eye(4))
            T = align.align_local(ref, test, src, dst, base)
        else:
            score_fn, base = align._make_referee(ref, test, seed=seed)
            T = align.align_local(ref, test, src, dst, base, score_fn=score_fn)
    elif P["path"] == "global":
        src, dst = align._icp_sample(ref, test, 25000, seed)
        src_pts = np.asarray(src.points)
        ref_pts = np.asarray(dst.points)
        if test_kind == "mesh":
            scene = align._test_scene(test)
            base = align._mean_ref_to_test(scene, ref_pts, np.eye(4))
            T = align.align_global(src, dst, src_pts, ref_pts, base, scene)
        else:
            score_fn, base = align._make_referee(ref, test, seed=seed)
            T = align.align_global(src, dst, src_pts, ref_pts, base, score_fn=score_fn)
    else:
        T = align.register(ref, test, seed=seed, mode=mode)

    result = {"transform_4x4": [[float(x) for x in row] for row in T],
              "test_kind": test_kind, "aligned_path": None}
    if P["write"]:
        aligned = _copy.deepcopy(test)
        aligned.transform(T)
        out = P.get("out")
        if not out:
            stem = os.path.splitext(os.path.basename(P["test"]))[0]
            out = f"{stem}_aligned.ply" if test_kind == "cloud" else f"{stem}_aligned.stl"
        if test_kind == "cloud":
            if not aligned.has_normals():
                aligned.estimate_normals()
            ok = o3d.io.write_point_cloud(out, aligned)
        else:
            aligned.compute_vertex_normals()
            ok = o3d.io.write_triangle_mesh(out, aligned)
        if not ok:
            raise ValueError(f"could not write {out}")
        print(f"Saved: {out}")
        result["aligned_path"] = out
    return result


def _pipeline(P, use_icp):
    from .pipeline import run_pair, run_pair_relative
    fn = run_pair_relative if P["relative"] else run_pair
    out = fn(P["ref"], P["test"], P["out"], use_icp=use_icp, seed=P["seed"],
             histogram=P["histogram"], save_per_sample=P["per_sample"],
             mode=P["mode"], metrics=P["metrics"],
             area_weighted=False if P["per_vertex_mae"] else None)
    res = {"mae_um": out[0], "p95_um": out[1], "out_dir": P["out"]}
    if P["relative"]:
        res.update(char_length_um=out[2], rel_mae_pct=out[3], rel_p95_pct=out[4])
    return res


def _chamfer(P):
    from .io import load_any
    from .measure import chamfer
    a = load_any(P["a"])[1]
    b = load_any(P["b"])[1]
    cd = chamfer(a, b, sample=P["sample"], seed=P["seed"])
    print(f"chamfer: {cd:.6f}")
    return {"chamfer": float(cd)}


def _ensemble(P):
    from .metrics import ensemble_consistency
    df = ensemble_consistency(P["meshes"], hub_index=P["hub_index"], seed=P["seed"])
    import pandas as pd
    with pd.option_context("display.width", 160):
        print(df.to_string(index=False))
    if P.get("out"):
        df.to_csv(P["out"], index=False)
        print(f"\nSaved to {P['out']}")
    return {"rows": df.to_dict(orient="records"),
            "hub": df.attrs.get("hub"),
            "median_um": df.attrs.get("ensemble_median_um"),
            "mad_um": df.attrs.get("ensemble_mad_um"),
            "csv": P.get("out")}


def _batch(P, progress=None):
    from .batch import _match_pairs, run_batch
    pairs = _match_pairs(P["refs"], P["tests"], P["pattern"])
    if not pairs:
        raise ValueError(f"no matched pairs between {P['refs']} and {P['tests']} "
                         f"(pattern {P['pattern']!r})")
    results = run_batch(pairs, P["out"], jobs=P["jobs"], use_icp=P["icp"],
                        mode=P["mode"], relative=not P["absolute"],
                        seed_base=P["seed_base"],
                        on_result=(lambda i, n, r: progress((i + 1) / n,
                                   f"{i + 1}/{n} {os.path.basename(r['test'])}"))
                        if progress else None)
    return {"results": results, "n_ok": sum(1 for r in results if r.get("ok")),
            "n_pairs": len(results), "summary_csv": os.path.join(P["out"], "batch_summary.csv")}


def execute(name, params, progress=None):
    """Run command ``name`` with validated-or-raw ``params``; returns a result dict.

    ``progress(fraction, message)`` is optional (used by the webapp for batch).
    """
    P = validate(name, params)
    if name == "run":
        return _pipeline(P, use_icp=True)
    if name == "measure":
        return _pipeline(P, use_icp=False)
    if name == "register":
        return _register(P)
    if name == "chamfer":
        return _chamfer(P)
    if name == "ensemble":
        return _ensemble(P)
    if name == "batch":
        return _batch(P, progress)
    raise ValueError(name)
