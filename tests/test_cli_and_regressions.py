"""CLI / registry / batch / I-O tests plus regressions for review findings."""
import csv
import os
import shutil

import numpy as np
import open3d as o3d
import pytest

import meshalign as ma
from meshalign import commands
from meshalign.cli import build_parser, main
from meshalign.demo import make_bracket, write_demo
from meshalign.measure import vertices_correspond


def _o3d(tm):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
                                  o3d.utility.Vector3iVector(np.asarray(tm.faces)))
    m.compute_vertex_normals()
    return m


def _shuffled(m, seed=0):
    perm = np.random.default_rng(seed).permutation(len(m.vertices))
    inv = np.argsort(perm)
    s = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices)[perm]),
        o3d.utility.Vector3iVector(inv[np.asarray(m.triangles)]))
    s.compute_vertex_normals()
    return s


@pytest.fixture(scope="module")
def pair_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    ref, test, _ = write_demo(str(d))
    return ref, test


# -- registry / argparse ------------------------------------------------------

def test_every_command_has_a_parser_and_flags():
    p = build_parser()
    a = p.parse_args(["register", "r.stl", "t.stl", "--global"])
    assert a.path == "global" and a.write is True
    a = p.parse_args(["register", "r.stl", "t.stl", "--local", "--no-write"])
    assert a.path == "local" and a.write is False
    a = p.parse_args(["run", "r.stl", "t.stl", "--out", "o"])
    assert a.histogram is None and a.per_sample is None   # preset decides
    a = p.parse_args(["run", "r.stl", "t.stl", "--out", "o", "--no-histogram",
                      "--per-sample", "--relative", "--per-vertex-mae"])
    assert a.histogram is False and a.per_sample is True
    assert a.relative and a.per_vertex_mae
    a = p.parse_args(["batch", "--refs", "a", "--tests", "b", "--out", "o", "--no-icp"])
    assert a.icp is False and a.pattern == "*"


def test_validate_fills_defaults_and_reports_field_errors():
    out = commands.validate("chamfer", {"a": "x.stl", "b": "y.stl"})
    assert out["sample"] == 12000 and out["seed"] == 0
    with pytest.raises(commands.ParamError) as e:
        commands.validate("run", {"ref": "a", "test": "b", "mode": "turbo", "seed": -1})
    fields = {p["field"] for p in e.value.problems}
    assert {"mode", "seed"} <= fields
    with pytest.raises(commands.ParamError):
        commands.validate("ensemble", {"meshes": ["only-one.stl"]})
    with pytest.raises(commands.ParamError):
        commands.validate("chamfer", {"a": "x", "b": "y", "bogus": 1})


def test_web_schema_has_every_non_cli_only_param():
    s = commands.web_schema()
    assert set(s) == set(commands.COMMANDS)
    for name, c in commands.COMMANDS.items():
        shown = {p["name"] for p in s[name]["params"]}
        assert shown == {p["name"] for p in c["params"] if not p.get("cli_only")}


def test_preset_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ma.preset("bogus")
    assert ma.preset("fast", n_samples=123)["n_samples"] == 123


def test_all_exports_exist():
    for n in ma.__all__:
        assert hasattr(ma, n), n


# -- I/O ----------------------------------------------------------------------

def test_load_any_cloud_extensions(tmp_path):
    pts = np.random.default_rng(0).random((50, 3))
    for ext in (".xyz", ".asc", ".txt"):
        p = tmp_path / f"c{ext}"
        np.savetxt(p, pts)
        kind, obj = ma.load_any(str(p))
        assert kind == "cloud" and len(obj.points) == 50


def test_load_any_missing_and_empty(tmp_path):
    with pytest.raises(ValueError):
        ma.load_any(str(tmp_path / "nope.stl"))
    (tmp_path / "empty.stl").write_bytes(b"")
    with pytest.raises(ValueError):
        ma.load_any(str(tmp_path / "empty.stl"))


# -- regressions --------------------------------------------------------------

def test_chamfer_cli_accepts_point_clouds(tmp_path, capsys):
    pts = np.random.default_rng(1).random((500, 3))
    a, b = tmp_path / "a.xyz", tmp_path / "b.xyz"
    np.savetxt(a, pts)
    np.savetxt(b, pts + 0.01)
    assert main(["chamfer", str(a), str(b)]) == 0
    assert "chamfer:" in capsys.readouterr().out


def test_vertices_correspond_rejects_unrelated_same_count_meshes():
    m = _o3d(make_bracket())
    assert vertices_correspond(m, m)
    assert not vertices_correspond(m, _shuffled(m))
    assert not vertices_correspond(m, m.subdivide_midpoint(1))   # count differs


def test_measure_same_count_shuffled_mesh_is_not_garbage(tmp_path):
    m = _o3d(make_bracket())
    ref_p, test_p = str(tmp_path / "r.stl"), str(tmp_path / "t.stl")
    o3d.io.write_triangle_mesh(ref_p, m)
    o3d.io.write_triangle_mesh(test_p, _shuffled(m))
    mae, _ = ma.run_pair(ref_p, test_p, str(tmp_path / "o"), use_icp=False,
                         histogram=False, save_per_sample=False)
    assert mae < 5.0          # microns; vertex-to-vertex would be thousands


def test_vote3_inside_runs_on_open3d_mesh():
    from meshalign.metrics_bundle import _vote3_inside
    m = _o3d(make_bracket())
    c = np.asarray(m.get_center())[None, :]
    inside = _vote3_inside(m, np.vstack([c, c + 1000.0]))
    assert inside.shape == (2,) and not inside[1]


def test_dft_single_vertex_has_no_nan():
    rows = ma.compute_directional_dft_um(np.zeros((1, 3)))
    assert all(np.isfinite(r["std_dft_um"]) for r in rows)


# -- area-weighted statistics -------------------------------------------------

def test_area_weighted_matches_plain_on_uniform_weights():
    d = np.abs(np.random.default_rng(0).normal(size=1000))
    a = np.ones(1000)
    assert ma.area_weighted_mean(d, a) == pytest.approx(d.mean())
    assert ma.area_weighted_quantile(d, a, 0.95) == pytest.approx(
        np.percentile(d, 95, method="inverted_cdf"))
    assert ma.area_weighted_cte(d, a, 1.0) == pytest.approx(d.mean())
    with pytest.raises(ValueError):
        ma.area_weighted_cte(d, a, 0.0)


def test_area_weighted_ignores_unreferenced_vertices():
    d = np.array([1.0, 1.0, 100.0])
    a = np.array([1.0, 1.0, 0.0])
    assert ma.area_weighted_mean(d, a) == pytest.approx(1.0)


# -- end to end CLI (fast mode) ----------------------------------------------

def test_cli_register_run_measure(pair_paths, tmp_path):
    ref, test = pair_paths
    out = tmp_path / "al.stl"
    assert main(["register", ref, test, "-o", str(out), "--mode", "fast"]) == 0
    assert out.exists()
    assert main(["register", ref, test, "--no-write", "--mode", "fast"]) == 0
    od = tmp_path / "run"
    assert main(["run", ref, test, "--out", str(od), "--mode", "fast", "--relative"]) == 0
    names = os.listdir(od)
    assert any(n.endswith("_dft_summary.csv") for n in names)
    assert not any(n.endswith("_surf_histogram.png") for n in names)   # fast: preset off
    od2 = tmp_path / "meas"
    assert main(["measure", ref, test, "--out", str(od2), "--mode", "fast",
                 "--histogram"]) == 0
    assert any(n.endswith("_surf_histogram.png") for n in os.listdir(od2))


def test_cli_ensemble(pair_paths, tmp_path):
    ref, test = pair_paths
    csvp = tmp_path / "e.csv"
    assert main(["ensemble", ref, test, "--out", str(csvp)]) == 0
    assert csvp.exists() and "ensemble_dft_um" in csvp.read_text()


def test_batch_summary_has_header_and_is_job_count_independent(pair_paths, tmp_path):
    ref, test = pair_paths
    rd, td = tmp_path / "refs", tmp_path / "tests"
    rd.mkdir()
    td.mkdir()
    shutil.copy(ref, rd / "part.stl")
    shutil.copy(test, td / "part.stl")
    res = {}
    for jobs in (1, 2):
        od = tmp_path / f"out{jobs}"
        assert main(["batch", "--refs", str(rd), "--tests", str(td), "--out", str(od),
                     "--mode", "fast", "--jobs", str(jobs)]) == 0
        rows = list(csv.DictReader(open(od / "batch_summary.csv")))
        assert rows and rows[0]["ok"] == "True"
        res[jobs] = rows[0]["mae_um"]
    assert res[1] == res[2]


def test_batch_no_matches_errors(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError):
        commands.execute("batch", {"refs": str(tmp_path / "a"), "tests": str(tmp_path / "b"),
                                   "out": str(tmp_path / "o")})
