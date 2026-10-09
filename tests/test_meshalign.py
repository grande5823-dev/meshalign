"""meshalign test suite -- all synthetic, all deterministic (seeded)."""
import io
import json
import os
import subprocess
import sys

import numpy as np
import open3d as o3d
import trimesh
import pytest

import meshalign as ma
from meshalign import align as ma_align
from meshalign.demo import make_bracket, make_pair, rotz


def to_o3d(mesh):
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    m.remove_duplicated_vertices()
    m.remove_duplicated_triangles()
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


def trimesh_roundtrip(mesh):
    return mesh


@pytest.fixture(scope="module")
def dome_pair():
    return make_pair(seed=0)


# -----------------------------------------------------------------------------

def test_surface_error_zero_for_identical():
    ref = to_o3d(make_bracket())
    errs, delta = ma.surface_error(ref, ref)
    assert np.max(np.abs(errs)) < 1e-4   # BVH queries are float32; sub-micron agreement
    assert np.max(np.abs(delta)) < 1e-4


def test_chamfer_zero_self_nonzero_offset():
    a = np.asarray(to_o3d(make_bracket()).vertices)
    b = a.copy()
    b[:, 1] += 2.0  # 2 mm offset along one axis only
    d0 = ma.chamfer(a, a, sample=2000)
    d = ma.chamfer(a, b, sample=2000)
    assert d0 < 1e-9
    # smooth-surface sampling means the NN distance after shift is less than the
    # shift (interpolation) but well above zero
    assert 0.1 < d < 2.1


def test_register_identity_no_regression():
    """An already-correct pose must not move (identity candidate wins)."""
    ref_t = make_bracket()
    ref = to_o3d(ref_t)
    test = to_o3d(ref_t)
    T = ma.register(ref, test, seed=0, verbose=False)
    assert np.allclose(T, np.eye(4), atol=1e-3), f"T drifted from identity:\n{T}"


def test_register_recovers_known_yaw(dome_pair, tmp_path):
    """Global path: recover a 35deg yaw + 7mm translation from raw poses."""
    ref_t, test_t, T_applied = dome_pair
    ref, test = to_o3d(ref_t), to_o3d(test_t)
    T = ma.register(ref, test, n_samples=20000, seed=0, verbose=False)

    aligned = trimesh.Trimesh(np.asarray(test.vertices), np.asarray(test.triangles))
    aligned.apply_transform(T)
    # evaluate precise chamfer between aligned-test and ref
    d_aligned = ma.chamfer(
        np.asarray(aligned.sample(4000)), np.asarray(ref_t.sample(4000)), sample=4000)
    # sanity: the raw test was far off; the aligned one must be close (noise-level)
    d_raw = ma.chamfer(
        np.asarray(test_t.sample(4000)), np.asarray(ref_t.sample(4000)), sample=4000)
    assert d_aligned < 1.0, f"aligned chamfer {d_aligned:.3f}mm too large"
    assert d_aligned < d_raw * 0.25


def test_align_local_recovers_180_yaw():
    """Symmetry trap: a dome yawed 180deg in-plane must resolve via the 180
    candidate of align_local (identity-start ICP alone cannot leave its basin)."""
    ref_t = make_bracket()
    test_t = ref_t.copy()
    # recentre to bbox-centre, then yaw 180 about Z, keep bbox frame identical-ish
    v = np.asarray(test_t.vertices)
    c = (v.min(axis=0) + v.max(axis=0)) / 2.0
    T_yaw = np.eye(4)
    T_yaw[:3, :3] = rotz(180.0)[:3, :3]
    T_yaw[:3, 3] = c - rotz(180.0)[:3, :3] @ c
    test_t.apply_transform(T_yaw)

    ref, test = to_o3d(ref_t), to_o3d(test_t)
    src, dst = ma_align._icp_sample(ref, test, 20000, 0)
    scene = ma_align._test_scene(test)
    base = ma_align._mean_ref_to_test(scene, np.asarray(dst.points), np.eye(4))
    T, scores = ma_align.align_local(ref, test, src, dst, base,
                                     yaw_inits_deg=(0.0, 180.0),
                                     return_scores=True, verbose=False)
    # the 180deg candidate should beat identity-start
    assert scores[180.0] < scores[0.0]
    assert scores[180.0] < scores[None] * 0.8


def test_measure_directional_dft_keys():
    ref = to_o3d(make_bracket())
    errs, delta = ma.surface_error(ref, to_o3d(make_bracket()))
    table = ma.compute_directional_dft_um(delta)
    assert {r["direction"] for r in table} == {"X", "Y", "Z"}
    assert all(abs(r["mean_abs_dft_um"]) < 1e-3 for r in table)


def test_run_pair_deterministic(dome_pair, tmp_path):
    """Same inputs, two runs -> EXACTLY the same outputs.

    Open3D's ICP/RANSAC are multi-threaded; under default threading the
    float-reduction order varies run-to-run (VALIDATION.md limitation #5 --
    the ~2-5 um pose wobble, amplified by the vertex-to-vertex shortcut on
    this same-vertex-count pair). With OMP_NUM_THREADS=1 the pipeline is
    bit-exact, so this test pins threads and asserts EQUALITY: the
    determinism CONTRACT is exact reproduction, not reproduction within
    noise. The wobble envelope under default threading is covered separately
    by test_register_seed_stability (C3).
    """
    ref_path = tmp_path / "ref.stl"
    test_path = tmp_path / "test.stl"
    ref_t, test_t, _ = dome_pair
    ref_t.export(str(ref_path))
    test_t.export(str(test_path))

    snippet = (
        "import sys, json\n"
        "import meshalign as ma\n"
        f"out = ma.run_pair({str(ref_path)!r}, {str(test_path)!r}, sys.argv[1],\n"
        "                   use_icp=True, seed=0)\n"
        "print(json.dumps(out))\n"
    )
    env = dict(os.environ, OMP_NUM_THREADS="1")
    outs = []
    for i in (1, 2):
        r = subprocess.run([sys.executable, "-c", snippet,
                            str(tmp_path / f"o{i}")],
                           capture_output=True, text=True, env=env, check=True)
        outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    (mae1, p95_1), (mae2, p95_2) = outs
    assert mae1 == mae2
    assert p95_1 == p95_2
    # CSV outputs should agree EXACTLY (bit-exact pipeline).
    import csv

    def read_rows(path):
        with open(path, newline="") as f:
            return list(csv.reader(f))

    for f in ("surf_mae_um.csv", "surf_dft_um.csv"):
        p = next((tmp_path / "o1").glob(f"*{f}"))
        q = next((tmp_path / "o2").glob(f"*{f}"))
        assert read_rows(p) == read_rows(q), f"CSV differs: {f}"


def test_run_pair_fixes_pose_and_scores_shift(dome_pair, tmp_path):
    """use_icp=True on a far-off mesh must both align and lower the DFT vs raw."""
    ref_path = tmp_path / "ref.stl"
    test_path = tmp_path / "test.stl"
    ref_t, test_t, _ = dome_pair
    ref_t.export(str(ref_path))
    test_t.export(str(test_path))
    mae_raw, _ = ma.run_pair(str(ref_path), str(test_path), str(tmp_path / "raw"),
                             use_icp=False, seed=0)
    mae_icp, _ = ma.run_pair(str(ref_path), str(test_path), str(tmp_path / "icp"),
                             use_icp=True, seed=0)
    assert mae_icp < mae_raw * 0.5
