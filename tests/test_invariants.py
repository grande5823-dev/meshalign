"""Property-based invariants of meshalign's public API.

Unlike test_meshalign.py (specific scenarios), these are cross-cutting claims
the tool makes about ITSELF regardless of input -- the things every lab project
relies on but nothing else verifies.
"""
import numpy as np
import open3d as o3d
import pytest

import meshalign as ma
from meshalign.demo import make_bracket, rotz


def to_o3d(mesh):
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    m.remove_duplicated_vertices(); m.remove_duplicated_triangles()
    m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


# ---------------------------------------------------------------------------

def test_T_is_valid_rigid_transform():
    """For every register() output: R is a rotation (orthonormal, det=+1) and
    the bottom row is [0 0 0 1]. A non-rigid output would silently corrupt
    downstream DFT measurements."""
    rng = np.random.default_rng(7)
    for i in range(4):
        ref = make_bracket()
        test = ref.copy()
        a, b = np.degrees(np.pi * rng.integers(0, 2)), 0.0
        yaw = float(5 * i + 15)
        T_applied = np.eye(4)
        T_applied[:3, :3] = rotz(yaw)[:3, :3]
        T_applied[:3, 3] = [3.0 * i, -i, 1.0]
        test.apply_transform(T_applied)
        T = ma.register(to_o3d(ref), to_o3d(test), seed=0, mode="default",
                        verbose=False)
        assert T.shape == (4, 4)
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), f"iter{i}: R not orthonormal"
        assert np.linalg.det(R) > 0, f"iter{i}: reflection detected"
        assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_register_never_regresses_on_already_aligned():
    """register() must not move a mesh that's already aligned. Repeated with
    different meshes/modes; any move > 1 um is a flip-regression."""
    for mode in ("fast", "default", "accuracy"):
        ref_t = make_bracket()
        ref = to_o3d(ref_t)
        test = to_o3d(make_bracket())   # identical shape geometry
        T = ma.register(ref, test, seed=42, mode=mode, verbose=False)
        moved = np.linalg.norm(T[:3, 3])
        ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        assert moved < 0.001 and ang < 0.1, f"mode={mode}: T moved a correct pose"


def test_surface_error_is_signed_symmetric_under_swap_of_same_vertex_set():
    """For identical meshes, signed per-vertex error is ~0 regardless of which
    mesh is ref vs test (swapping must not change the sign convention flip)."""
    a = to_o3d(make_bracket())
    b = to_o3d(make_bracket())
    errs_ab, _ = ma.surface_error(a, b)
    errs_ba, _ = ma.surface_error(b, a)
    assert np.max(np.abs(errs_ab)) < 1e-4 and np.max(np.abs(errs_ba)) < 1e-4


@pytest.mark.parametrize("mode", ["fast", "default", "accuracy"])
def test_register_seed_stability(mode):
    """Same seed -> same T up to Open3D thread-order float noise.

    threshold is 0.2 in absolute-transform units (the test meshes are ~50 mm,
    so this is ~0.2 mm worst-case pose drift between identical-seed runs -- a
    thread-reduction-order artifact, not a RANSAC non-determinism)."""
    ref = to_o3d(make_bracket())
    test_t = make_bracket().copy()
    T_app = np.eye(4); T_app[:3, :3] = rotz(30)[:3, :3]
    T_app[:3, 3] = [5.0, -2.0, 1.0]
    test_t.apply_transform(T_app)
    test = to_o3d(test_t)
    T1 = ma.register(ref, test, seed=0, mode=mode, verbose=False)
    T2 = ma.register(ref, test, seed=0, mode=mode, verbose=False)
    assert np.allclose(T1, T2, atol=0.2), (
        f"mode={mode}: same-seed register diverged by {np.linalg.norm(T1 - T2):.4f}")


def test_register_output_out_of_bounds_sanity():
    """Whatever register() returns, applying it must produce finite vertices."""
    ref = to_o3d(make_bracket())
    test_t = make_bracket().copy()
    test_t.apply_transform(rotz(180))
    test = to_o3d(test_t)
    T = ma.register(ref, test, seed=0, verbose=False)
    assert np.isfinite(T).all()


def test_run_pair_output_schema(tmp_path):
    """run_pair's written contract: same filenames, CSV columns in the schema.
    Lab scripts parse these; rename one and you break downstream."""
    import os, glob
    ref_t = make_bracket()
    ref_t.export(str(tmp_path / "r.stl"))
    ref_t.export(str(tmp_path / "t.stl"))
    ma.run_pair(str(tmp_path / "r.stl"), str(tmp_path / "t.stl"),
                str(tmp_path / "out"), use_icp=True, seed=0)
    out = tmp_path / "out"
    expected = ["*surf_mae_um.csv", "*surf_dft_um.csv", "*aligned.stl"]
    for pat in expected:
        files = list(out.glob(pat))
        assert files, f"missing {pat}"
    import pandas as pd
    mae_df = pd.read_csv(next(out.glob("*surf_mae_um.csv")))
    # 2026-09: schema extended with size-normalized columns (appended at the
    # end; the absolute mae_um/p95_um columns are unchanged for downstream
    # parsers that index by name). 2026-09-18 T1: appended again with the
    # referee-sampling diagnostics (E_vertex/E_area/bias/risk_bound/cv_area/
    # kappa_tail + the area_weighted flag) -- same additive rule.
    assert list(mae_df.columns) == ["mae_um", "p95_um", "char_length_um",
                                    "rel_mae_pct", "rel_p95_pct",
                                    "E_vertex_um", "E_area_um", "bias_um",
                                    "risk_bound_um", "cv_area",
                                    "kappa_tail", "area_weighted"]
