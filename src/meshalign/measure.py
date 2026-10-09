"""Surface-distance measurement + directional DFT + Chamfer.

Pure distance computation -- no alignment, no I/O.
"""
import weakref

import numpy as np
import open3d as o3d

from .config import UM_PER_MM


# --------------------------------------------------
# T1: area-uniform statistics (docs/theory/t1-sampling-bias.md).
#
# The per-vertex mean over a reference mesh's raw vertex list is a biased
# estimator of the area-uniform surface integral E whenever the tessellation
# is non-uniform: the exact signed bias is -Cov_V(a_v, g_v) / mean(a_v)
# (Theorem T1a-2), bounded by CV(a) * sigma_V(g) (T1c-1) -- and EQUAL to that
# bound on single-defect structure (T1c-3). The fix is NOT a resample: it is
# the deterministic lumped re-weighting
#     G_hat = sum_v a_v * |d_v| / sum_v a_v
# computed from the SAME per-vertex distances the referee already produces
# (change the weights, not the queries). Zero RNG, zero extra BVH queries,
# byte-identical candidate pools (the resampling route re-rolls the RANSAC
# lottery -- 50k-sample MC noise 0.22 um exceeds the documented 0.2% tie
# margins; see t1-sampling-bias.md section 6.4).
# --------------------------------------------------

# Lumped areas cached per mesh OBJECT (weak-keyed: dies with the mesh, so the
# cache can never serve a stale array for a recycled id). a_v is a property
# of the tessellation only -- one O(F+V) computation per mesh lifetime.
_AREA_CACHE = weakref.WeakKeyDictionary()


def lumped_areas(mesh):
    """Lumped (barycentric) vertex areas a_v = (1/3) * sum_{f ∋ v} area(f).

    Float64, cached per mesh object (weak-keyed -- the entry dies with the
    mesh, so a recycled id can never be served a stale array). Lemma 1.5
    (partition of unity): sum_v a_v == sum_f A_f exactly, up to float64
    roundoff; verified on cache fill and REPORTED (never crashed on) if a
    mesh violates it by more than 1e-9 relative (non-manifold /
    duplicate-vertex surprises; Open3D's STL loader was measured to dedup
    vertices on load, so corpus parts hold). Guards: degenerate faces
    (A_f = 0) contribute zero; unreferenced vertices get a_v = 0 -- they
    carry no surface and are EXCLUDED from every area-weighted statistic
    (today's per-vertex mean still averages their distances in; excluding
    is strictly more correct -- t1-sampling-bias.md section 5.3).

    Deterministic: pure arithmetic over the vertex/triangle arrays, no RNG.
    """
    cached = _AREA_CACHE.get(mesh)
    if cached is not None:
        return cached
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles, dtype=np.int64)
    if len(v) == 0 or len(f) == 0:
        a = np.zeros(len(v), dtype=np.float64)
        _AREA_CACHE[mesh] = a
        return a
    tri = v[f]                                              # (m, 3, 3)
    # float64 cross products (the BVH query path is float32; the WEIGHTS are
    # accumulated in float64 -- t1-sampling-bias.md section 7.1 item 4)
    Af = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                       tri[:, 2] - tri[:, 0]), axis=1)
    a = np.bincount(f.ravel(), weights=np.repeat(Af / 3.0, 3),
                    minlength=len(v)).astype(np.float64)
    # partition-of-unity check (Lemma 1.5): report, don't crash
    sum_a, sum_Af = float(a.sum()), float(Af.sum())
    if sum_Af > 0.0 and abs(sum_a - sum_Af) > 1e-9 * sum_Af:
        print(f"[meshalign] WARNING: lumped-area partition of unity violated "
              f"(sum a_v = {sum_a!r}, sum A_f = {sum_Af!r}, rel err "
              f"{abs(sum_a - sum_Af) / sum_Af:.2e}) -- non-manifold or "
              f"duplicate-vertex geometry; area-weighted statistics may be off.")
    _AREA_CACHE[mesh] = a
    return a


def _kept(d, a):
    """Filter to referenced vertices (a_v > 0); returns (d, a) float64 views."""
    d = np.abs(np.asarray(d, dtype=np.float64))
    a = np.asarray(a, dtype=np.float64)
    keep = a > 0.0
    if not keep.all():
        return d[keep], a[keep]
    return d, a


def area_weighted_mean(errors, areas):
    """G_hat = sum_v a_v * |d_v| / sum_v a_v -- the area-uniform statistic.

    The deterministic lumped re-weighting of the SAME per-vertex distances
    (change the weights, not the queries). Unreferenced vertices (a_v = 0)
    are excluded. Returns a float in the input units.
    """
    d, a = _kept(errors, areas)
    if len(d) == 0:
        return 0.0
    return float((a * d).sum() / a.sum())


def area_weighted_quantile(errors, areas, q):
    """Type-1 (inverse-CDF) quantile of |errors| under lumped-area weights.

    The area-measure analogue of ``np.percentile(|d|, 100*q)``: the smallest
    |d_v| whose cumulative area weight reaches q of the total. Deterministic
    (stable sort). Unreferenced vertices excluded.
    """
    d, a = _kept(errors, areas)
    if len(d) == 0:
        return 0.0
    order = np.argsort(d, kind="stable")
    d, a = d[order], a[order]
    cum = np.cumsum(a)
    idx = int(np.searchsorted(cum, q * cum[-1], side="left"))
    return float(d[min(idx, len(d) - 1)])


def area_weighted_cte(errors, areas, phi):
    """Area-conditional tail expectation CTE_phi of |errors| (t1 doc section 6.5).

    The mean of |d| over the worst ``phi`` FRACTION OF AREA (not of vertices):
    CTE = (1/phi) * integral_0^phi g_down(s) ds. For a localized signal on
    area fraction varphi_a <= phi it returns the signal AMPLITUDE regardless
    of both varphi_a and the tessellation -- the principled replacement for
    the per-vertex p95 tail term (p95 is blind to features below 5% of the
    VERTEX mass and its per-vertex quantile mixes conventions; section 6.6).
    Computed from per-vertex data by ordering on |d| and walking cumulative
    area (partial weight on the straddling vertex) -- deterministic, no
    resampling. ``phi`` in (0, 1]; unreferenced vertices excluded.
    """
    if not (0.0 < phi <= 1.0):
        raise ValueError(f"phi must be in (0, 1], got {phi!r}")
    d, a = _kept(errors, areas)
    if len(d) == 0:
        return 0.0
    order = np.argsort(-d, kind="stable")
    d, a = d[order], a[order]
    cum = np.cumsum(a)
    target = phi * cum[-1]
    idx = int(np.searchsorted(cum, target, side="left"))
    w = a[:idx]
    dv = d[:idx]
    rem = target - (float(cum[idx - 1]) if idx > 0 else 0.0)
    if idx < len(a) and rem > 0.0:
        w = np.append(w, rem)
        dv = np.append(dv, d[idx])
    return float(np.dot(w, dv) / target)


def area_weighted_stats(errors, areas, area_fraction=0.01):
    """The per-run T1 diagnostics (t1-sampling-bias.md section 5.3).

    All quantities are O(V) from data the referee already holds -- no extra
    BVH queries. ``errors`` are the per-vertex signed errors (mesh-mesh
    measurement path); ``areas`` the matching lumped areas. Both statistics
    EXCLUDE unreferenced vertices so ``bias`` is the pure weighting effect
    (the reported per-vertex MAE in ``run_pair`` keeps including them for
    backward comparability).

    Returns a dict:
      E_vertex    per-vertex mean of |d| (today's referee statistic)
      E_area      G_hat, the area-weighted mean (the T1 fix)
      bias        E_vertex - E_area == -Cov_V(a, g)/mean(a)  (exact, T1a-2)
      risk_bound  CV(a) * sigma_V(d)  (the T1c-1 worst case; EQUALS bias on
                  single-defect structure, T1c-3)
      cv_area     std(a)/mean(a) -- tessellation non-uniformity
      kappa_tail  local refinement factor of the worst-``area_fraction``
                  tail: (tail vertex fraction) / (tail area fraction) --
                  the section-6 amplification diagnostic
    """
    d, a = _kept(errors, areas)
    if len(d) == 0:
        return dict(E_vertex=0.0, E_area=0.0, bias=0.0, risk_bound=0.0,
                    cv_area=0.0, kappa_tail=1.0)
    E_vertex = float(d.mean())
    E_area = float((a * d).sum() / a.sum())
    risk_bound = float(a.std() * d.std() / a.mean())
    cv_area = float(a.std() / a.mean())
    # tail amplification: worst area-fraction vertices' refinement factor
    order = np.argsort(-d, kind="stable")
    at, dt = a[order], d[order]
    cum = np.cumsum(at)
    target = float(np.clip(area_fraction, 1e-6, 1.0) * cum[-1])
    k = max(1, int(np.searchsorted(cum, target, side="left")) + 1)
    k = min(k, len(at))
    kappa_tail = float((k / len(at)) / (at[:k].sum() / at.sum()))
    return dict(E_vertex=E_vertex, E_area=E_area,
                bias=E_vertex - E_area, risk_bound=risk_bound,
                cv_area=cv_area, kappa_tail=kappa_tail)




def _signed_from_delta(delta_xyz, ref_normals):
    dists = np.linalg.norm(delta_xyz, axis=1)
    signs = np.sign(np.sum(delta_xyz * ref_normals, axis=1))
    signs[signs == 0.0] = 1.0
    return dists * signs


def vertices_correspond(ref_mesh, test_mesh, n_check=2000, ratio=2.0):
    """True when ``test_mesh`` vertex i plausibly IS ref vertex i.

    Compares, on a deterministic subsample of vertices, the vertex-to-vertex
    distance with the exact nearest-surface distance. For a vertex-ordered copy
    (with real defects) the two agree to within a small factor; for two
    unrelated meshes that merely share a vertex count the vertex-to-vertex
    distance is far larger. Requires equal vertex counts (else False).
    """
    rv = np.asarray(ref_mesh.vertices)
    tv = np.asarray(test_mesh.vertices)
    if len(rv) != len(tv) or len(rv) == 0:
        return False
    idx = np.arange(len(rv))
    if len(idx) > n_check:
        idx = idx[np.linspace(0, len(idx) - 1, n_check).astype(np.int64)]
    d_vv = np.linalg.norm(tv[idx] - rv[idx], axis=1)
    d_ns = np.linalg.norm(_nearest_delta_points(rv[idx], test_mesh), axis=1)
    span = float(np.ptp(rv, axis=0).max()) or 1.0
    return bool(np.median(d_vv) <= ratio * np.median(d_ns) + 1e-4 * span)


def _nearest_delta_points(points, test_mesh):
    """Vectors from each point to the nearest point on ``test_mesh``."""
    test_t = o3d.t.geometry.TriangleMesh.from_legacy(test_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(test_t)
    pts = np.asarray(points, dtype=np.float64)
    result = scene.compute_closest_points(
        o3d.core.Tensor(np.ascontiguousarray(pts, dtype=np.float32)))
    return result["points"].numpy().astype(np.float64) - pts


def _nearest_surface_delta(ref_mesh, test_mesh):
    """Exact nearest-surface displacement: for every ref vertex, the vector to the
    nearest point on the test mesh surface, via Open3D ``RaycastingScene`` --
    BVH-accelerated exact point-to-triangle query (~16x faster than
    trimesh.proximity.closest_point, sub-micron agreement). Deterministic across
    processes (no parallel reduction order), unlike KD-tree distance queries.
    """
    print("[open3d_raycast] Computing exact point-to-triangle distances (open3d RaycastingScene)...")
    test_t = o3d.t.geometry.TriangleMesh.from_legacy(test_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(test_t)
    ref_vertices = np.asarray(ref_mesh.vertices)
    query = o3d.core.Tensor(ref_vertices.astype(np.float32))
    result = scene.compute_closest_points(query)
    nearest = result["points"].numpy().astype(np.float64)
    return nearest - ref_vertices


def compute_vertex_error_and_delta_nearest(ref_mesh, test_mesh):
    """Signed nearest-surface distance + XYZ displacement for every ref vertex.

    Measures FROM every reference vertex TO the nearest point on the test mesh
    surface, then signs via dot product with the ref vertex normal (positive =
    test surface outside the ref surface along its normal).

    Returns ``(signed_errors, delta_xyz)`` in the meshes' units.
    """
    delta_xyz = _nearest_surface_delta(ref_mesh, test_mesh)
    ref_normals = np.asarray(ref_mesh.vertex_normals)
    signed_errors = _signed_from_delta(delta_xyz, ref_normals)
    return signed_errors, delta_xyz


# Friendly alias -- the primary public name.
surface_error = compute_vertex_error_and_delta_nearest


def point_error_signed(ref_mesh, points, transform=None):
    """Signed nearest-surface distance for every point in a cloud, vs a mesh.

    The cloud-side counterpart of ``compute_vertex_error_and_delta_nearest``:
    measures FROM every cloud point TO the nearest point on the reference mesh
    surface (exact point-to-triangle via the RaycastingScene), then signs via
    the dot product of the point's displacement with the reference FACE normal
    at the nearest triangle. SIGN CONVENTION (matches mesh-mesh, where
    positive = test surface outside the ref surface): **positive = the point
    sits OUTSIDE the reference surface along its normal** (a bulged-out
    region), negative = the point is inside it (a dent).

    ``points`` is an (N, 3) array or an Open3D PointCloud. ``transform`` is an
    optional 4x4 rigid pose applied to the points before measuring (the
    reference stays fixed) -- scoring candidate poses this way avoids rebuilding
    the BVH per candidate.

    Returns ``(signed_errors, delta_xyz)`` in the input units, where ``delta_xyz``
    is the per-point deviation field ``point - nearest_surface_point`` (the
    displacement FROM the surface TO the point, so its sign agrees with
    ``signed_errors`` and the directional DFT reads as a deviation map).
    Deterministic: same BVH machinery as the mesh-mesh referee.
    """
    print("[open3d_raycast] Computing exact point-to-triangle distances (open3d RaycastingScene)...")
    pts = np.asarray(points.points if hasattr(points, "points") else points, dtype=np.float64)
    if transform is not None:
        R = np.asarray(transform)[:3, :3]
        t = np.asarray(transform)[:3, 3]
        pts = pts @ R.T + t
    ref_t = o3d.t.geometry.TriangleMesh.from_legacy(ref_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(ref_t)
    # face normals for signing (unit, from the ref triangles)
    v = np.asarray(ref_mesh.vertices)
    f = np.asarray(ref_mesh.triangles)
    tri = v[f]                                              # (T, 3, 3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
    query = o3d.core.Tensor(np.ascontiguousarray(pts, dtype=np.float32))
    result = scene.compute_closest_points(query)
    nearest = result["points"].numpy().astype(np.float64)
    prim = result["primitive_ids"].numpy().astype(np.int64)
    delta_xyz = pts - nearest          # surface -> point deviation field
    valid = prim >= 0
    signs = np.ones(len(pts))
    signs[valid] = np.sign(np.sum(delta_xyz[valid] * fn[prim[valid]], axis=1))
    signs[signs == 0.0] = 1.0
    signed_errors = np.linalg.norm(delta_xyz, axis=1) * signs
    return signed_errors, delta_xyz


def cloud_cloud_error(a, b, sample=100_000, seed=0):
    """Symmetric unsigned NN distance between two clouds (cloud-vs-cloud referee).

    mean over BOTH directions of nearest-neighbor distances, subsampled to
    ``sample`` per side (seeded, deterministic). Uses ``scipy.spatial.cKDTree``
    with the default single-worker query -- the parallel KD-tree paths are the
    nondeterministic ones (see align._test_scene); single-worker queries are
    deterministic across processes.

    ``a``/``b``: (N, 3) arrays or Open3D PointClouds. No sign -- clouds carry no
    consistent orientation. Returns ``(mean_dist, a_to_b, b_to_a)``.
    """
    from scipy.spatial import cKDTree

    pa = np.asarray(a.points if hasattr(a, "points") else a, dtype=np.float64)
    pb = np.asarray(b.points if hasattr(b, "points") else b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(pa) > sample:
        pa = pa[rng.choice(len(pa), sample, replace=False)]
    if len(pb) > sample:
        pb = pb[rng.choice(len(pb), sample, replace=False)]
    ta, tb = cKDTree(pa), cKDTree(pb)
    d_ab, _ = ta.query(pb, k=1)     # b points -> nearest in a
    d_ba, _ = tb.query(pa, k=1)     # a points -> nearest in b
    return float(0.5 * (d_ab.mean() + d_ba.mean())), float(d_ab.mean()), float(d_ba.mean())


def cloud_point_error(test_cloud, ref_cloud, delta=False):
    """Per-point unsigned NN deviation of a test cloud against a reference cloud.

    The cloud-vs-cloud measurement counterpart of ``surface_error``: for every
    test point, the distance to the nearest reference point (no sign -- clouds
    carry no consistent orientation; no surface -- exact point-to-triangle is
    impossible). Single-worker cKDTree, deterministic.

    Returns ``(errors, delta_xyz)`` where ``delta_xyz`` (when ``delta=True``) is
    ``nearest_ref_point - test_point`` per test point, else ``None``.
    """
    from scipy.spatial import cKDTree

    tp = np.asarray(test_cloud.points if hasattr(test_cloud, "points") else test_cloud, dtype=np.float64)
    rp = np.asarray(ref_cloud.points if hasattr(ref_cloud, "points") else ref_cloud, dtype=np.float64)
    tree = cKDTree(rp)
    d, idx = tree.query(tp, k=1)
    delta_xyz = (rp[idx] - tp) if delta else None
    return d, delta_xyz


def compute_directional_dft_um(delta_xyz_mm):
    """Per-axis summary of the nearest-surface displacement field, in microns.

    ``delta_xyz_mm`` is the (N, 3) second return of ``compute_vertex_error_and_delta_nearest``
    with the meshes in millimeters. Returns a list of per-direction dicts.
    """
    directions = ["X", "Y", "Z"]

    mean_signed_um = np.mean(delta_xyz_mm, axis=0) * UM_PER_MM
    mean_abs_um = np.mean(np.abs(delta_xyz_mm), axis=0) * UM_PER_MM
    std_um = np.std(delta_xyz_mm, axis=0,
                    ddof=1 if len(delta_xyz_mm) > 1 else 0) * UM_PER_MM

    dft_table = []
    for i, direction in enumerate(directions):
        dft_table.append(
            {
                "direction": direction,
                "mean_signed_dft_um": mean_signed_um[i],
                "mean_abs_dft_um": mean_abs_um[i],
                "std_dft_um": std_um[i],
            }
        )
    return dft_table


def chamfer(a, b, sample=12000, seed=0):
    """Symmetric Chamfer distance between two point sets (mean NN distance both ways).

    CD = 0.5 * ( mean_{p in a} min_{q in b} |p-q| + mean_{q in b} min_{p in a} |q-p| )

    ``a``/``b`` can be (N, 3) arrays, Open3D TriangleMesh (sampled uniformly on the
    surface) or PointCloud, or trimesh Trimesh/PointCloud. Both sides are subsampled
    to ``sample`` points so density normalizes out; the distance then measures
    surface coverage/completeness and shape agreement. Lower = closer.

    Requires scipy.
    """
    from scipy.spatial import cKDTree

    a = _to_points(a, sample, seed)
    b = _to_points(b, sample, seed)
    rng = np.random.default_rng(seed)
    if len(a) > sample:
        a = a[rng.choice(len(a), sample, replace=False)]
    if len(b) > sample:
        b = b[rng.choice(len(b), sample, replace=False)]
    ta, tb = cKDTree(a), cKDTree(b)
    da, _ = ta.query(b, k=1)
    db, _ = tb.query(a, k=1)
    return float(0.5 * (da.mean() + db.mean()))


def _to_points(obj, sample, seed):
    """Accept point arrays or meshes; return an (N, 3) float64 array."""
    if isinstance(obj, np.ndarray):
        return np.asarray(obj, float)
    # Open3D remains the canonical backend, but duck-type common mesh shapes
    # (trimesh too) via .vertices and optional sampling methods.
    if isinstance(obj, o3d.geometry.TriangleMesh):
        o3d.utility.random.seed(seed)
        return np.asarray(obj.sample_points_uniformly(number_of_points=sample).points, float)
    if isinstance(obj, o3d.geometry.PointCloud):
        return np.asarray(obj.points, float)
    v = getattr(obj, "vertices", None)
    if v is not None:
        if hasattr(obj, "sample"):                      # trimesh
            try:
                return np.asarray(obj.sample(sample), float)
            except Exception:
                pass
        return np.asarray(v, float)
    raise TypeError(f"cannot extract points from {type(obj)!r}")
