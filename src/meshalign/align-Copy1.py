"""Rigid registration: symmetry-safe local refine + global FPFH/RANSAC search.

Design contract (the reason this module exists instead of one Open3D call):

1. ``align_local`` tests THREE candidates -- no-ICP identity, bounded ICP from
   identity, bounded ICP from a 180deg yaw seed -- and arbitrates between them on
   the ACTUAL fine point-to-surface metric, not ICP's own fitness. An
   already-correct mesh can never regress (identity is always a candidate).

2. Thresholds are CAPPED (the coarse-to-fine ladder never opens wide) so a started-
   close pair cannot leave its basin. A free global search on a near-symmetric shape
   can flip or tilt into a wrong pose that still scores well; this module never
   takes that risk on a pair that's already close.

3. Point-to-POINT throughout. Point-to-plane lets correspondences slide along the
   surface tangent, which settles in tilted/flipped local minima on near-symmetric
   shapes and can wreck a good init.

4. Rigid only -- no scaling. A wrong scale is a genuine defect and must show up in
   the measurement.

5. Scoring uses exact point-to-triangle distances via Open3D RaycastingScene (BVH),
   not a KD-tree -- deterministic across processes.

Units: the code assumes millimeters by lab convention, but all thresholds are
arguments -- the code itself is unit-agnostic.
"""
import os
import copy

import numpy as np
import open3d as o3d

from .config import DEFAULT_SEED, GLOBAL_YAW_INITS_DEG, MODE_PRESETS, DEFAULT_MODE, preset
from .io import set_global_seed, load_mesh, load_ref_cached, characteristic_length_um
from .measure import (compute_vertex_error_and_delta_nearest, lumped_areas,
                      area_weighted_mean, area_weighted_cte)

# Identical-ish routing threshold: pairs closer than this mean point-to-surface
# distance (mm) take the bounded LOCAL path; farther pairs take the GLOBAL search.
SKIP_THRESH_MM = 3.0


def _kind(obj):
    """'mesh' | 'cloud' for Open3D TriangleMesh / PointCloud (or anything with
    .points and no .vertices)."""
    if isinstance(obj, o3d.geometry.TriangleMesh):
        return "mesh"
    if isinstance(obj, o3d.geometry.PointCloud):
        return "cloud"
    if hasattr(obj, "vertices") and hasattr(obj, "triangles"):
        return "mesh"
    if hasattr(obj, "points"):
        return "cloud"
    raise TypeError(f"expected an Open3D TriangleMesh or PointCloud, got {type(obj)!r}")


def _sample_cloud(cloud, n, rng):
    """Seeded subsample of a cloud to <= n points (all of them if smaller).

    Deterministic: np.default_rng(seed) choice with a fixed point order.
    """
    pts = np.asarray(cloud.points)
    if len(pts) <= n:
        return pts
    idx = rng.choice(len(pts), n, replace=False)
    idx.sort()                       # fixed order -> deterministic downstream
    return pts[idx]


def _icp_sample(ref, test, n_samples, seed):
    """Sample n points on both sides (test=src, ref=dst); seed the global RNG
    first so sampling and any downstream RANSAC are deterministic for this call.

    Meshes are sampled area-uniformly (with domain normals); clouds are
    subsampled with a seeded RNG and get normals estimated where missing.
    Estimated normals are then oriented CONSISTENTLY toward the cloud centroid
    -- per-point plane fits have ambiguous signs, and FPFH on mixed-orientation
    normals degrades badly (measured: RANSAC candidate misses on near-symmetric
    parts). Centroid-orientation is deterministic and outward-consistent for
    star-shaped CAD; this is candidate generation only -- the referee never
    uses cloud normals."""
    if seed is not None:
        set_global_seed(seed)
    rng = np.random.default_rng(seed if seed is not None else DEFAULT_SEED)
    if _kind(test) == "mesh":
        src = test.sample_points_uniformly(number_of_points=n_samples)
    else:
        src = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(_sample_cloud(test, n_samples, rng)))
    if _kind(ref) == "mesh":
        dst = ref.sample_points_uniformly(number_of_points=n_samples)
    else:
        dst = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(_sample_cloud(ref, n_samples, rng)))
    for cloud in (src, dst):
        if not cloud.has_normals():
            cloud.estimate_normals()
            pts = np.asarray(cloud.points)
            cloud.orient_normals_towards_camera_location(pts.mean(axis=0))
    return src, dst


def _test_scene(test_mesh):
    """Build a BVH raycasting scene from the test mesh triangles (once) for exact
    point-to-triangle distance queries. Used by the fit score instead of any
    KD-tree -- the raycasting path is deterministic across processes (no parallel
    reduction order), unlike scipy cKDTree(workers=-1) or Open3D's point-cloud
    KD-tree distance."""
    test_t = o3d.t.geometry.TriangleMesh.from_legacy(test_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(test_t)
    return scene


def _mean_ref_to_test(test_scene, ref_pts, T, weights=None):
    """Mean point-to-SURFACE distance from each ref point to the test mesh
    transformed by ``T`` -- exact point-to-triangle via the RaycastingScene.

    The scene holds the test mesh in its untransformed frame, so ref points are
    queried mapped by the inverse rigid transform (distances are rigid-invariant).
    This avoids rebuilding the BVH per candidate transform and stays deterministic.
    Used as a coarse fit score to rank inits and gate acceptance; final arbitration
    in ``align_local`` uses the full ``compute_vertex_error_and_delta_nearest``.

    ``weights`` (T1, opt-in): per-ref-point lumped areas -- the mean becomes
    the area-weighted G_hat. Default None keeps the per-vertex mean
    (the tuned behavior, byte-identical).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    q = ref_pts @ R - R.T @ t  # T^-1 applied to ref points (rigid inverse)
    query = o3d.core.Tensor(q.astype(np.float32))
    nearest = test_scene.compute_closest_points(query)["points"].numpy().astype(np.float64)
    d = np.linalg.norm(q - nearest, axis=1)
    if weights is None:
        return float(np.mean(d))
    w = np.asarray(weights, dtype=np.float64)
    keep = w > 0.0
    return float((w[keep] * d[keep]).sum() / w[keep].sum())


# --------------------------------------------------
# Referee dispatch for mixed mesh/point-cloud inputs.
#
# The mesh-mesh referee measures ref-points -> test-SURFACE (exact
# point-to-triangle, BVH). When the test side is a cloud there is no test
# surface, so the referee flips: cloud points (in the candidate pose) ->
# ref-surface BVH. When neither side has a surface (cloud-vs-cloud), exact
# point-to-triangle is impossible; the referee falls back to a symmetric
# nearest-neighbor mean over seeded subsamples via scipy cKDTree with the
# default SINGLE-WORKER query -- the parallel KD-tree reductions are the
# nondeterministic ones (see _test_scene); single-worker queries are
# deterministic across processes.
#
# Referee directions by input kind (documented in the README):
#   mesh/mesh  : ref vertices -> test surface          (unchanged, tuned)
#   mesh/cloud : test cloud  -> ref surface             (flipped, no choice)
#   cloud/mesh : ref cloud   -> test surface            (same machinery)
#   cloud/cloud: symmetric NN mean (capped, seeded)     (unsigned)
# --------------------------------------------------
CLOUD_REFEREE_SAMPLE = 50_000     # per-side cap for cloud-vs-cloud arbitration


def _make_referee(ref, test, seed=DEFAULT_SEED, area_referee=False):
    """Build ``score_fn(T) -> mean distance (mm)`` + identity score for any input
    combination. Mesh-mesh callers keep their legacy in-function scorers; this
    dispatch serves the cloud paths (and any caller wanting a uniform interface).

    ``area_referee=True`` (T1, opt-in, J2 gate -- default OFF): for a MESH
    reference the per-vertex mean becomes the area-weighted G_hat (lumped
    re-weighting of the same queries; zero RNG, deterministic). Cloud
    references have no tessellation and keep the plain point mean.

    Returns ``(score_fn, base_score)`` where ``base_score`` is score_fn(I).
    """
    rk, tk = _kind(ref), _kind(test)
    rng = np.random.default_rng(seed if seed is not None else DEFAULT_SEED)

    if tk == "mesh":
        # ref points -> test surface (works for mesh OR cloud references:
        # the referee only needs the test-side surface)
        scene = _test_scene(test)
        if rk == "mesh":
            ref_pts = np.asarray(ref.vertices)
            w = lumped_areas(ref) if area_referee else None
        else:
            ref_pts = _sample_cloud(ref, CLOUD_REFEREE_SAMPLE, rng)
            w = None

        def score_fn(T):
            return _mean_ref_to_test(scene, ref_pts, T, weights=w)

        return score_fn, score_fn(np.eye(4))

    if rk == "mesh":
        # test cloud (in pose T) -> ref surface: exact, deterministic BVH
        ref_t = o3d.t.geometry.TriangleMesh.from_legacy(ref)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(ref_t)
        test_pts = _sample_cloud(test, CLOUD_REFEREE_SAMPLE, rng)

        def score_fn(T):
            R, t = T[:3, :3], T[:3, 3]
            q = test_pts @ R.T + t
            query = o3d.core.Tensor(q.astype(np.float32))
            nearest = scene.compute_closest_points(query)["points"].numpy().astype(np.float64)
            return float(np.mean(np.linalg.norm(q - nearest, axis=1)))

        return score_fn, score_fn(np.eye(4))

    # cloud vs cloud: symmetric capped NN mean, single-worker cKDTree
    from scipy.spatial import cKDTree
    ref_pts = _sample_cloud(ref, CLOUD_REFEREE_SAMPLE, rng)
    test_pts = _sample_cloud(test, CLOUD_REFEREE_SAMPLE, rng)
    tree_ref = cKDTree(ref_pts)

    def score_fn(T):
        R, t = T[:3, :3], T[:3, 3]
        moved = test_pts @ R.T + t
        moved_tree = cKDTree(moved)          # rebuild per pose: clouds are small
        d_tr, _ = tree_ref.query(moved, k=1)      # moved test -> ref
        d_rt, _ = moved_tree.query(ref_pts, k=1)  # ref -> moved test
        return float(0.5 * (d_tr.mean() + d_rt.mean()))

    return score_fn, score_fn(np.eye(4))


# --------------------------------------------------
# Refinement-engine options (2026-09 A/B, Tier-D of the candidates doc).
#
# "gicp": swap the ICP-ladder rungs for koide3's small_gicp engine
# (GICP: plane-aware covariances without point-to-plane's sliding failure
# mode -- the plane information enters through covariances, not the
# objective). Same candidates, same ladder thresholds (as correspondence
# caps), same referee: the ONLY variable is the refinement engine.
# Opt-in per-call (``refine_engine="gicp"`` through preset overrides) or
# per-process (MESHALIGN_REFINE_ENGINE=gicp); the default "o3d" path is
# byte-identical to the tuned profile.
# --------------------------------------------------
REFINE_ENGINE = os.environ.get("MESHALIGN_REFINE_ENGINE", "o3d")


def _gicp_run_ladder(src, dst, T0, ladder, max_iter):
    """Coarse-to-fine refinement with small_gicp's GICP engine.

    Each rung's threshold becomes the GICP max_correspondence_distance
    (same coarse-to-fine contraction as the o3d ladder). Downsampling is
    tied to the FINAL (finest) rung, not the bbox: a 5%-of-bbox voxel
    destroys the small asymmetric features that disambiguate near-symmetric
    parts (measured on hard near-symmetric test geometry
    at 1.0 mm -- the fine structure IS the signal).
    """
    import small_gicp
    src_pts = np.ascontiguousarray(np.asarray(src.points), dtype=np.float64)
    dst_pts = np.ascontiguousarray(np.asarray(dst.points), dtype=np.float64)
    T = np.asarray(T0, dtype=np.float64)
    ds_res = max(min(float(ladder[-1]), 1.0), 0.05) if ladder else 1.0
    for thresh in ladder:
        res = small_gicp.align(
            dst_pts, src_pts, T,                    # target, source, init T (src->dst)
            registration_type="GICP",
            downsampling_resolution=ds_res,
            max_correspondence_distance=float(thresh),
            max_iterations=int(max_iter),
        )
        T = np.asarray(res.T_target_source)
    return T


def _adaptive_run_ladder(src, dst, T0, first_thresh, max_iter, ladder_eps=0.0,
                         floor_mm=None, factor=2.0, max_rungs=10):
    """Self-annealing KISS-ICP-style refinement (Tier-D #3, candidates doc).

    Instead of fixed rungs (15 -> 5 -> 2 -> 0.5 mm), each next correspondence
    threshold is derived from the data: ``max(factor * inlier_rmse, floor)``.
    The threshold tracks the ACTUAL misalignment + noise scale, so a noisy
    part keeps its wider catch basin and a clean part anneals deeper than the
    fixed 0.5 mm rung (KISS-ICP, Vizzo et al. RA-L 2023 -- the one mechanism
    that beat meshalign on a corpus cell: auto-tool 0.864 vs 0.816).

    ``floor_mm=None`` (default) makes the floor SCALE-RELATIVE: 0.1% of the
    reference-cloud PCA extent (0.1 mm at the 100 mm reference scale) -- an
    absolute floor would mean different things on a 10 mm micro part and a
    500 mm frame. Deterministic given the ICP results; the referee still
    arbitrates candidates.
    """
    if floor_mm is None:
        _dp = np.asarray(dst.points)
        _, _, _vt = np.linalg.svd(_dp - _dp.mean(axis=0), full_matrices=False)
        _proj = (_dp - _dp.mean(axis=0)) @ _vt.T
        _ext = float((_proj.max(axis=0) - _proj.min(axis=0)).max())
        floor_mm = max(1e-3 * _ext, 0.01)
    T = np.asarray(T0, dtype=np.float64)
    thresh = float(first_thresh)
    for _ in range(int(max_rungs)):
        reg = o3d.pipelines.registration.registration_icp(
            src, dst, thresh, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )
        T_new = reg.transformation
        if ladder_eps > 0.0 and np.linalg.norm(T_new - T) < ladder_eps:
            return np.asarray(T_new)
        T = np.asarray(T_new)
        # anneal from the data: 2x the inlier RMSE, floored
        thresh = max(factor * float(reg.inlier_rmse), floor_mm)
    return T


def _resolve_refine_engine(explicit=None):
    return explicit if explicit is not None else REFINE_ENGINE


def align_local(ref_mesh, test_mesh, src, dst, base_score,
                yaw_inits_deg=(0.0, 180.0), src_pts=None, ref_pts=None,
                return_scores=False, verbose=True,
                ladder=None, max_iter=None, ladder_eps=0.0, score_fn=None,
                refine_engine=None, adaptive_ladder=False,
                area_referee=False, arb_beta=0.25, arb_phi=0.01):
    """Bounded LOCAL point-to-point ICP for an already-well-aligned pair.

    Tests THREE candidates and keeps the lowest fine surface DFT::

      - no-ICP (the identity pose, as-is)
      - 0deg   (bounded-local ICP refined from identity)
      - 180deg (bounded-local ICP refined from a centroid-aligned 180deg yaw)

    The 180deg candidate recovers the occasional "cleaned the wrong way up" failure
    an identity-start ladder alone cannot escape (it's outside the identity basin).
    No-ICP (identity) is always a candidate, so an already-correct mesh can never
    regress. A full global FPFH+RANSAC search is not just wasted work on an
    already-close pair, it is actively dangerous -- it can flip/tilt a near-symmetric
    shape into a wrong pose that still scores well.

    Each candidate is arbitrated by the ACTUAL fine point-to-triangle DFT (not the
    coarse pre-fit proxy): the two metrics don't always agree.

    ``ref_mesh``/``test_mesh`` may be meshes or point clouds. For mesh-mesh the
    arbitration measures ref vertices -> test surface via
    ``compute_vertex_error_and_delta_nearest`` (the tuned behavior, unchanged).
    When ``score_fn(T) -> mae`` is passed (cloud inputs, see ``_make_referee``),
    candidates are arbitrated on it instead and the same contract holds.

    ``area_referee=True`` (T1 J2 gate, opt-in -- DEFAULT OFF: the per-vertex
    weighting is a near-symmetric margin amplifier
    results depend on, t1-sampling-bias.md section 6): for mesh-mesh pairs the
    arbitration statistic becomes the area-uniform hybrid
    ``S = G_hat + arb_beta * CTE_arb_phi`` (area-weighted mean + area-conditional
    tail expectation over the worst ``arb_phi`` fraction of AREA), replacing the
    per-vertex ``mean + 0.5 * p95``. Same queries, re-weighted -- zero RNG, the
    candidate pool is byte-identical; identity is scored on the SAME statistic
    as the winners (the no-regression invariant keeps its shape). Ignored when
    ``score_fn`` is given (cloud inputs have no ref tessellation).

    Returns the winning 4x4 T, or ``(T, scores)`` when ``return_scores`` --
    ``scores`` maps each tested yaw-deg -> fine DFT MAE (same units as the meshes),
    keyed ``None`` for the no-ICP identity baseline.
    """
    src_c = np.asarray(src.points).mean(axis=0) if src_pts is None else src_pts.mean(axis=0)
    ref_c = np.asarray(dst.points).mean(axis=0) if ref_pts is None else ref_pts.mean(axis=0)

    inits = [(_yaw_init(src_c, ref_c, d) if d != 0.0 else np.eye(4)) for d in yaw_inits_deg]

    _dev = _resolve_icp_device()
    _engine = _resolve_refine_engine(refine_engine)
    _tensor_icp = None
    if _engine == "gicp":
        pass    # small_gicp path: no tensor upload, rungs call small_gicp.align
    elif _dev is not None:
        try:
            _tensor_icp = _TensorICP(src, dst, _dev)
        except Exception as _e:
            print(f"tensor-ICP unavailable ({_e!r}); falling back to CPU ICP")
            _tensor_icp = None

    def run_ladder(T0):
        T = T0
        lad = ladder if ladder is not None else (4.0, 3.0, 2.0, 1.0, 0.5)
        mi = max_iter if max_iter is not None else 200
        if adaptive_ladder:
            return _adaptive_run_ladder(src, dst, T, lad[0], mi, ladder_eps)
        if _engine == "gicp":
            return _gicp_run_ladder(src, dst, T, lad, mi)
        if _tensor_icp is not None:
            return _tensor_icp.run_ladder(T, lad, mi, ladder_eps)
        for thresh in lad:
            reg = o3d.pipelines.registration.registration_icp(
                src, dst, thresh, T,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=mi),
            )
            if ladder_eps > 0.0 and np.linalg.norm(reg.transformation - T) < ladder_eps:
                T = reg.transformation
                break   # remaining (finer) rungs won't move the pose either
            T = reg.transformation
        return T

    # T1 area-referee weights (mesh-mesh only): lumped ref vertex areas,
    # computed once per call from the tessellation -- zero RNG, zero extra
    # BVH queries (the candidate pool stays byte-identical).
    _arb_areas = lumped_areas(ref_mesh) if (area_referee and score_fn is None) else None

    if score_fn is not None:
        mae_identity = score_fn(np.eye(4))
    else:
        err_identity, _ = compute_vertex_error_and_delta_nearest(ref_mesh, test_mesh)
        if area_referee:
            mae_identity = area_weighted_mean(err_identity, _arb_areas)
        else:
            mae_identity = float(np.mean(np.abs(err_identity)))
    # Tail-aware arbitration (2026-09 real-CT finding): the MEAN-DFT referee
    # is tail-blind -- on nozzle scan 011 the 180-deg mate scored BETTER on
    # the mean (115.5 vs 118.2 um) while its p95 was 2.7x WORSE (494 vs
    # ~180 um): a flip traded tail error for mean error. Arbitrate yaw mates
    # on mean + 0.5 x p95 (the lab's deviation-map tail is the defect
    # signal; the mean hides it). The plain mean stays in ``scores`` for
    # reporting.
    #
    # area_referee=True (T1 J2, opt-in): the same functional shape on the
    # AREA measure -- G_hat + beta * CTE_phi (t1-sampling-bias.md section
    # 6.5/6.6: per-vertex p95 quantiles mix sampling conventions, so the
    # tail term moves to the area-CTE in the same change).
    def _arb_score(T):
        if score_fn is not None:
            return score_fn(T)
        mesh_c = copy.deepcopy(test_mesh)
        mesh_c.transform(T)
        err, _ = compute_vertex_error_and_delta_nearest(ref_mesh, mesh_c)
        ad = np.abs(err)
        if area_referee:
            s = area_weighted_mean(ad, _arb_areas)
            if arb_beta > 0.0:
                s += arb_beta * area_weighted_cte(ad, _arb_areas, arb_phi)
            return s
        return float(ad.mean() + 0.5 * np.percentile(ad, 95))

    scores = {None: mae_identity}
    # identity joins the arbitration pool at its OWN tail-aware score (the
    # no-regression guarantee must hold under the same metric that picks the
    # winner -- initializing at np.inf would let a worse-than-identity yaw
    # candidate win and regress the pose, which the invariant tests caught)
    best_T, best_mae, best_deg = np.eye(4), _arb_score(np.eye(4)), None
    for deg, T0 in zip(yaw_inits_deg, inits):
        T = run_ladder(T0)
        if score_fn is not None:
            mae = score_fn(T)
            arb = mae            # cloud/other referees: no signed per-point data here
        else:
            mesh_c = copy.deepcopy(test_mesh)
            mesh_c.transform(T)
            err, _ = compute_vertex_error_and_delta_nearest(ref_mesh, mesh_c)
            ad = np.abs(err)
            if area_referee:
                mae = area_weighted_mean(ad, _arb_areas)
                arb = mae
                if arb_beta > 0.0:
                    arb += arb_beta * area_weighted_cte(ad, _arb_areas, arb_phi)
            else:
                mae = float(ad.mean())
                arb = float(ad.mean() + 0.5 * np.percentile(ad, 95))
        scores[deg] = mae
        if arb < best_mae:
            best_T, best_mae, best_deg = T, arb, deg

    if best_deg is None:
        if verbose:
            print(f"ICP local-refine did not improve surface error "
                  f"({mae_identity*1000:.1f} -> {best_mae*1000:.1f} um); keeping original pose.")
        return (np.eye(4), scores) if return_scores else np.eye(4)
    ang_deg = np.degrees(np.arccos(np.clip((np.trace(best_T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    if verbose:
        print(f"ICP local-refined (already aligned, fit {base_score:.3f} mm; "
              f"surface DFT {mae_identity*1000:.1f} -> {best_mae*1000:.1f} um, "
              f"best yaw {best_deg:.0f}deg, rotation {ang_deg:.2f}deg).")
    return (best_T, scores) if return_scores else best_T


def _yaw_init(src_centroid, ref_centroid, deg):
    """Centroid-aligned Z-rotation init: rotate ``deg`` about Z, then translate so
    the source centroid lands on the reference centroid. Coarse seed for both the
    global (RANSAC+yaw) and local (bounded-ICP-from-yaw) paths."""
    a = np.radians(deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0],
                   [0.0, 0.0, 1.0]])
    T0 = np.eye(4)
    T0[:3, :3] = Rz
    T0[:3, 3] = ref_centroid - Rz @ src_centroid
    return T0


# --------------------------------------------------
# GPU acceleration of the ICP refinement ladder (align_global / align_local).
#
# Open3D 0.19's tensor-API ICP runs on CUDA; the legacy-geometry API is
# CPU-only. The GPU path uploads both sampled clouds ONCE per register()
# call and runs every ladder rung of every pose candidate on the device;
# results are numerically equivalent to the CPU path (IoU identical to
# 1e-6, dT frob ~2e-4).
#
# MEASURED on wpk12 (RTX PRO 6000 Blackwell, o3d 0.19.0-cuda): at our cloud
# sizes the GPU path is a NET LOSS -- fast mode (5k pts) 2.9s GPU vs 1.17s
# CPU; accuracy (100k pts) 55.8s GPU vs 32.5s CPU. Per-call upload + kernel
# launch overhead dominates; the legacy CPU KD-tree path with warm TBB
# threading wins at this scale. GPU only pays off far above our sizes, so
# the default is CPU. Opt in per-process with MESHALIGN_ICP_DEVICE=cuda:0
# (useful if parts grow to 500k+ sampled points).
# --------------------------------------------------
ICP_DEVICE = os.environ.get("MESHALIGN_ICP_DEVICE", "cpu")


def _resolve_icp_device():
    """Resolve the ICP device once. Returns None to mean 'legacy CPU path'."""
    global _ICP_DEVICE_RESOLVED
    if "_ICP_DEVICE_RESOLVED" not in globals():
        dev = ICP_DEVICE
        if dev == "auto":   # explicit opt-in: prefer CUDA when present
            dev = "cuda:0" if o3d.core.cuda.is_available() else None
        elif dev in ("cpu", ""):
            dev = None
        _ICP_DEVICE_RESOLVED = dev
    return _ICP_DEVICE_RESOLVED


class _TensorICP:
    """Tensor-API ICP refinement on one device, reusing resident clouds.

    Built once per align_* call (points uploaded once); run_ladder() is then
    called for every candidate init. Deterministic given the same inputs
    (gradient descent ICP, no sampling inside).
    """

    def __init__(self, src, dst, device):
        self.device = o3d.core.Device(device)
        # NB: build the positions tensors ON the device -- assigning a CPU
        # tensor into a CUDA PointCloud leaves the attribute on CPU.
        self.src_t = o3d.t.geometry.PointCloud(o3d.core.Tensor(
            np.ascontiguousarray(np.asarray(src.points), dtype=np.float32)).to(self.device))
        self.dst_t = o3d.t.geometry.PointCloud(o3d.core.Tensor(
            np.ascontiguousarray(np.asarray(dst.points), dtype=np.float32)).to(self.device))
        self.estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()

    def run_ladder(self, T0, ladder, max_iter, ladder_eps=0.0):
        T = np.eye(4, dtype=np.float64)
        T[:] = T0
        for thresh in ladder:
            reg = o3d.t.pipelines.registration.icp(
                self.src_t, self.dst_t, float(thresh),
                o3d.core.Tensor(T.astype(np.float32)).to(self.device),
                self.estimation,
                o3d.t.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=int(max_iter)),
            )
            T_new = reg.transformation.cpu().numpy().astype(np.float64)
            if ladder_eps > 0.0 and np.linalg.norm(T_new - T) < ladder_eps:
                return T_new
            T = T_new
        return T


def _fpfh_features(pcd, voxel):
    """Voxel-downsample + normals + FPFH features at a given feature scale.

    Extracted from align_global's closure (2026-09) so the composable
    CoarseFineAligner shares the exact same feature pipeline -- RNG-free
    (FPFH/normal estimation consume no global RNG), so sharing is
    behavior-identical.
    """
    d = pcd.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return d, f


def _ransac_draw(src_down, dst_down, src_fpfh, dst_fpfh, voxel,
                 mutual_filter=False):
    """One FPFH+RANSAC pose draw (module level for reuse by the composable
    pipeline). RANSAC consumes the Open3D global RNG -- seeded by the caller
    via set_global_seed / _icp_sample. mutual_filter=False: sparse meshes
    produce only a handful of mutual correspondences, causing RANSAC to miss
    the correct pose entirely; one-way correspondences give it enough points,
    outliers are handled by the RANSAC loop itself.
    """
    r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, dst_down, src_fpfh, dst_fpfh, mutual_filter, voxel * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))
    return r.transformation


def align_global(src, dst, src_pts, ref_pts, base_score, test_scene=None,
                 yaw_inits_deg=(0.0, 90.0, 180.0, 270.0), ransac_attempts=None,
                 verbose=True, ladder=None, max_iter=None, ladder_eps=0.0,
                 score_fn=None, refine_engine=None, fgr_seed=False,
                 symmetry_sweep=False, adaptive_ladder=False,
                 gate_k=None, fine_ref_pts=None, fine_ref_weights=None):
    """Pose-free global registration: FPFH feature matching + RANSAC, refined
    coarse-to-fine with point-to-point ICP.

    ``yaw_inits_deg`` adds centroid-aligned Z-rotation seeds tried alongside the
    RANSAC inits as a DETERMINISTIC fallback when RANSAC cannot find the pose on
    sparse geometry. Default is symmetric quarter-turns.

    ``fgr_seed=True`` adds an Open3D FGR (Fast Global Registration, Zhou et al.
    ECCV 2016) candidate to the init pool. FGR is UNRELIABLE on its own (flips
    on the symmetric dome grid) -- but under this architecture an unsafe
    candidate is FREE: the referee arbitrates, a bad candidate just loses
    (same principle as the Go-ICP candidate in accuracy mode).

    ``symmetry_sweep=True`` adds a final candidate stage for lattice/TPMS-class
    geometry: the 24 rotations of the CUBE GROUP applied to the best-so-far
    pose (about the ref centroid), each refined with the full ladder and
    referee-arbitrated. Lattices and TPMS shells have cubic symmetry groups
    whose mates form deep wrong basins that capture the main candidate pool
    (measured: gyroid 120-deg, schwarz-p 120-deg, strut-bcc 180-deg errors
    with the standard pool; the sweep recovers the true basin whenever any
    symmetry mate was found -- the referee margins are thin but real,
    0.0451 true-mate vs 0.0463 wrong-mate on schwarz-p). Unsafe by design,
    free by architecture: bad mates just lose arbitration. Included in
    ``accuracy`` mode.

    ``refine_engine='gicp'`` swaps the ICP-ladder rungs for small_gicp's GICP
    engine (plane-aware covariances) -- same candidates, same thresholds, same
    referee; the only variable is the refinement engine.

    ``fine_ref_weights`` (T1 J2, opt-in): per-ref-vertex lumped areas matching
    ``fine_ref_pts`` -- the fine re-arbitration mean becomes the area-weighted
    G_hat (same queries, re-weighted; zero RNG). Default None keeps the
    per-vertex mean.

    Point-to-POINT throughout, rigid only. ``test_scene`` is a prebuilt
    RaycastingScene from the test mesh; the per-init fit score is the exact
    point-to-surface distance (deterministic, no KD-tree). For cloud inputs
    (no test surface) pass ``score_fn`` from ``_make_referee`` instead and
    leave ``test_scene`` as None -- candidates are then ranked on the
    input-appropriate referee and the same accept-only-if-better-than-identity
    gate applies.

    The transform is accepted only if it beats the identity fit; otherwise
    np.eye(4) is returned (an already-correct mesh never regresses).
    """
    src_c = src_pts.mean(axis=0)
    ref_c = ref_pts.mean(axis=0)

    # GPU/legacy dispatch for the ICP ladder: build once, reuse for every
    # candidate init (clouds uploaded to device a single time).
    _dev = _resolve_icp_device()
    _engine = _resolve_refine_engine(refine_engine)
    _tensor_icp = None
    if _engine == "gicp":
        pass    # small_gicp path: rungs call small_gicp.align directly
    elif _dev is not None:
        try:
            _tensor_icp = _TensorICP(src, dst, _dev)
        except Exception as _e:          # CUDA OOM, driver hiccup, ...
            print(f"tensor-ICP unavailable ({_e!r}); falling back to CPU ICP")
            _tensor_icp = None

    yaw_inits = [_yaw_init(src_c, ref_c, d) for d in yaw_inits_deg]

    # Voxel scaled to object size (bbox diagonal / 50) so feature radii are sensible
    # regardless of the meshes' raw units.
    voxel = float(np.linalg.norm(ref_pts.max(axis=0) - ref_pts.min(axis=0))) / 50.0

    src_down, src_fpfh = _fpfh_features(src, voxel)
    dst_down, dst_fpfh = _fpfh_features(dst, voxel)

    def ransac_init(mutual_filter=False):
        return _ransac_draw(src_down, dst_down, src_fpfh, dst_fpfh, voxel,
                           mutual_filter)

    # RANSAC draws are seeded via the Open3D global RNG (set_global_seed in
    # _icp_sample above), so the attempts are reproducible run-to-run.
    n_attempts = ransac_attempts if ransac_attempts is not None else 10
    inits = yaw_inits + [ransac_init() for _ in range(n_attempts)]

    if gate_k is not None and len(inits) > int(gate_k):
        # Coarse gate (improvement idea #3, 2026-09): score every RAW init with
        # one cheap referee query (the BVH is already built) and spend the
        # expensive 4-rung ladder only on the top-k. Pose-safety is unchanged
        # -- the referee still arbitrates the refined survivors and the
        # identity bay still gates acceptance. A raw init that is grossly
        # misaligned would burn a full ladder to lose anyway.
        def _raw_score(T0):
            if score_fn is not None:
                return score_fn(T0)
            return _mean_ref_to_test(test_scene, ref_pts, T0)
        ranked = sorted(inits, key=_raw_score)
        inits = ranked[:int(gate_k)]
        if verbose:
            print(f"coarse gate: kept {len(inits)}/{len(ranked)} inits for the ladder")

    if fgr_seed:
        # Tier-D #4 (candidates doc): FGR as ONE more candidate in the pool.
        # FGR alone flips on near-symmetric shapes -- under the referee that
        # costs nothing: a bad candidate just loses arbitration.
        try:
            fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                src_down, dst_down, src_fpfh, dst_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=voxel * 1.5))
            inits.append(fgr.transformation)
        except Exception as _e:
            print(f"fgr candidate unavailable ({_e!r}); skipping")

    def run_icp(T):
        lad = ladder if ladder is not None else (15.0, 5.0, 2.0, 1.0, 0.5)
        mi = max_iter if max_iter is not None else 200
        if adaptive_ladder:
            return _adaptive_run_ladder(src, dst, T, lad[0], mi, ladder_eps)
        if _engine == "gicp":
            return _gicp_run_ladder(src, dst, T, lad, mi)
        if _tensor_icp is not None:
            return _tensor_icp.run_ladder(T, lad, mi, ladder_eps)
        for thresh in lad:
            reg = o3d.pipelines.registration.registration_icp(
                src, dst, thresh, T,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=mi),
            )
            if ladder_eps > 0.0 and np.linalg.norm(reg.transformation - T) < ladder_eps:
                T = reg.transformation
                break   # remaining (finer) rungs won't move the pose either
            T = reg.transformation
        return T

    best_T, best_score = None, np.inf
    scored = []                     # (score, T) for every refined candidate
    for T0 in inits:
        T = run_icp(T0)
        if score_fn is not None:
            s = score_fn(T)
        else:
            s = _mean_ref_to_test(test_scene, ref_pts, T)
        scored.append((s, T))
        if s < best_score:
            best_score, best_T = s, T

    if symmetry_sweep and best_T is not None:
        # Lattice/TPMS rescue stage (2026-09 complex-geometry cohort): the
        # cubic symmetry group's mates form deep wrong basins that capture
        # the main pool. Sweep the 24 cube-group rotations of the best pose
        # (about the ref centroid), refine each with the full ladder, and let
        # the referee arbitrate -- the same "unsafe candidates are free"
        # principle as the Go-ICP/FGR candidates.
        import itertools as _it
        ref_c = ref_pts.mean(axis=0)
        n_sw = 0
        for _perm in _it.permutations(range(3)):
            for _signs in _it.product((1, -1), repeat=3):
                M = np.zeros((3, 3))
                for _i, _j in enumerate(_perm):
                    M[_i, _j] = _signs[_i]
                if round(np.linalg.det(M)) != 1:
                    continue
                R = M @ best_T[:3, :3]
                t = best_T[:3, 3] + ref_c - M @ ref_c
                T0s = np.eye(4); T0s[:3, :3] = R; T0s[:3, 3] = t
                T_sw = run_icp(T0s)
                if score_fn is not None:
                    s_sw = score_fn(T_sw)
                else:
                    s_sw = _mean_ref_to_test(test_scene, ref_pts, T_sw)
                scored.append((s_sw, T_sw))
                if s_sw < best_score:
                    best_score, best_T = s_sw, T_sw
                n_sw += 1
        if verbose:
            print(f"symmetry sweep: {n_sw} cube-group mates refined; best fit now "
                  f"{best_score:.4f} mm")

    if fine_ref_pts is not None and len(scored) > 1:
        # Fine-DFT re-arbitration (improvement idea #4, 2026-09): the global
        # winner above was picked on the COARSE proxy (the 10k sampled ref
        # points). Near-tied coarse scores let thread noise pick the winner --
        # the documented thing-1992245 basin wobble (config.py:79-89). Re-score
        # the top-3 refined candidates on the FINE referee (all ref vertices,
        # exact point-to-surface) and re-run the identity gate on the same
        # metric so the no-regression guarantee stays consistent.
        def _fine_score(T):
            if score_fn is not None:
                return score_fn(T)
            return _mean_ref_to_test(test_scene, fine_ref_pts, T,
                                     weights=fine_ref_weights)
        top = sorted(scored, key=lambda c: c[0])[:3]
        fine_best_s, fine_best_T = min(((_fine_score(T), T) for _, T in top),
                                       key=lambda c: c[0])
        fine_identity = _fine_score(np.eye(4))
        if verbose and len(top) > 1:
            coarse = [f"{s:.4f}" for s, _ in top]
            fine = [f"{_fine_score(T):.4f}" for _, T in top]
            print(f"fine re-arbitration: coarse [{', '.join(coarse)}] -> "
                  f"fine [{', '.join(fine)}] mm (identity {fine_identity:.4f})")
        best_T, best_score = fine_best_T, fine_best_s
        base_score_for_gate = fine_identity
    else:
        base_score_for_gate = base_score

    T = best_T
    icp_score = best_score
    if icp_score >= base_score_for_gate:
        if verbose:
            print(f"ICP rejected (fit {icp_score:.3f} >= original {base_score:.3f} mm); "
                  f"keeping original pose.")
        return np.eye(4)

    # Ambiguity advisory (2026-09 hard-cohort finding): on near-unidentifiable
    # geometry (sub-noise asymmetry -- measured cases: a flat disc's in-plane
    # spin, a near-180-deg-symmetric housing, a shallow-basin cam), SEVERAL
    # DISTINCT poses score within a few percent of the winner, so the
    # "winning" pose is weakly determined. A tie between candidates that
    # converged to the SAME pose is consensus, not ambiguity -- so the rival
    # must also differ by a meaningful rotation or translation.
    if verbose:
        rivals = []
        for s_r, T_r in scored:
            if s_r > icp_score * 1.02 or s_r <= icp_score:
                continue
            dR = T_r[:3, :3] @ T[:3, :3].T
            dang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
            dtr = float(np.linalg.norm(T_r[:3, 3] - T[:3, 3]))
            if dang > 5.0 or dtr > 2.0:
                rivals.append((s_r, dang, dtr))
        if rivals:
            worst = min(rivals, key=lambda r: r[0])
            print(f"[meshalign] AMBIGUOUS POSE advisory: {len(rivals)} rival pose(s) "
                  f"score within 2% of the winner ({icp_score:.4f} mm; best rival "
                  f"{worst[0]:.4f} mm at {worst[1]:.1f} deg / {worst[2]:.2f} mm off) "
                  f"-- the part's asymmetry is near the noise floor; treat the "
                  f"pose as weakly determined and verify before trusting "
                  f"measurements.")

    R = T[:3, :3]
    t = T[:3, 3]
    ang_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if verbose:
        print(f"ICP applied: rotation {ang_deg:.2f}deg, translation {np.linalg.norm(t):.2f} mm "
              f"(fit {base_score:.3f} -> {icp_score:.3f} mm)")
    return T


def _goicp_candidate(ref_mesh, test_mesh, timeout_s=600, seed=DEFAULT_SEED):
    """Go-ICP (Yang et al. 2013, branch-and-bound) wrapped as one more candidate.

    Go-ICP is a POINT-CLOUD method (point-to-point L2 over SE(3)); STLs enter
    its world through an area-uniform SURFACE sample (seeded), not raw
    vertices -- vertex density is a tessellation artifact, not a property of
    the shape. The downsample to n=250 rides on top of that pool. The wrapper
    solves moving->model with rot @ mov + t == model, so ref is the model.
    Returns the 4x4 candidate, or None if unavailable (module missing), timed
    out, or errored.

    PRONE TO WRONG POSES on near-symmetric geometry (documented in
    config.MODE_PRESETS): the caller must arbitrate on the fine DFT and keep
    the identity bay -- never trust this output directly.
    """
    try:
        import py_simple_go_icp_interface as gi
    except ImportError:
        print("go-icp candidate unavailable: py_simple_go_icp_interface not "
              "installed; accuracy mode falls back to the tuned solver alone.")
        return None

    import multiprocessing as mp
    set_global_seed(seed)
    ref_v = np.asarray(ref_mesh.sample_points_uniformly(
        number_of_points=5000).points)
    test_v = np.asarray(test_mesh.sample_points_uniformly(
        number_of_points=5000).points)
    try:
        with mp.get_context("spawn").Pool(1) as pool:
            async_ = pool.apply_async(
                _goicp_worker, (ref_v, test_v, 0, 250))
            rot, t = async_.get(timeout=timeout_s)
    except mp.TimeoutError:
        print(f"go-icp candidate timed out after {timeout_s}s "
              "(near-symmetric part: branch-and-bound never prunes); skipping.")
        return None
    except Exception as e:
        print(f"go-icp candidate errored ({e!r}); skipping.")
        return None
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T


def _goicp_worker(ref_v, test_v, seed, n):
    import py_simple_go_icp_interface as gi
    rot, t = gi.go_icp_match(ref_v, test_v, random_seed=seed,
                             random_downsample=n, print_output=False)
    return rot, t


def _goicp_arbitrate(ref_mesh, test_mesh, T_ma, timeout_s, verbose=True,
                     seed=DEFAULT_SEED, area_referee=False):
    """Run the Go-ICP candidate and let the fine-DFT referee decide.

    The tuned-solver pose ``T_ma`` is already no-worse-than-identity; Go-ICP
    only wins by scoring strictly better on the same fine point-to-surface
    DFT used everywhere else, so it can never regress the result. Its wins
    are advisory cross-checks (it is prone to wrong poses on near-symmetric
    geometry -- see config.MODE_PRESETS["accuracy"]).

    ``area_referee=True`` (T1 J2, opt-in): the arbitration mean becomes the
    area-weighted G_hat (same queries, lumped weights).
    """
    if verbose:
        print("accuracy mode: running the Go-ICP candidate -- WARNING: "
              "prone to wrong poses on near-symmetric geometry; it is "
              "arbitrated on the fine DFT and cannot regress the result.")
    T_gi = _goicp_candidate(ref_mesh, test_mesh, timeout_s, seed=seed)
    if T_gi is None:
        return T_ma

    _w = lumped_areas(ref_mesh) if area_referee else None

    def _score(T):
        m = copy.deepcopy(test_mesh)
        m.transform(T)
        errs, _ = compute_vertex_error_and_delta_nearest(ref_mesh, m)
        if _w is None:
            return float(np.mean(np.abs(errs)))
        return area_weighted_mean(errs, _w)

    s_ma, s_gi = _score(T_ma), _score(T_gi)
    if s_gi < s_ma:
        if verbose:
            print(f"go-icp candidate won arbitration "
                  f"(fine DFT {s_ma*1000:.1f} -> {s_gi*1000:.1f} um) -- "
                  f"advisory cross-check; verify before trusting.")
        return T_gi
    if verbose:
        print(f"go-icp candidate rejected "
              f"(fine DFT {s_gi*1000:.1f} >= {s_ma*1000:.1f} um); "
              f"keeping the tuned-solver pose.")
    return T_ma


def _teaser_candidate(ref_mesh, test_mesh, noise_bound=None, seed=DEFAULT_SEED):
    """TEASER++ (Yang, Shi & Carlone, T-RO 2021) wrapped as one more arbitrated
    candidate: certifiable truncated-least-squares registration -- max-clique
    inlier selection over FPFH correspondences + GNC-TLS rotation search.

    Unlike the Go-ICP candidate: fast (sub-second to seconds), handles
    thousands of correspondences natively (no 250-point downsample), and its
    TLS formulation is robust to extreme outlier ratios. Like Go-ICP it can
    land in wrong basins on near-symmetric geometry -- the caller MUST
    arbitrate on the fine DFT and keep the identity bay (never trust it
    directly). The full DRS optimality certificate is a separate sub-solver
    and is not run here; the fine-DFT referee is the trust anchor.

    Returns the 4x4 candidate, or None if unavailable (module missing) or
    errored.
    """
    try:
        import teaserpp_python as tp
    except ImportError:
        print("teaser candidate unavailable: teaserpp_python not installed; "
              "accuracy mode falls back to the remaining candidates alone.")
        return None
    from scipy.spatial import cKDTree

    set_global_seed(seed)
    src, dst = _icp_sample(ref_mesh, test_mesh, 5000, seed)   # src=test, dst=ref
    src_pts = np.asarray(src.points)
    dst_pts = np.asarray(dst.points)
    voxel = float(np.linalg.norm(dst_pts.max(axis=0) - dst_pts.min(axis=0))) / 50.0
    sd = src.voxel_down_sample(voxel)
    dd = dst.voxel_down_sample(voxel)
    sd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    dd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    sf = o3d.pipelines.registration.compute_fpfh_feature(
        sd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    df = o3d.pipelines.registration.compute_fpfh_feature(
        dd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    # one-way feature correspondences: src[i] <-> dst[nn[i]]
    nn = cKDTree(np.asarray(df.data).T).query(np.asarray(sf.data).T, k=1)[1]
    src_matched = np.asarray(sd.points)
    dst_matched = np.asarray(dd.points)[nn]

    p = tp.RobustRegistrationSolver.Params()
    p.estimate_scaling = False                     # rigid only (repo rule)
    p.noise_bound = float(noise_bound) if noise_bound is not None else max(voxel * 1.5, 0.05)
    solver = tp.RobustRegistrationSolver(p)
    try:
        sol = solver.solve(src_matched.T.astype(np.float64),
                           dst_matched.T.astype(np.float64))
    except Exception as e:
        print(f"teaser candidate errored ({e!r}); skipping.")
        return None
    T = np.eye(4)
    T[:3, :3] = np.asarray(sol.rotation)
    T[:3, 3] = np.asarray(sol.translation).ravel()
    if not np.isfinite(T).all():
        print("teaser candidate returned a non-finite transform; skipping.")
        return None
    return T


def _teaser_arbitrate(ref_mesh, test_mesh, T_ma, verbose=True,
                      seed=DEFAULT_SEED, noise_bound=None, area_referee=False):
    """Run the TEASER++ candidate and let the fine-DFT referee decide.

    Same contract as ``_goicp_arbitrate``: the incoming pose ``T_ma`` is
    already no-worse-than-identity; TEASER only wins by scoring strictly
    better on the same fine point-to-surface DFT, so this can never regress
    the result. Its wins are advisory cross-checks (it can land in wrong
    basins on near-symmetric geometry).

    ``area_referee=True`` (T1 J2, opt-in): the arbitration mean becomes the
    area-weighted G_hat (same queries, lumped weights).
    """
    if verbose:
        print("accuracy mode: running the TEASER++ candidate (certifiable TLS; "
              "arbitrated on the fine DFT, cannot regress).")
    T_te = _teaser_candidate(ref_mesh, test_mesh, noise_bound=noise_bound, seed=seed)
    if T_te is None:
        return T_ma

    _w = lumped_areas(ref_mesh) if area_referee else None

    def _score(T):
        m = copy.deepcopy(test_mesh)
        m.transform(T)
        errs, _ = compute_vertex_error_and_delta_nearest(ref_mesh, m)
        if _w is None:
            return float(np.mean(np.abs(errs)))
        return area_weighted_mean(errs, _w)

    s_ma, s_te = _score(T_ma), _score(T_te)
    if s_te < s_ma:
        if verbose:
            print(f"teaser candidate won arbitration "
                  f"(fine DFT {s_ma*1000:.1f} -> {s_te*1000:.1f} um) -- "
                  f"advisory cross-check; verify before trusting.")
        return T_te
    if verbose:
        print(f"teaser candidate rejected "
              f"(fine DFT {s_te*1000:.1f} >= {s_ma*1000:.1f} um); "
              f"keeping the tuned-solver pose.")
    return T_ma


def register(ref_mesh, test_mesh, n_samples=None, seed=None,
             yaw_inits_deg=(0.0, 90.0, 180.0, 270.0), verbose=True,
             mode=DEFAULT_MODE, **overrides):
    """Rigidly register ``test`` onto ``ref``. Returns the 4x4 transform to
    apply to the test geometry. Does NOT modify the inputs.

    ``ref``/``test`` may each be an Open3D TriangleMesh OR PointCloud, in any
    combination (see ``load_any`` for file loading). The referee adapts to the
    inputs (see ``_make_referee``): mesh-mesh arbitrates on exact ref-vertex ->
    test-surface distances (the tuned behavior, unchanged); a cloud test flips
    the metric to cloud-points -> ref-surface; cloud-vs-cloud uses a
    deterministic capped symmetric NN mean.

    Routes on the identity-fit mean point-to-surface distance: a pair already
    aligned within ``SKIP_THRESH_MM`` gets bounded LOCAL refinement
    (``align_local`` -- identity / 0deg / 180deg candidates arbitrated on the fine
    DFT, no flip risk); a far-off raw mesh gets the GLOBAL FPFH+RANSAC + yaw-seed
    search (``align_global``).

    ``mode`` ("fast"/"default"/"accuracy"): three real tiers. ``fast`` is the
    speed profile (~2x cheaper, accepted accuracy loss on hard parts:
    3k samples, 2 RANSAC attempts, 2-3-rung ladders, 30 ICP iters -- yaw
    seeds and the fine-DFT referee + identity bay are KEPT, so speed never
    buys pose-safety). ``default`` is the tuned profile (10k samples, 3
    attempts, accuracy/speed knee). ``accuracy`` additionally runs a wrapped
    Go-ICP (branch-and-bound) candidate and lets the fine-DFT referee
    decide. Go-ICP is PRONE TO WRONG POSES on near-symmetric geometry and can
    time out when branch-and-bound never prunes -- both are handled
    (arbitration + timeout), so accuracy can never regress a result, but
    treat Go-ICP wins as advisory cross-checks, not truth. See
    config.MODE_PRESETS.

    ``**overrides`` forwards to ``config.preset()`` to tweak individual knobs
    (``n_samples``, ``ransac_attempts``, ``local_ladder``, ``global_ladder``,
    ``icp_max_iter``, ``ladder_eps_mm``) without editing MODE_PRESETS --
    used by benchmark sweeps (benchmarks/tune_fast.py). An explicit
    ``n_samples=`` argument still wins.

    ``seed`` (when given) reseeds the global RNG so the sampling and RANSAC are
    deterministic for this call.

    T1 J2 arbitration knob (opt-in via ``**overrides``, DEFAULT OFF --
    ``area_referee=True``): the fine-arbitration statistic (S1 align_local,
    S3 audit arbiters, S4 fine re-arbitration, S5 CoarseFineAligner) switches
    to the area-uniform hybrid ``G_hat + arb_beta * CTE_arb_phi``. Zero RNG:
    candidate pools are byte-identical, only the arbitration weights change.
    Far-pair candidate ranking (S7) and the CLI base score (S8) are already
    area-uniform and untouched. Reporting (S6) is a SEPARATE knob
    (``pipeline.run_pair(area_weighted=...)``).
    """
    p = preset(mode, **overrides)
    _ar = bool(p.get("area_referee", False))
    _abeta = float(p.get("arb_beta", 0.25))
    _aphi = float(p.get("arb_phi", 0.01))
    rk, tk = _kind(ref_mesh), _kind(test_mesh)
    # Scale-relative thresholds (2026-09, idea #2): SKIP_THRESH_MM and the
    # ladder rungs are ABSOLUTE mm, tuned on the ~30-200 mm corpus -- a 10 mm
    # micro-part gets "local" routing for a gross misalignment (3 mm = 30% of
    # the part; measured: 2/3 seeds flip 180 deg) and rungs coarse relative
    # to its error scale. ``scale_relative`` re-expresses both as fractions of
    # the REFERENCE bbox diagonal L (fractions = the tuned absolutes at the
    # 100 mm reference scale), so routing and annealing track part size:
    #   skip: 3% of L     local: (4, 1.5, 0.5)% of L     global: (15, 5, 2, 0.5)% of L
    # Modes: True (always), "auto" (engage only when L is outside the tuned
    # regime [30, 400] mm -- in-regime behavior is measured identical to the
    # fixed values, so the auto guard is invisible on corpus parts), or
    # False (DEFAULT: never -- opt in explicitly via scale_relative=True/"auto").
    _sr_setting = p.get("scale_relative", False)
    skip_thresh = SKIP_THRESH_MM
    if _sr_setting is True or _sr_setting == "auto":
        # Scale basis = PCA-longest-extent (characteristic length):
        # rotation-INVARIANT (the bbox diagonal is not -- it varies ~18% with
        # part orientation on corpus parts, which would make the routing
        # threshold depend on how the part sits in its frame).
        L = characteristic_length_um(mesh=ref_mesh) / 1000.0
        _engage = True if _sr_setting is True else not (20.0 <= L <= 250.0)
        if _engage:
            # All thresholds become fractions of L (the tuned absolutes at
            # the 100 mm reference scale). Extends 2026-09 idea #2 from
            # routing-only to the FULL absolute-criteria sweep: ladder
            # early-exit eps and the metrics voxel pitch scale too, so exit
            # criteria and reported-grid resolution track part size.
            #   skip 3% | local (4, 1.5, 0.5)% | global (15, 5, 2, 0.5)%
            #   ladder_eps 3e-4 mm@100mm -> 3e-6 L | voxel pitch frac x L/100
            _f = L / 100.0
            skip_thresh = 0.03 * L
            p = dict(p)
            p["local_ladder"] = tuple(max(0.01, f * L) for f in (0.04, 0.015, 0.005))
            p["global_ladder"] = tuple(max(0.01, f * L) for f in (0.15, 0.05, 0.02, 0.005))
            if p.get("ladder_eps_mm") is not None:
                p["ladder_eps_mm"] = max(1e-7, p["ladder_eps_mm"] * _f)
            if p.get("voxel_pitch_mm") is not None:
                p["voxel_pitch_mm"] = max(1e-4, p["voxel_pitch_mm"] * _f)
    # Cloud TEST inputs get a RANSAC-attempt floor: estimated normals (clouds
    # without them) degrade FPFH slightly (measured fitness 0.92-0.98 vs 1.00
    # for mesh-sampled points), so per-attempt hit rate drops and all 3 tuned
    # attempts can miss on near-symmetric parts -- the flip then wins
    # arbitration (measured: thing-1992245 m/c 1/3 runs at attempts=3, 3/3 at
    # attempts=5). The referee itself is NOT the problem (it prefers the true
    # pose whenever a candidate reaches it). Mesh inputs keep the tuned
    # attempt count byte-for-byte.
    if _kind(test_mesh) == "cloud" and p.get("ransac_attempts") is not None:
        p["ransac_attempts"] = max(int(p["ransac_attempts"]), 5)
    n_samples = n_samples if n_samples is not None else p["n_samples"]
    mesh_mesh = (rk == "mesh" and tk == "mesh")
    src, dst = _icp_sample(ref_mesh, test_mesh, n_samples, seed)
    src_pts = np.asarray(src.points)
    ref_pts = np.asarray(dst.points)

    if mesh_mesh:
        # legacy path, byte-for-byte the tuned behavior (when the Tier-D knobs
        # are unset; they are opt-in and absent from every preset profile)
        test_scene = _test_scene(test_mesh)
        base_score = _mean_ref_to_test(test_scene, ref_pts, np.eye(4))
        _engine = p.get("refine_engine")
        _fgr = bool(p.get("fgr_seed", False))
        _sweep = bool(p.get("symmetry_sweep", False))
        _adapt = bool(p.get("adaptive_ladder", False))
        _gate_k = p.get("global_gate_k")
        if base_score < skip_thresh:
            T = align_local(ref_mesh, test_mesh, src, dst, base_score, verbose=verbose,
                            ladder=p["local_ladder"], max_iter=p["icp_max_iter"],
                            ladder_eps=p.get("ladder_eps_mm", 0.0),
                            refine_engine=_engine, adaptive_ladder=_adapt,
                            area_referee=_ar, arb_beta=_abeta, arb_phi=_aphi)
        else:
            T = align_global(src, dst, src_pts, ref_pts, base_score, test_scene,
                             yaw_inits_deg=yaw_inits_deg, verbose=verbose,
                             ransac_attempts=p["ransac_attempts"],
                             ladder=p["global_ladder"], max_iter=p["icp_max_iter"],
                             ladder_eps=p.get("ladder_eps_mm", 0.0),
                             refine_engine=_engine, fgr_seed=_fgr,
                             symmetry_sweep=_sweep, adaptive_ladder=_adapt,
                             gate_k=_gate_k,
                             fine_ref_pts=(np.asarray(ref_mesh.vertices)
                                           if p.get("fine_rearbitrate", False) else None),
                             fine_ref_weights=(lumped_areas(ref_mesh)
                                               if (p.get("fine_rearbitrate", False)
                                                   and _ar) else None))
    else:
        score_fn, base_score = _make_referee(ref_mesh, test_mesh, seed=seed,
                                              area_referee=_ar)
        _engine = p.get("refine_engine")
        _fgr = bool(p.get("fgr_seed", False))
        _sweep = bool(p.get("symmetry_sweep", False))
        _adapt = bool(p.get("adaptive_ladder", False))
        if base_score < skip_thresh:
            T = align_local(ref_mesh, test_mesh, src, dst, base_score, verbose=verbose,
                            ladder=p["local_ladder"], max_iter=p["icp_max_iter"],
                            ladder_eps=p.get("ladder_eps_mm", 0.0),
                            score_fn=score_fn, refine_engine=_engine,
                            adaptive_ladder=_adapt,
                            area_referee=_ar, arb_beta=_abeta, arb_phi=_aphi)
        else:
            T = align_global(src, dst, src_pts, ref_pts, base_score,
                             yaw_inits_deg=yaw_inits_deg, verbose=verbose,
                             ransac_attempts=p["ransac_attempts"],
                             ladder=p["global_ladder"], max_iter=p["icp_max_iter"],
                             ladder_eps=p.get("ladder_eps_mm", 0.0),
                             score_fn=score_fn, refine_engine=_engine,
                             fgr_seed=_fgr, symmetry_sweep=_sweep,
                             adaptive_ladder=_adapt)

    # accuracy mode: add the Go-ICP candidate and arbitrate on the fine DFT.
    # The tuned pose T is already no-worse-than-identity; Go-ICP can only win
    # by scoring strictly better, so this can never regress the result.
    # MESH INPUTS ONLY in this cut: the Go-ICP candidate samples surfaces; for
    # cloud inputs accuracy mode falls back to the tuned solver alone (same
    # as when the binding is missing).
    if p.get("goicp_refine") and mesh_mesh:
        T = _goicp_arbitrate(ref_mesh, test_mesh, T,
                             timeout_s=p.get("goicp_timeout_s", 600),
                             verbose=verbose, seed=seed if seed is not None else DEFAULT_SEED,
                             area_referee=_ar)
    # TEASER++ audit candidate (2026-09, idea #8): fast certifiable-TLS
    # cross-check, arbitrated on the same fine DFT. Chained AFTER Go-ICP so
    # each arbitration step only improves -- the no-regression guarantee is
    # preserved through the chain. Both are advisory cross-checks.
    if p.get("teaser_refine") and mesh_mesh:
        T = _teaser_arbitrate(ref_mesh, test_mesh, T, verbose=verbose,
                              seed=seed if seed is not None else DEFAULT_SEED,
                              noise_bound=p.get("teaser_noise_bound"),
                              area_referee=_ar)
    return T


# Backward-compatible alias (the breast-project name).
icp_register = register


def _default_prefix(ref_stl_path, test_stl_path):
    ref_name = os.path.splitext(os.path.basename(ref_stl_path))[0]
    test_name = os.path.splitext(os.path.basename(test_stl_path))[0]
    return f"{ref_name}_vs_{test_name}"


def align_local_to_reference(ref_stl_path, test_stl_path, out_dir, prefix=None,
                             seed=DEFAULT_SEED, save_stl=True):
    """Standalone LOCAL registration of a mesh to a reference.

    For inputs that are already close (within ~SKIP_THRESH_MM): tests no-ICP /
    0deg / 180deg candidates, keeps the lowest fine DFT. No RANSAC, no global
    search -- grossly-misaligned meshes need ``align_global_to_reference``.

    Persists the winning pose as ``{prefix}_aligned.stl`` when ``save_stl``.
    Returns ``(aligned_mesh, T, aligned_stl_path, scores)`` where ``scores`` maps
    each tested yaw-deg -> fine DFT MAE (plus ``None`` -> no-ICP baseline).
    """
    ref_mesh = load_ref_cached(ref_stl_path)
    test_mesh = load_mesh(test_stl_path)
    os.makedirs(out_dir, exist_ok=True)
    if prefix is None:
        prefix = _default_prefix(ref_stl_path, test_stl_path)

    src, dst = _icp_sample(ref_mesh, test_mesh, 25000, seed)
    src_pts = np.asarray(src.points)
    ref_pts = np.asarray(dst.points)
    test_scene = _test_scene(test_mesh)
    base_score = _mean_ref_to_test(test_scene, ref_pts, np.eye(4))

    best_T, scores = align_local(ref_mesh, test_mesh, src, dst, base_score,
                                 yaw_inits_deg=(0.0, 180.0),
                                 src_pts=src_pts, ref_pts=ref_pts,
                                 return_scores=True)

    aligned = copy.deepcopy(test_mesh)
    aligned.transform(best_T)
    aligned.compute_vertex_normals()
    aligned_stl_path = os.path.join(out_dir, f"{prefix}_aligned.stl")
    if save_stl:
        best_deg = min((k for k in scores if k is not None), key=lambda k: scores[k],
                       default=None)
        best_mae = scores[best_deg] if best_deg is not None else scores[None]
        o3d.io.write_triangle_mesh(aligned_stl_path, aligned)
        print(f"Local-aligned mesh saved to: {aligned_stl_path} "
              f"(best yaw {best_deg}deg, DFT {best_mae*1000:.1f} um, fit {base_score:.3f} mm)")
    print(f"  local scores (mae_um): identity={scores[None]*1000:.1f}  "
          + "  ".join(f"{d}deg={scores[d]*1000:.1f}" for d in (0.0, 180.0) if d in scores))
    return aligned, best_T, aligned_stl_path, scores


def align_global_to_reference(ref_stl_path, test_stl_path, out_dir, prefix=None,
                              seed=DEFAULT_SEED, yaw_inits_deg=GLOBAL_YAW_INITS_DEG,
                              save_stl=True):
    """Standalone GLOBAL registration for non-pre-aligned (raw) meshes.

    Pose-free FPFH+RANSAC + coarse-to-fine point-to-point ICP, with
    centroid-aligned yaw seeds. The transform is accepted only if it beats the
    identity fit. For meshes that are NOT pre-aligned; already-close meshes should
    use ``align_local_to_reference``.

    Persists ``{prefix}_aligned.stl`` when ``save_stl``. Returns
    ``(aligned_mesh, T, aligned_stl_path)``. Does NOT measure -- follow with
    ``pipeline.run_pair(ref, aligned_stl, ..., align=False)`` for the DFT.
    """
    ref_mesh = load_ref_cached(ref_stl_path)
    test_mesh = load_mesh(test_stl_path)
    os.makedirs(out_dir, exist_ok=True)
    if prefix is None:
        prefix = _default_prefix(ref_stl_path, test_stl_path)

    src, dst = _icp_sample(ref_mesh, test_mesh, 25000, seed)
    src_pts = np.asarray(src.points)
    ref_pts = np.asarray(dst.points)
    test_scene = _test_scene(test_mesh)
    base_score = _mean_ref_to_test(test_scene, ref_pts, np.eye(4))

    T = align_global(src, dst, src_pts, ref_pts, base_score, test_scene,
                     yaw_inits_deg=yaw_inits_deg)
    aligned = copy.deepcopy(test_mesh)
    aligned.transform(T)
    aligned.compute_vertex_normals()
    aligned_stl_path = os.path.join(out_dir, f"{prefix}_aligned.stl")
    if save_stl:
        o3d.io.write_triangle_mesh(aligned_stl_path, aligned)
        print(f"Global-aligned mesh saved to: {aligned_stl_path}")
    return aligned, T, aligned_stl_path


def _align_test(ref_mesh, test_mesh, seed, yaw_inits_deg, verbose=True, mode=DEFAULT_MODE):
    """Deep-copy test (mesh or cloud), rigidly register it onto ref, return (T, aligned).

    Thin shared helper for the coupled measurement path and the standalone align
    entry points. Re-seeds the global RNG inside ``register`` (via _icp_sample).
    Does not mutate ``test_mesh``. The returned geometry is a deep copy with
    normals recomputed (mesh) or estimated if missing (cloud).
    """
    tmesh = copy.deepcopy(test_mesh)
    T = register(ref_mesh, tmesh, seed=seed, yaw_inits_deg=yaw_inits_deg,
                 verbose=verbose, mode=mode)
    tmesh.transform(T)

    ##### CUSTOM VISUALIZATION #####
    plot_and_save_stage(tmesh, ref_mesh, 
                        "Stage: Post-Register Function", 
                        "mid-print_images/stage_post_register.png")
    ##########
    
    if isinstance(tmesh, o3d.geometry.TriangleMesh):
        tmesh.compute_vertex_normals()
    elif not tmesh.has_normals():
        tmesh.estimate_normals()
    return T, tmesh
