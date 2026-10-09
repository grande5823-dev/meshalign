"""Multi-metric evaluation bundle: Chamfer, Hausdorff, normal consistency,
voxel Dice/IoU.

No single surface metric measures both COVERAGE (how much of the true shape the
recon captures) and ACCURACY (how close its surface is where it exists):
point-to-surface MAE handles accuracy but is coverage-blind, Chamfer handles
both coarsely, Hausdorff95 catches thin spikes, normal consistency catches
orientation flips at correct distance, and volume Dice/IoU catches internal
voids/over-inflation invisible from the outside.

Standard references:
  * Chamfer / Hausdorff: Du et al., Point Set Registration Metrics
    Reloaded for Computer-Assisted Surgery (2025); Cignoni, Rocchini & Scopigno,
    "Metro" (1998).
  * Normal consistency: Mescheder et al. Occupancy Networks (2019).
  * Voxel Dice/IoU: Taha & Hanbury, "Metrics for evaluating 3D medical image
    segmentation" (2015).

All functions take aligned Open3D meshes in millimeters; reported distances in mm.
"""
import numpy as np
import open3d as o3d

from .measure import chamfer, _to_points


def _scene(mesh):
    t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(t)
    return s


def _nearest_info(query_pts, other_mesh):
    """Exact nearest point on ``other_mesh`` + its face index, for each query point."""
    scene = _scene(other_mesh)
    q = o3d.core.Tensor(np.asarray(query_pts, np.float32))
    ans = scene.compute_closest_points(q)
    return ans["points"].numpy().astype(np.float64), ans["primitive_ids"].numpy()


def _sample_points(obj, n_samples, seed=0):
    """Seeded surface points from a mesh (area-uniform) or cloud (subsample)."""
    o3d.utility.random.seed(seed)
    if isinstance(obj, o3d.geometry.PointCloud):
        pts = np.asarray(obj.points)
        if len(pts) > n_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(pts), n_samples, replace=False)
            idx.sort()
            return pts[idx]
        return pts
    return np.asarray(obj.sample_points_uniformly(number_of_points=n_samples).points)


def hausdorff(mesh_a, mesh_b, n_samples=5000, seed=0, two_way=True):
    """Hausdorff distances (symmetric) between two aligned meshes/clouds.

    Returns ``(max_mm, p95_mm)`` -- the max and 95th percentile of BOTH
    directions combined. p95 is the recommended headline (max is noise-prone).
    Point-cloud inputs are supported: distances are then point-to-surface when
    the other side is a mesh (exact) or point-to-point when both are clouds.
    """
    a = _sample_points(mesh_a, n_samples, seed)
    b = _sample_points(mesh_b, n_samples, seed)
    if isinstance(mesh_b, o3d.geometry.TriangleMesh):
        na, _ = _nearest_info(a, mesh_b)
        d_a2b = np.linalg.norm(a - na, axis=1)
    else:
        from scipy.spatial import cKDTree
        d_a2b, _ = cKDTree(_sample_points(mesh_b, n_samples, seed)).query(a, k=1)
    d = [d_a2b]
    if two_way:
        if isinstance(mesh_a, o3d.geometry.TriangleMesh):
            nb, _ = _nearest_info(b, mesh_a)
            d_b2a = np.linalg.norm(b - nb, axis=1)
        else:
            from scipy.spatial import cKDTree
            d_b2a, _ = cKDTree(a).query(b, k=1)
        d.append(d_b2a)
    d = np.concatenate(d)
    return float(d.max()), float(np.percentile(d, 95))


def normal_consistency(mesh_a, mesh_b, n_samples=5000, seed=0):
    """Mean absolute cosine of normals at mutually closest points (0..1, 1=perfect).

    Sample points on A; for each, take A's sampled-point normal (nearest-vertex
    approximation) and B's face normal at the exact closest triangle. Catches
    orientation/shading flips invisible to distance metrics: a dented or
    inverted-patch region can sit at correct distance but with flipped normals.
    Requires scipy (nearest-vertex lookup on A).
    """
    o3d.utility.random.seed(seed)
    if not mesh_a.has_vertex_normals():
        mesh_a.compute_vertex_normals()
    if not mesh_b.has_triangle_normals():
        mesh_b.compute_triangle_normals()
    va = np.asarray(mesh_a.vertices)
    na = np.asarray(mesh_a.vertex_normals)
    pa = np.asarray(mesh_a.sample_points_uniformly(number_of_points=n_samples).points)
    _, tri_ids = _nearest_info(pa, mesh_b)
    face_n = np.asarray(mesh_b.triangle_normals)[tri_ids]
    from scipy.spatial import cKDTree
    _, idx = cKDTree(va).query(pa, k=1)
    npa = na[idx]
    cos = np.abs(np.einsum("ij,ij->i", npa, face_n))
    return float(np.mean(cos))


def voxel_dice_iou(mesh_a, mesh_b, pitch_mm=0.25, max_cells=4_000_000):
    """Volume Dice and IoU on voxelized interiors (requires watertight meshes).

    Both meshes must be CLOSED/watertight -- open surfaces (e.g. base-cut domes)
    produce meaningless inside/outside tests and this will return (nan, nan).
    Uses Open3D RaycastingScene.compute_occupancy on a regular grid of spacing
    ``pitch_mm`` covering the union bbox + one pitch margin. If the grid exceeds
    ``max_cells`` the pitch is thickened until it fits (a fitted pitch, not the
    caller's, is used) -- the caller should not expect a specific pitch.
    """
    scene_a = _scene(mesh_a)
    scene_b = _scene(mesh_b)
    fitted_pitch = pitch_mm
    while True:
        mins = np.minimum(np.asarray(mesh_a.get_min_bound()),
                          np.asarray(mesh_b.get_min_bound())) - fitted_pitch
        maxs = np.maximum(np.asarray(mesh_a.get_max_bound()),
                          np.asarray(mesh_b.get_max_bound())) + fitted_pitch
        nx = int(np.ceil((maxs[0] - mins[0]) / fitted_pitch))
        ny = int(np.ceil((maxs[1] - mins[1]) / fitted_pitch))
        nz = int(np.ceil((maxs[2] - mins[2]) / fitted_pitch))
        n_cells = nx * ny * nz
        if n_cells <= max_cells or fitted_pitch > 100.0:
            break
        # scale pitch so the total cells drop under the cap; keep cubic cells
        fitted_pitch = fitted_pitch * (n_cells / max_cells) ** (1.0 / 3.0) * 1.05
    if fitted_pitch > pitch_mm:
        print(f"[meshalign] voxel pitch raised from {pitch_mm} to {fitted_pitch:.3f} mm "
              f"to fit the {max_cells} cell cap")
    xs = np.arange(mins[0], maxs[0] + fitted_pitch, fitted_pitch)
    ys = np.arange(mins[1], maxs[1] + fitted_pitch, fitted_pitch)
    zs = np.arange(mins[2], maxs[2] + fitted_pitch, fitted_pitch)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    qt = o3d.core.Tensor(pts)
    oa = scene_a.compute_occupancy(qt).numpy().astype(bool)
    ob = scene_b.compute_occupancy(qt).numpy().astype(bool)
    if not oa.any() or not ob.any():
        return float("nan"), float("nan")
    inter = np.logical_and(oa, ob).sum()
    union = np.logical_or(oa, ob).sum()
    dice = 2.0 * inter / (oa.sum() + ob.sum())
    iou = inter / union
    return float(dice), float(iou)


def bundle(ref_mesh, test_mesh, seed=0, n_samples=5000, pitch_mm=None):
    """One shot multi-metric evaluation. Returns a dict of floats.

    ``pitch_mm`` None skips voxel metrics (Dice/IoU returned as None). All inputs
    must already be aligned (the caller's ``run_pair`` output, typically).
    Meshes or point clouds: Chamfer and Hausdorff work for any combination;
    normal consistency needs normals on both sides (meshes, or clouds that
    carry them) and is skipped otherwise; voxel Dice/IoU need watertight
    MESHES on both sides and are skipped for cloud inputs.
    """
    out = {
        "chamfer_mm": round(chamfer(ref_mesh, test_mesh, sample=n_samples, seed=seed), 6),
    }
    hmax, h95 = hausdorff(ref_mesh, test_mesh, n_samples=n_samples, seed=seed)
    out["hausdorff_max_um"] = round(hmax * 1000.0, 3)
    out["hausdorff95_um"] = round(h95 * 1000.0, 3)
    try:
        out["normal_consistency"] = round(normal_consistency(ref_mesh, test_mesh,
                                                              n_samples=n_samples, seed=seed), 6)
    except (AttributeError, TypeError):
        print("[meshalign] normal consistency skipped: needs normals on both "
              "sides (meshes, or clouds that carry them)")
        out["normal_consistency"] = None
    if pitch_mm is not None:
        both_mesh = (isinstance(ref_mesh, o3d.geometry.TriangleMesh)
                     and isinstance(test_mesh, o3d.geometry.TriangleMesh))
        if not both_mesh:
            print("[meshalign] voxel metrics skipped: voxel Dice/IoU need "
                  "watertight meshes on both sides")
            dice = iou = None
        else:
            try:
                dice, iou = voxel_dice_iou(ref_mesh, test_mesh, pitch_mm=pitch_mm)
            except ValueError as e:
                print(f"[meshalign] voxel metrics skipped: {e}")
                dice = iou = None
    else:
        dice = iou = None
    out["dice"] = None if dice is None else round(dice, 6)
    out["iou"] = None if iou is None else round(iou, 6)
    return out

def gwn_iou(mesh_a, mesh_b, n_samples=40000, seed=0):
    """Volume IoU via Monte-Carlo occupancy with generalized winding number.

    Works on ANY mesh (watertight, open, self-intersecting, multi-body)
    because the GWN inside test is well-defined for all of them.
    Falls back to vote3 parity when libigl is not installed.

    Returns (iou, dice, stderr).
    """
    import numpy as np
    va = np.asarray(mesh_a.vertices)
    fa = np.asarray(mesh_a.triangles)
    vb = np.asarray(mesh_b.vertices)
    fb = np.asarray(mesh_b.triangles)

    # sampling box: union of both bboxes + 2% per-axis pad
    lo = np.minimum(va.min(0), vb.min(0))
    hi = np.maximum(va.max(0), vb.max(0))
    ext = hi - lo
    pad = np.where(ext > 0, 0.02 * ext, 0.02 * ext.max())
    lo, hi = lo - pad, hi + pad

    pts = lo + np.random.default_rng(seed).random((n_samples, 3)) * (hi - lo)

    try:
        import igl
        wa = np.asarray(igl.winding_number(
            np.ascontiguousarray(va, np.float64),
            np.ascontiguousarray(fa, np.int64),
            np.ascontiguousarray(pts, np.float64))).ravel()
        wb = np.asarray(igl.winding_number(
            np.ascontiguousarray(vb, np.float64),
            np.ascontiguousarray(fb, np.int64),
            np.ascontiguousarray(pts, np.float64))).ravel()
        in_a = wa > 0.5
        in_b = wb > 0.5
    except ImportError:
        # fallback: three-axis parity vote (requires watertight)
        in_a = _vote3_inside(mesh_a, pts)
        in_b = _vote3_inside(mesh_b, pts)

    inter = int((in_a & in_b).sum())
    union = int((in_a | in_b).sum())
    if union == 0:
        return float("nan"), float("nan"), 0.0
    iou = inter / union
    dice = 2.0 * inter / (in_a.sum() + in_b.sum()) if (in_a.sum() + in_b.sum()) > 0 else float("nan")
    stderr = float(np.sqrt(iou * (1 - iou) / union))
    return float(iou), float(dice), stderr


def _vote3_inside(mesh, pts):
    """Three-axis parity vote inside test (from the labmate's pipeline).
    Deterministic; requires watertight meshes."""
    import open3d as o3d
    tm = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
        o3d.core.Tensor(np.asarray(mesh.triangles, np.uint32)))
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(tm)
    n = len(pts)
    rays = np.zeros((n, 6), np.float32)
    rays[:, :3] = pts
    votes = np.zeros(n, dtype=np.int8)
    for ax in range(3):
        rays[:, 3 + ax] = 1.0
        cnt = sc.count_intersections(
            o3d.core.Tensor(rays, dtype=o3d.core.float32)).numpy()
        votes += (cnt % 2).astype(np.int8)
        rays[:, 3 + ax] = 0.0
    return votes >= 2


def identity_control(mesh, metric_fn=None, **kwargs):
    """Self-vs-self check: any scorer must return IoU = 1.0 on a byte copy.

    The labmate's protocol rule 1: "Score every part against a copy of itself
    first; byte copy must give exactly 1.0." Catches estimator bugs before
    they poison benchmark results.

    Returns (passed, iou, detail).
    """
    import copy
    mesh_copy = copy.deepcopy(mesh)
    if metric_fn is None:
        metric_fn = voxel_dice_iou
    try:
        if metric_fn is voxel_dice_iou:
            dice, iou = metric_fn(mesh, mesh_copy, **kwargs)
        else:
            iou = metric_fn(mesh, mesh_copy, **kwargs)
            if isinstance(iou, tuple):
                iou = iou[0]
        passed = abs(iou - 1.0) < 0.001
        return passed, iou, f"identity IoU = {iou:.6f}"
    except Exception as e:
        return False, None, f"identity check errored: {e}"
