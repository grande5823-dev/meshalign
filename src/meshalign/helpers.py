"""Higher-level registration helpers built on trimesh meshes (optional deps:
trimesh + scipy). These are the pre-alignment orientation tools -- brute-but-safe
yaw searching, flip disambiguation, bounded local ICP -- that run before the core
``align.register`` when the input mesh has no known pose.

All functions take and return trimesh meshes, never mutate inputs.
"""
import numpy as np


def _load_trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError as e:
        raise ImportError(
            "meshalign.helpers requires trimesh (pip install meshalign[helpers])") from e


def _center(m):
    """Recenter by BBOX center (xy) + base at z=0, NOT vertex mean.

    Two meshes of the same object at different vertex densities (e.g. a 69k-vertex
    laser scan vs an 8.5k-vertex recon) have different centroids; mean-centering
    offsets them laterally. Bbox center depends only on the extremes, so it matches.
    """
    v = np.asarray(m.vertices)
    cx = (v[:, 0].min() + v[:, 0].max()) / 2.0
    cy = (v[:, 1].min() + v[:, 1].max()) / 2.0
    m.apply_translation([-cx, -cy, -v[:, 2].min()])
    return m


def _rotz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def min_rect_angle(pts):
    """Rotation (radians) + long_is_x flag of the min-area bounding rectangle of 2D
    points. Deterministic long-axis estimate -- the thing that reliably breaks a
    near-circular shape's rotational symmetry, where registration metrics are
    ill-conditioned (aspect near 1.0 gives ICP/Chamfer no signal)."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return 0.0, True
    h = pts[ConvexHull(pts).vertices]
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        a = -np.arctan2(e[1], e[0])
        c, s = np.cos(a), np.sin(a)
        r = h @ np.array([[c, -s], [s, c]]).T
        w, hh = np.ptp(r[:, 0]), np.ptp(r[:, 1])
        if best is None or w * hh < best[0]:
            best = (w * hh, a, w >= hh)
    return best[1], best[2]


def orient_long_axis_y(mesh, base_frac=0.2):
    """Rotate a base-down mesh so its footprint's long axis (min-area rectangle)
    lies along Y, then re-center. No 180deg disambiguation -- see
    ``disambiguate_tip``."""
    m = mesh.copy()
    v = np.asarray(m.vertices)
    z0 = v[:, 2].min()
    h = v[:, 2].max() - z0
    base = v[v[:, 2] <= z0 + base_frac * h][:, :2]
    ang, long_is_x = min_rect_angle(base)
    if long_is_x:
        ang += np.pi / 2
    m.apply_transform(_rotz(np.degrees(ang)))
    return _center(m)


def disambiguate_tip(mesh, band=0.2, thr_mm=8.0):
    """180deg in-plane disambiguation by tip width: the narrower end of the
    footprint points to +Y ends as the narrower of the two tips. Rotates 180deg
    when needed, then re-centers."""
    m = mesh.copy()
    v = np.asarray(m.vertices)
    z0 = v[:, 2].min()
    h = v[:, 2].max() - z0
    fp = v[v[:, 2] <= z0 + band * h, :2]
    fy = fp[:, 1]
    top_tip = fp[fy >= fy.max() - thr_mm]
    bot_tip = fp[fy <= fy.min() + thr_mm]
    if len(top_tip) > 2 and len(bot_tip) > 2:
        w_top = float(top_tip[:, 0].max() - top_tip[:, 0].min())
        w_bot = float(bot_tip[:, 0].max() - bot_tip[:, 0].min())
        if w_top < w_bot:
            m.apply_transform(_rotz(180))
    else:
        if float(v[:, 1].mean()) >= -0.5:
            m.apply_transform(_rotz(180))
    return _center(m)


align_canonical = disambiguate_tip  # breast-project name kept as alias


def pick_flip(test, ref, n=15000):
    """Resolve a remaining 180deg in-plane ambiguity between two trimesh meshes by
    lower mean ref->test nearest-neighbor distance. Returns the better-oriented
    copy of ``test``."""
    from scipy.spatial import cKDTree
    rp = ref.sample(n)

    def fit(m):
        return float(np.mean(cKDTree(np.asarray(m.sample(n))).query(rp, k=1)[0]))

    t1 = test.copy()
    t1.apply_transform(_rotz(180))
    _center(t1)
    return test if fit(test) <= fit(t1) else t1


def local_icp(test, ref, n=40000):
    """LOCAL point-to-point ICP starting from an already-yaw-aligned pose,
    coarse->fine. Corrects residual tilt between the meshes without leaving the
    starting basin (bounded thresholds, capped low).

    Returns ``(aligned_mesh, rotation_angle_deg)``. trimesh in/out."""
    import open3d as o3d
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(np.asarray(test.sample(n)))
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(np.asarray(ref.sample(n)))
    T = np.eye(4)
    for th in (4.0, 3.0, 2.0, 1.0, 0.5):    # mm; capped low: from a close start it stays local
        T = o3d.pipelines.registration.registration_icp(
            src, dst, th, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)).transformation
    out = test.copy()
    out.apply_transform(T)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    return out, ang


def yaw_align(test, ref, coarse=3.0, fine=0.5, n=15000):
    """Resolve the one remaining DOF of an already base-down mesh: in-plane yaw.
    Brute-scan the Z rotation minimizing mean ref->test NN distance, coarse then
    fine. Physically constrained to yaw, so it CANNOT flip/tilt the shape.

    Returns ``(rotated_mesh, best_yaw_deg, best_score)``. trimesh in/out."""
    from scipy.spatial import cKDTree
    rp = ref.sample(n)

    def score(deg):
        tc = test.copy()
        tc.apply_transform(_rotz(deg))
        _center(tc)
        d, _ = cKDTree(np.asarray(tc.sample(n))).query(rp, k=1)
        return float(np.mean(d))

    best = min(np.arange(0, 360, coarse), key=score)
    best = min(np.arange(best - coarse, best + coarse + 1e-9, fine), key=score)
    out = test.copy()
    out.apply_transform(_rotz(best))
    _center(out)
    return out, float(best), score(best)


def render_views(mesh, out_prefix, max_faces=60000, dpi=110):
    """Render 6 standard orthographic views (top/bottom/front/back/left/right)
    to ``{out_prefix}_{view}.png`` via matplotlib Poly3DCollection (headless-safe).
    Returns the list of paths. trimesh or any .vertices/.faces object in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    if len(f) > max_faces:
        idx = np.random.default_rng(0).choice(len(f), max_faces, replace=False)
        f = f[idx]
    tris = v[f]
    lims = [(v[:, i].min(), v[:, i].max()) for i in range(3)]
    spans = [hi - lo for lo, hi in lims]

    views = [("top", 89, -90), ("bottom", -89, -90), ("front", 5, -90),
             ("back", 5, 90), ("left", 5, 180), ("right", 5, 0)]
    paths = []
    for name, elev, azim in views:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        pc = Poly3DCollection(tris, facecolor="tan", edgecolor="dimgray",
                              linewidth=0.05, alpha=1.0)
        ax.add_collection3d(pc)
        for i, set_lim in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            set_lim(*lims[i])
        ax.set_box_aspect(spans)
        ax.view_init(elev=elev, azim=azim)
        # No axes/ticks/grid: tick-label clutter reads as floating artifacts.
        ax.set_axis_off()
        fig.suptitle(name, y=0.9, fontsize=14)
        out = f"{out_prefix}_{name}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths
