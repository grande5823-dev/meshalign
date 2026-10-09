"""Mesh/point-cloud I/O, caching, and RNG seeding."""
import os

import numpy as np
import open3d as o3d


def set_global_seed(seed: int) -> None:
    """Seed every RNG meshalign touches.

    Open3D exposes a single global engine (``o3d.utility.random.seed``) that drives
    both ``sample_points_uniformly`` and ``registration_ransac_*`` -- neither accepts
    a per-call seed in this build. NumPy is seeded too for incidental draws.
    """
    o3d.utility.random.seed(int(seed))
    np.random.seed(int(seed))


def load_mesh(path):
    """Load a mesh from disk, cleaned and with vertex normals.

    Accepts any format Open3D reads (stl/ply/obj/...). Raises ValueError on empty.
    """
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Could not load mesh from {path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


CLOUD_EXTENSIONS = {".pcd", ".xyz", ".pts", ".asc", ".txt"}


def load_point_cloud(path, estimate_normals=False):
    """Load a point cloud from disk (ply/pcd/xyz/pts/asc).

    Reading is plain file IO -> deterministic. Normals are NOT estimated by
    default: candidate generation (FPFH in the global path) estimates them when
    needed; the referee never needs cloud-side normals (signs come from the
    mesh side when a surface exists, and cloud-vs-cloud distances are
    unsigned). Raises ValueError on empty.
    """
    p = str(path)
    if os.path.splitext(p)[1].lower() in (".asc", ".txt"):
        # Open3D has no reader for these: parse whitespace/comma text, XYZ = first 3 columns
        try:
            arr = np.loadtxt(p, delimiter=None if "," not in open(p).readline() else ",",
                             usecols=(0, 1, 2), ndmin=2, comments="#")
        except (ValueError, IndexError, OSError) as e:
            raise ValueError(f"Could not load a point cloud from {path}: {e}") from e
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr))
    else:
        cloud = o3d.io.read_point_cloud(p)
    if cloud.is_empty() and p.lower().endswith(".ply"):
        # a PLY with faces only: read_triangle_mesh got the vertices as points?
        # o3d falls back silently; try the mesh reader and take its vertices.
        try:
            m = o3d.io.read_triangle_mesh(p)
            if not m.is_empty():
                cloud = o3d.geometry.PointCloud(m.vertices)
        except Exception:
            pass
    if cloud.is_empty():
        raise ValueError(f"Could not load a point cloud from {path}")
    # deterministic cleanup: exact-duplicate points only (order-preserving)
    pts = np.asarray(cloud.points)
    _, keep = np.unique(np.round(pts, 9), axis=0, return_index=True)
    keep = np.sort(keep)
    if len(keep) < len(pts):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep]))
    if estimate_normals and not cloud.has_normals():
        cloud.estimate_normals()
    return cloud


def load_any(path, estimate_normals=False):
    """Load a mesh OR a point cloud, sniffing content (not just extension).

    Returns ``(kind, obj)`` with ``kind in {"mesh", "cloud"}``:
      - .stl/.obj and any file Open3D reads as triangles -> cleaned TriangleMesh
      - .pcd/.xyz/.pts/.asc and point-only PLYs -> PointCloud
    A .ply holding faces loads as a mesh even if it also carries points.
    """
    p = str(path)
    ext = os.path.splitext(p)[1].lower()
    if ext in CLOUD_EXTENSIONS:
        return "cloud", load_point_cloud(p, estimate_normals=estimate_normals)
    _prev = o3d.utility.get_verbosity_level()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    try:
        mesh = o3d.io.read_triangle_mesh(p)
        if not mesh.is_empty() and len(mesh.triangles) > 0:
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
            return "mesh", mesh
        cloud = o3d.io.read_point_cloud(p)
        if not cloud.is_empty():
            return "cloud", load_point_cloud(p, estimate_normals=estimate_normals)
    finally:
        o3d.utility.set_verbosity_level(_prev)
    raise ValueError(f"Could not load mesh or point cloud from {path}")


def save_point_cloud(cloud, path):
    """Write a point cloud (ply/pcd/xyz by extension)."""
    ok = o3d.io.write_point_cloud(str(path), cloud)
    if not ok:
        raise ValueError(f"Could not write point cloud to {path}")


# Cache loaded+processed reference meshes by path. A batch run measures many test
# meshes against the same few references; without this each pair re-reads and
# re-normals the reference. References are only read (sampled/vertices/normals), so
# sharing the cached object is safe.
_REF_CACHE = {}


def load_ref_cached(path):
    """Load a reference mesh, cached by path for batch reuse."""
    m = _REF_CACHE.get(str(path))
    if m is None:
        m = load_mesh(path)
        _REF_CACHE[str(path)] = m
    return m


_REF_CLOUD_CACHE = {}


def load_ref_cloud_cached(path):
    """Load a reference point cloud, cached by path for batch reuse."""
    c = _REF_CLOUD_CACHE.get(str(path))
    if c is None:
        c = load_point_cloud(path)
        _REF_CLOUD_CACHE[str(path)] = c
    return c


def characteristic_length_um(mesh_path=None, mesh=None, points=None):
    """PCA-longest-extent characteristic length in microns.

    The size normalizer for relative (size-normalized) surface-error reporting:
    project the reference geometry onto its PCA axes, take the longest span,
    convert mm -> um.

    Rotation-invariant and -- unlike a bbox axis extent -- insensitive to the
    geometry's coordinate frame. Works for meshes (``mesh=``/``mesh_path=``) and
    point clouds (``points=`` as an (N,3) array, or a cloud passed via ``mesh=``).

    Pass exactly one of ``mesh_path`` (uses the reference cache when already
    loaded), ``mesh`` (in-memory Open3D mesh or point cloud), or ``points``.
    """
    if points is None:
        if mesh is None:
            if mesh_path is None:
                raise ValueError("pass mesh_path=, mesh=, or points=")
            mesh = load_ref_cached(mesh_path)
        pts = np.asarray(mesh.vertices if hasattr(mesh, "vertices") else mesh.points)
    else:
        pts = np.asarray(points)
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt.T
    spans = proj.max(axis=0) - proj.min(axis=0)
    return float(spans.max()) * 1000.0  # mm -> um
