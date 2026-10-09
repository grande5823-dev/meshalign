"""No-reference quality metrics: ensemble consistency of candidate meshes of the
SAME object when no ground truth exists. Requires pandas + scipy."""
import copy

import numpy as np


def _pick_medoid(mesh_paths, seed=0, sample=5000):
    """Hub = the mesh with the minimum TOTAL symmetric Chamfer distance to all
    others (true pairwise medoid). Requires scipy."""
    from .io import load_mesh
    from .measure import chamfer
    n = len(mesh_paths)
    meshes = [load_mesh(p) for p in mesh_paths]
    pts = []
    for m in meshes:
        import open3d as o3d
        o3d.utility.random.seed(seed)
        pts.append(np.asarray(m.sample_points_uniformly(number_of_points=sample).points, float))
    totals = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = chamfer(pts[i], pts[j], sample=sample, seed=seed)
            totals[i] += d
            totals[j] += d
    return int(np.argmin(totals))


def ensemble_consistency(mesh_paths, hub_index=None, seed=0, n_samples=25000,
                         area_weighted=None):
    """Register every candidate onto one hub candidate; report surface distance to
    the hub. For >= 2 meshes presumed to show the SAME physical subject (repeat
    captures, different pipelines, different frame subsets) when NO ground truth
    exists. A candidate close to the pack is corroborated by independent evidence;
    a candidate far from the pack (or a pack with high spread) is the no-reference
    answer to "which of these recons should I believe?"

    ``hub_index=None`` picks the hub as the pairwise-Chamfer MEDOID of the set
    (NOT index 0 -- an arbitrary hub silently inverts the signal; "close to the
    hub" is only "close to the truth" if the hub itself is trustworthy).

    ``area_weighted`` (T1, None = config.AREA_WEIGHTED_REPORT default = True):
    report the area-uniform G_hat over the hub's tessellation instead of the
    per-vertex mean (same per-vertex distances, lumped re-weighting).

    IMPORTANT: this is a proxy, not a validated accuracy measurement. Validate on a
    set that DOES have ground truth first (the proxy should correlate with true
    surface-error / volume error there) before trusting it clinically.

    Returns a pandas DataFrame; requires pandas.
    """
    import pandas as pd
    from .io import load_mesh
    from .align import register
    from .measure import (compute_vertex_error_and_delta_nearest, lumped_areas,
                          area_weighted_mean)
    from .config import AREA_WEIGHTED_REPORT
    if area_weighted is None:
        area_weighted = AREA_WEIGHTED_REPORT

    mesh_paths = [str(p) for p in mesh_paths]
    if len(mesh_paths) < 2:
        raise ValueError("ensemble_consistency needs at least 2 candidate meshes")
    if hub_index is None:
        hub_index = _pick_medoid(mesh_paths, seed=seed)

    hub_path = mesh_paths[hub_index]
    hub_mesh = load_mesh(hub_path)

    rows = []
    for i, p in enumerate(mesh_paths):
        if i == hub_index:
            rows.append({"mesh": p, "is_hub": True, "ensemble_dft_um": 0.0, "rotation_deg": 0.0})
            continue
        test_mesh = load_mesh(p)
        T = register(hub_mesh, test_mesh, n_samples=n_samples, seed=seed)
        aligned = copy.deepcopy(test_mesh)
        aligned.transform(T)
        aligned.compute_vertex_normals()
        errors, _ = compute_vertex_error_and_delta_nearest(hub_mesh, aligned)
        if area_weighted:
            # T1: area-uniform G_hat over the hub's tessellation (same
            # per-vertex distances, lumped weights -- comparable across hub
            # tessellations; see docs/theory/t1-sampling-bias.md section 7)
            mae_um = float(area_weighted_mean(errors, lumped_areas(hub_mesh)) * 1000.0)
        else:
            mae_um = float(np.mean(np.abs(errors)) * 1000.0)
        ang_deg = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
        rows.append({"mesh": p, "is_hub": False, "ensemble_dft_um": round(mae_um, 1),
                     "rotation_deg": round(ang_deg, 2)})

    df = pd.DataFrame(rows)
    non_hub = df.loc[~df["is_hub"], "ensemble_dft_um"]
    med = float(non_hub.median())
    mad = float((non_hub - med).abs().median()) * 1.4826  # normal-consistent MAD
    if mad > 0:
        df["ensemble_outlier"] = (~df["is_hub"]) & ((df["ensemble_dft_um"] - med).abs() > 3 * mad)
    else:
        df["ensemble_outlier"] = False
    df.attrs["hub"] = hub_path
    df.attrs["ensemble_median_um"] = med
    df.attrs["ensemble_mad_um"] = mad
    return df
