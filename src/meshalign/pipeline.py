"""Pipeline orchestration: align + measure + write artifacts for one mesh/cloud pair."""
import os
import copy

import numpy as np
import open3d as o3d

from .config import (UM_PER_MM, DEFAULT_SEED, CLIM_MM, GLOBAL_YAW_INITS_DEG,
                     DEFAULT_MODE, AREA_WEIGHTED_REPORT, preset)
from .io import (set_global_seed, load_mesh, load_ref_cached, load_any,
                 characteristic_length_um)
from .measure import (compute_vertex_error_and_delta_nearest, compute_directional_dft_um,
                      point_error_signed, cloud_point_error, vertices_correspond,
                      lumped_areas, area_weighted_mean, area_weighted_quantile,
                      area_weighted_stats)
from .output import save_error_outputs
from .align import _align_test, _default_prefix
import os
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg') # Required for headless cloud environments
import matplotlib.pyplot as plt

def plot_and_save_stage(test, reference, title, filename):
    # 1. ADAPTIVELY MULTIPLY TARGET DENSITY
    # Check if it is a solid mesh with vertices. If so, double its native vertex count.
    if hasattr(reference, 'triangles') and hasattr(reference, 'vertices'):
        reference_doubled_count = int(len(reference.vertices) * 0.75)
        dense_reference = reference.sample_points_uniformly(number_of_points=reference_doubled_count)
        tgt_pts = np.asarray(dense_reference.points)
    else:
        # Fallback if reference is already a point cloud
        tgt_pts = np.asarray(reference.vertices if hasattr(reference, 'vertices') else reference.points)

    # 2. ADAPTIVELY MULTIPLY SOURCE DENSITY
    if hasattr(test, 'triangles') and hasattr(test, 'vertices'):
        test_doubled_count = int(len(test.vertices) * 0.7)
        dense_test = test.sample_points_uniformly(number_of_points=test_doubled_count)
        src_pts = np.asarray(dense_test.points)
    else:
        # Fallback if test is already a point cloud
        src_pts = np.asarray(test.vertices if hasattr(test, 'vertices') else test.points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Reference (Ground Truth) - Original blue, semi-transparent
    ax.scatter(tgt_pts[:, 0], tgt_pts[:, 1], tgt_pts[:, 2], 
               c='blue', s=1, alpha=0.5, label='Reference (Ground Truth)')
    
    # Test (Reconstruction) - Original red, highly opaque
    ax.scatter(src_pts[:, 0], src_pts[:, 1], src_pts[:, 2], 
               c='red', s=1, alpha=0.4, label='Test (Moving Mesh)')

    # Standardize axes to prevent visual stretching/distortion
    all_pts = np.vstack((src_pts, tgt_pts))
    max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), 
                          all_pts[:,1].max()-all_pts[:,1].min(), 
                          all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
    
    mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
    mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
    mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title(title)
    ax.legend(loc='upper right')
    
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _measure_once(ref, test, seed, scale, use_icp, mode=DEFAULT_MODE):
    """One full seeded align+measure pass for any mesh/cloud input combination.

    Works on a deep copy of ``test`` (ref and the caller's test are left
    untouched). Returns ``(errors_mm, delta_xyz_mm, aligned)`` already multiplied
    by ``scale``. ``use_icp=False`` measures as-is (no registration).

    Measurement direction by input kind:
      mesh/mesh  : ref vertices -> test surface, signed by ref vertex normals
                   (identical-vertex shortcut when counts match)
      mesh/cloud : test points -> ref surface, signed by ref face normals
      cloud/mesh : ref points  -> test surface, signed by test face normals
      cloud/cloud: test points -> nearest ref point, unsigned
    """
    set_global_seed(seed)
    tmesh = copy.deepcopy(test)
    T_used = np.eye(4)

    #Using ICP#
    if use_icp:
        T_used, tmesh = _align_test(ref, test, seed, GLOBAL_YAW_INITS_DEG, mode=mode)
        print(T_used)
    
    ref_is_mesh = isinstance(ref, o3d.geometry.TriangleMesh)
    test_is_mesh = isinstance(tmesh, o3d.geometry.TriangleMesh)

    if ref_is_mesh and test_is_mesh:
        # Identical-vertex shortcut -- ONLY when the final pose is near-identity
        # RELATIVE TO PART SIZE: equal vertex counts imply correspondence only
        # if the test was not re-posed (measured 2026-09: a rotationally-
        # symmetric part registers to a VALID but different pose -- e.g. any
        # sphere rotation -- and the shortcut then measures vertex-to-DIAMETER
        # distances, ~30 mm on a 50 mm sphere, instead of the surface error).
        # Guard: translation < 0.1% of the characteristic length (floored at
        # 0.01 mm for numerical sanity) and rotation < ~0.6 deg. Scale-free:
        # an absolute 0.1 mm would be 1% of a 10 mm micro part.
        _L_um = characteristic_length_um(mesh=ref) if ref_is_mesh else \
            characteristic_length_um(points=np.asarray(ref.points))
        _t_tol = max(_L_um * 1e-3 / 1000.0, 0.01)
        # |trace - 3| = 2(1 - cos th) ~ th^2: 1.2e-4 <=> th < ~0.6 deg
        _near_identity = (np.linalg.norm(T_used[:3, 3]) < _t_tol
                          and abs(np.trace(T_used[:3, :3]) - 3.0) < 1.2e-4)
        # Equal vertex counts only IMPLY correspondence if the test really is
        # a vertex-ordered copy of the ref; verify it instead of assuming
        # (two unrelated meshes decimated to the same count would otherwise
        # be scored vertex-to-vertex, which is meaningless).
        if (len(ref.vertices) == len(tmesh.vertices)) and _near_identity                 and vertices_correspond(ref, tmesh):
            # Identical vertex correspondence (e.g. same mesh decimated identically) —
            # cheaper and exact.
            ref_v = np.asarray(ref.vertices)
            test_v = np.asarray(tmesh.vertices)
            ref_normals = np.asarray(ref.vertex_normals)
            delta_xyz_mm = test_v - ref_v
            dist = np.linalg.norm(delta_xyz_mm, axis=1)
            sign = np.sign(np.sum(delta_xyz_mm * ref_normals, axis=1))
            sign[sign == 0.0] = 1.0
            errors_mm = dist * sign
            print("Using direct vertex-to-vertex signed distances.")
        else:
            errors_mm, delta_xyz_mm = compute_vertex_error_and_delta_nearest(ref, tmesh)
            print("Using nearest-neighbor signed distances (method=open3d_raycast).")
    elif ref_is_mesh and not test_is_mesh:
        errors_mm, delta_xyz_mm = point_error_signed(ref, tmesh)
        print("Using point-to-ref-surface signed distances (method=open3d_raycast, "
              "cloud test input).")
    elif test_is_mesh and not ref_is_mesh:
        # ref cloud points -> test surface, signed by test face normals
        errors_mm, delta_xyz_mm = point_error_signed(tmesh, ref)
        print("Using ref-point-to-test-surface signed distances (method=open3d_raycast, "
              "cloud reference input).")
    else:
        errors_mm, delta_xyz_mm = cloud_point_error(tmesh, ref, delta=True)
        print("Using point-to-point NN distances (cloud vs cloud; unsigned -- "
              "clouds carry no consistent orientation).")

    return errors_mm * scale, delta_xyz_mm * scale if delta_xyz_mm is not None else None, tmesh


def run_pair(ref_stl_path, test_stl_path, out_dir, scale=1.0,
             use_icp=False, prefix=None, seed=DEFAULT_SEED,
             histogram=None, save_per_sample=None, mode=DEFAULT_MODE,
             metrics=False, area_weighted=None):
    """Align (optional) + measure signed surface error for one ref/test mesh pair.

    Units: meshes in MILLIMETERS by lab convention; ``scale`` rescales the reported
    distances afterward (keep scale=1.0 so distances are µm when the mesh is mm).

    ``mode`` ("fast"/"default"/"accuracy") sets sampling/RANSAC/ICP budgets and
    the default ``histogram``/``save_per_sample`` tails (see config.MODE_PRESETS);
    explicit ``histogram=``/``save_per_sample=`` args override the mode default.
    ``metrics=True`` additionally computes the multi-metric bundle (Chamfer,
    Hausdorff95, normal consistency, voxel Dice/IoU in ``accuracy`` mode).

    ``area_weighted`` (T1, None = config.AREA_WEIGHTED_REPORT default = True):
    for MESH-MESH pairs the reported MAE/p95 are the area-uniform statistics
    (G_hat lumped re-weighting + area-weighted 95th quantile of the same
    per-vertex distances -- change the weights, not the queries) instead of
    the per-vertex mean/percentile. The per-vertex referee statistic is
    biased by -Cov_V(a_v, g_v)/mean(a_v) on non-uniform tessellations
    (docs/theory/t1-sampling-bias.md). Cloud inputs have no tessellation:
    always per-point. ``area_weighted=False`` recovers the per-vertex
    statistic exactly. The T1 diagnostics (E_vertex, E_area, bias,
    risk_bound, cv_area, kappa_tail) are appended to the CSV either way
    (blank when not applicable).

    Writes per pair:
      - ``{prefix}_surf_mae_um.csv``     -- headline MAE / p95 in microns
                                            (+ T1 diagnostics columns)
      - ``{prefix}_surf_dft_um.csv``     -- per-axis directional DFT summary
      - ``{prefix}_aligned.stl``         -- test mesh in the registered pose
      - ``{prefix}_surf_histogram.png``  -- if histogram=True
      - ``{prefix}_metrics.json``        -- if metrics=True
      - per-vertex dumps                 -- if save_per_sample=True

    Returns ``(mae_um, p95_um)``.
    """
    p = preset(mode)
    if histogram is None:
        histogram = p["histogram"]
    if save_per_sample is None:
        save_per_sample = p["save_per_sample"]
    print(f"\nLoading inputs...")
    print(f"  ref : {ref_stl_path}")
    print(f"  test: {test_stl_path}")
    ref_kind, ref = load_any(ref_stl_path)
    test_kind, test = load_any(test_stl_path)

    ##### CUSTOM VISUALIZATION ###
    os.makedirs("mid-print_images", exist_ok=True)
    
    # Note that we pass 'test' first (source/red), then 'ref' second (target/blue)
    plot_and_save_stage(test, ref, 
                        "Stage 1: Initial Misalignment", 
                        "mid-print_images/stage_1_raw.png")
    ##########
    
    os.makedirs(out_dir, exist_ok=True)

    if prefix is None:
        prefix = _default_prefix(ref_stl_path, test_stl_path)

    print(f"\nMeasuring (icp={use_icp}, seed={seed}, mode={mode})")
    print(f"  ref={ref_kind}, test={test_kind}")
    # Single deterministic measurement: with deterministic seeded align+measure the
    # per-seed spread is sub-micron -- averaging seeds only multiplies runtime.
    errors_mm, delta_xyz_mm, aligned = _measure_once(
        ref, test, seed, scale, use_icp, mode=mode)

    # T1 reporting statistic (docs/theory/t1-sampling-bias.md section 7):
    # mesh-mesh pairs report the AREA-UNIFORM MAE/p95 (G_hat + area-weighted
    # q95) computed from the same per-vertex distances -- the per-vertex mean
    # is biased by -Cov_V(a_v, g_v)/mean(a_v) on non-uniform tessellations
    # and is not comparable across tessellations of the same surface. The
    # weights are the ref mesh's lumped areas (scale-invariant in the mean,
    # so mixing scaled errors with raw-mesh areas is exact). Zero RNG.
    _aw = AREA_WEIGHTED_REPORT if area_weighted is None else bool(area_weighted)
    diag = None
    if _aw and ref_kind == "mesh" and test_kind == "mesh":
        _areas = lumped_areas(ref)
        mae_um = float(area_weighted_mean(errors_mm, _areas) * UM_PER_MM)
        p95_um = float(area_weighted_quantile(errors_mm, _areas, 0.95) * UM_PER_MM)
        # diagnostics (section 7.3): both statistics EXCLUDE unreferenced
        # vertices so `bias` is the pure weighting effect
        diag = area_weighted_stats(errors_mm, _areas)
    else:
        mae_um = float(np.mean(np.abs(errors_mm)) * UM_PER_MM)
        p95_um = float(np.percentile(np.abs(errors_mm), 95) * UM_PER_MM)
    # Relative headline (2026-09): the absolute um value scales with part
    # size, so it means nothing across orders of magnitude -- a 10 mm part
    # at 30 um and a 500 mm frame at 300 um can be the SAME quality. The
    # size-normalized percent-of-characteristic-length is the comparable
    # quantity. Both are reported; the relative number is the headline.
    _cl_um = (characteristic_length_um(mesh=ref) if ref_kind == "mesh"
              else characteristic_length_um(points=np.asarray(ref.points)))
    rel_mae_pct = mae_um / _cl_um * 100.0
    rel_p95_pct = p95_um / _cl_um * 100.0
    print(f"\n{'='*60}")
    print(f"Surface DFT = {mae_um:.2f} um  (p95 {p95_um:.2f} um)"
          + ("  [area-weighted]" if diag is not None else ""))
    if diag is not None:
        print(f"T1 referee diagnostics: E_vertex {diag['E_vertex']*UM_PER_MM:.2f} um  "
              f"E_area {diag['E_area']*UM_PER_MM:.2f} um  "
              f"bias {diag['bias']*UM_PER_MM:+.2f} um  "
              f"risk_bound {diag['risk_bound']*UM_PER_MM:.2f} um  "
              f"cv_area {diag['cv_area']:.3f}  kappa_tail {diag['kappa_tail']:.2f}")
    print(f"Size-normalized = {rel_mae_pct:.4f}% of characteristic length "
          f"({_cl_um/1000.0:.1f} mm)  [p95 {rel_p95_pct:.4f}%]")
    print(f"{'='*60}")

    # CSV schema: T1 diagnostics APPENDED (additive columns, parsers that
    # index by name keep working -- the 2026-09 relative-metrics precedent).
    # Blank cells when the statistics do not apply (cloud inputs).
    mae_path = os.path.join(out_dir, f"{prefix}_surf_mae_um.csv")
    with open(mae_path, "w") as _f:
        _f.write("mae_um,p95_um,char_length_um,rel_mae_pct,rel_p95_pct,"
                 "E_vertex_um,E_area_um,bias_um,risk_bound_um,cv_area,"
                 "kappa_tail,area_weighted\n")
        base = [f"{mae_um:.4f}", f"{p95_um:.4f}", f"{_cl_um:.4f}",
                f"{rel_mae_pct:.6f}", f"{rel_p95_pct:.6f}"]
        if diag is not None:
            _f.write(",".join(base + [
                f"{diag['E_vertex']*UM_PER_MM:.4f}",
                f"{diag['E_area']*UM_PER_MM:.4f}",
                f"{diag['bias']*UM_PER_MM:.4f}",
                f"{diag['risk_bound']*UM_PER_MM:.4f}",
                f"{diag['cv_area']:.6f}", f"{diag['kappa_tail']:.6f}",
                str(int(_aw))]) + "\n")
        else:
            # not applicable (cloud input): blanks, and the reported
            # statistic is per-point, not area-weighted
            _f.write(",".join(base + ["", "", "", "", "", "", "0"]) + "\n")

    dft_table = compute_directional_dft_um(delta_xyz_mm if delta_xyz_mm is not None
                                            else np.zeros((len(errors_mm), 3)))
    # max_abs for the histogram x-limits: auto from data, or CLIM_MM fixed.
    max_abs_err_mm = max(float(np.max(np.abs(errors_mm))), 1e-9) if CLIM_MM is None else float(CLIM_MM)
    save_error_outputs(ref, errors_mm, delta_xyz_mm, max_abs_err_mm, dft_table,
                       out_dir, prefix, histogram=histogram, save_per_sample=save_per_sample)

    if test_kind == "cloud":
        aligned_path = os.path.join(out_dir, f"{prefix}_aligned.ply")
        o3d.io.write_point_cloud(aligned_path, aligned)
        print(f"Aligned point cloud saved to: {aligned_path}")
    else:
        aligned_path = os.path.join(out_dir, f"{prefix}_aligned.stl")
        o3d.io.write_triangle_mesh(aligned_path, aligned)
        print(f"Aligned mesh saved to: {aligned_path}")

    if metrics:
        import json
        from .metrics_bundle import bundle as _bundle
        bundle = _bundle(ref, aligned, seed=seed, pitch_mm=p["voxel_pitch_mm"])
        with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w") as f:
            json.dump(bundle, f, indent=1)
        print(f"Metric bundle saved (chamfer={bundle['chamfer_mm']:.4f} mm, "
              f"hausdorff95={bundle['hausdorff95_um']:.1f} um, "
              f"normal_consistency={bundle['normal_consistency']:.4f}"
              + (", dice=%.4f, iou=%.4f" % (bundle["dice"], bundle["iou"])
                 if bundle.get("dice") is not None else ""))

    return mae_um, p95_um


def run_pair_relative(ref_stl_path, test_stl_path, out_dir, scale=1.0,
                      use_icp=True, prefix=None, seed=DEFAULT_SEED, histogram=None,
                      save_per_sample=None, mode=DEFAULT_MODE, metrics=False,
                      area_weighted=None):
    """``run_pair`` + size-normalized (relative) DFT reporting -- the recommended
    benchmark entry point.

    Registers via the routing aligner (``use_icp=True``): a mesh already close to
    the reference gets bounded LOCAL refinement with the 0deg/180deg disambiguation
    (+ identity baseline -- never regresses); a far-off mesh gets the global
    FPFH+RANSAC search. No scaling is fit -- a wrong-scale reconstruction is a
    genuine defect and must show up in the DFT.

    Then normalizes by the reference mesh's PCA-longest extent (characteristic
    length) and writes ``{prefix}_dft_summary.csv`` alongside the usual run_pair
    artifacts.

    Returns ``(mae_um, p95_um, char_length_um, rel_mae_pct, rel_p95_pct)``.
    """
    print(use_icp)
    mae_um, p95_um = run_pair(
        ref_stl_path, test_stl_path, out_dir, scale=scale,
        use_icp=use_icp, prefix=prefix, seed=seed,
        histogram=histogram, save_per_sample=save_per_sample, mode=mode,
        metrics=metrics, area_weighted=area_weighted)

    _rk, ref_obj = load_any(ref_stl_path)
    char_length_um = characteristic_length_um(mesh=ref_obj)
    rel_mae_pct = mae_um / char_length_um * 100.0
    rel_p95_pct = p95_um / char_length_um * 100.0

    if prefix is None:
        prefix = _default_prefix(ref_stl_path, test_stl_path)
    summary_path = os.path.join(out_dir, f"{prefix}_dft_summary.csv")
    with open(summary_path, "w") as f:
        f.write("mae_um,p95_um,char_length_um,rel_mae_pct,rel_p95_pct\n")
        f.write(f"{mae_um:.4f},{p95_um:.4f},{char_length_um:.4f},"
                f"{rel_mae_pct:.6f},{rel_p95_pct:.6f}\n")
    print(f"Characteristic length: {char_length_um:.1f} um")
    print(f"Relative Surface DFT: {rel_mae_pct:.3f}% (p95 {rel_p95_pct:.3f}%)")
    return mae_um, p95_um, char_length_um, rel_mae_pct, rel_p95_pct
