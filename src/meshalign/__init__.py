"""meshalign -- rigid mesh/point-cloud alignment and surface-error evaluation.

Built on Open3D's registration and raycasting primitives, with the policy layer
on top: deterministic seeding, exact point-to-triangle (BVH) scoring,
symmetry-safe candidate arbitration, and no-regression identity fallback.

Inputs may be meshes or point clouds, in any combination (``load_any`` sniffs
content). The referee adapts: mesh-mesh arbitrates on exact ref-vertex ->
test-surface distances; a cloud test flips the metric to cloud-points ->
ref-surface; cloud-vs-cloud uses a deterministic capped symmetric NN mean.

Quick start:
    import meshalign as ma

    # Full pipeline: align + measure + write artifacts (meshes OR clouds)
    mae_um, p95_um = ma.run_pair("gt.stl", "recon.ply", "out/", use_icp=True)

    # Alignment only
    ref_kind, ref = ma.load_any("gt.stl"); _, test = ma.load_any("scan.ply")
    T = ma.register(ref, test)
    test.transform(T)

    # Measurement only
    errs, delta_xyz = ma.surface_error(ref, test_aligned)      # mesh/mesh
    errs, delta_xyz = ma.point_error_signed(ref_mesh, cloud)    # mesh ref, cloud test
"""
from .config import DEFAULT_SEED, GLOBAL_YAW_INITS_DEG, UM_PER_MM, CLIM_UM
from .io import (set_global_seed, load_mesh, load_ref_cached, characteristic_length_um,
                 load_point_cloud, load_any, save_point_cloud)
from .measure import (
    compute_vertex_error_and_delta_nearest,
    surface_error,
    compute_directional_dft_um,
    chamfer,
    point_error_signed,
    cloud_point_error,
    cloud_cloud_error,
    lumped_areas,
    area_weighted_mean,
    area_weighted_quantile,
    area_weighted_cte,
    area_weighted_stats,
)
from .align import (
    register,
    icp_register,
    align_local,
    align_global,
    align_local_to_reference,
    align_global_to_reference,
    SKIP_THRESH_MM,
)
from .pipeline import run_pair, run_pair_relative
from .pose_error import pose_recovery_error, residual_transform, rotation_angle_deg
from .batch import run_batch
from .metrics_bundle import bundle, hausdorff, normal_consistency, voxel_dice_iou, gwn_iou, identity_control
from .config import MODE_PRESETS, DEFAULT_MODE, preset

__version__ = "0.3.0"

__all__ = [
    "DEFAULT_SEED", "GLOBAL_YAW_INITS_DEG", "UM_PER_MM", "CLIM_UM",
    "set_global_seed", "load_mesh", "load_ref_cached", "characteristic_length_um",
    "load_point_cloud", "load_any", "save_point_cloud",
    "surface_error", "compute_vertex_error_and_delta_nearest",
    "compute_directional_dft_um", "chamfer",
    "point_error_signed", "cloud_point_error", "cloud_cloud_error",
    "lumped_areas", "area_weighted_mean", "area_weighted_quantile",
    "area_weighted_cte", "area_weighted_stats",
    "register", "icp_register", "align_local", "align_global",
    "align_local_to_reference", "align_global_to_reference", "SKIP_THRESH_MM",
    "run_pair", "run_pair_relative", "run_batch",
    "bundle", "hausdorff", "normal_consistency", "voxel_dice_iou",
    "MODE_PRESETS", "DEFAULT_MODE", "preset",
]
