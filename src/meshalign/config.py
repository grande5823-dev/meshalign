"""Shared constants and determinism knobs for meshalign.

Defaults assume meshes in MILLIMETERS (the lab convention). All thresholds
(mid/outer ICP ladders, identity-fit routing) are in those units.
"""

# --------------------------------------------------
# Fixed histogram x-limits in microns. None = auto-scale per pair.
CLIM_UM = None
CLIM_MM = CLIM_UM / 1000.0 if CLIM_UM is not None else None
# --------------------------------------------------

UM_PER_MM = 1000.0

# --------------------------------------------------
# Determinism controls.
#
# Everything here is driven by Open3D's single global RNG (sample_points_uniformly
# and registration_ransac_*) plus NumPy. set_global_seed() seeds both, so any
# single run is bit-for-bit reproducible given the same inputs. There is no
# repeat/averaging loop anywhere in the package: one deterministic seeded
# measurement per pair.
# --------------------------------------------------
DEFAULT_SEED = 0

# Centroid-aligned Z-rotation seeds (deg) tried alongside RANSAC in the GLOBAL
# registration path. The default is symmetric quarter-turns -- no dataset-specific
# magic angle. Override per-call via the yaw_inits_deg argument when your data has
# a known canonical frame already.
GLOBAL_YAW_INITS_DEG = (0.0, 90.0, 180.0, 270.0)

# --------------------------------------------------
# Mode presets: THE tuned profile + the speed tier + the audit profile.
#
# 2026-09 convergence, mesh-to-mesh scope (tune_fast sweep, 911 DB cells,
# 14 watertight parts x 3 seeds, 17 meshalign variants + 5 open-source
# baselines): the tuned profile tops the board at the accuracy/speed knee.
#
# Design notes (mesh-to-mesh scope; scan-vs-scan is out of scope for the
# tuned profile):
#   - fixes plain-fast's only mesh-to-mesh weakness: thing-1992245
#     0.941 -> 0.972 (that was a 5k-sample RANSAC issue, not the ladder)
#   - symmetric-dome grid (no-regression gate): passes (see
#     benchmark_results/synthetic_grid_symmetric.csv)
#
# What stayed from old-fast: 5k->10k samples (the measured knee for
# FPFH+RANSAC seed quality), 3-rung ladders, 100-iter cap. The old
# default's 5-rung/200-iter profile bought nothing on meshes and cost
# 3.3x. "default" is the canonical name of the tuned profile.
#
# Determinism envelope (VALIDATION.md #5, measured): run-to-run pose
# wobble under default Open3D threading is ~5 um MAE / ~6 um p95 on the
# dome pair (exactly 0 under OMP_NUM_THREADS=1). Same mechanism as the
# old fast mode (tolerance widened in 10264c5), slightly larger envelope
# -- still far below a real pose error (~10s of um).
# --------------------------------------------------
# Speed profile: ~2x faster than default on typical parts. Shares the
# same yaw seeds and fine-DFT referee + identity bay — speed never buys
# pose-safety.
# --------------------------------------------------
_FAST_PROFILE = dict(
    n_samples=3_000,
    ransac_attempts=2,
    local_ladder=(4.0, 1.0),
    global_ladder=(15.0, 3.0, 1.0),
    icp_max_iter=30,
    ladder_eps_mm=3e-4,
    histogram=False,
    save_per_sample=False,
    voxel_pitch_mm=0.5,
)

# --------------------------------------------------
_TUNED_PROFILE = dict(
    n_samples=10_000,
    ransac_attempts=3,
    local_ladder=(4.0, 1.5, 0.5),
    global_ladder=(15.0, 5.0, 2.0, 0.5),
    icp_max_iter=100,
    ladder_eps_mm=3e-4,   # 1e-3 let thread noise pick the stop rung (4+ um
                          # run-to-run MAE deltas); 3e-4 holds determinism
    histogram=True,
    save_per_sample=False,
    voxel_pitch_mm=0.25,
)

# --------------------------------------------------
# Profile tiers: "fast" is the speed profile, "default" is the tuned
# accuracy/speed knee, "accuracy" adds the Go-ICP audit candidate.
#
# Why 2 RANSAC attempts (not 1): with a single draw, hard near-symmetric
# parts land in near-tied arbitrations where thread wobble flips the basin.
# 2 draws keep the speed win and degrade gracefully. Restoring
# icp_max_iter=100 instead stabilizes fully but gives back the speedup.
#
# It keeps all 4 yaw seeds and the fine-DFT referee + identity bay --
# speed never buys pose-safety.
# --------------------------------------------------
MODE_PRESETS = {
    "fast": _FAST_PROFILE,    # speed tier: ~2x faster, accepted accuracy loss
    "default": _TUNED_PROFILE,   # the tuned profile (accuracy/speed knee)
    # Audit mode: the SAME tuned solver plus a Go-ICP (Yang et al., branch-
    # and-bound, "globally optimal in its bounded search space") candidate,
    # arbitrated on the fine point-to-surface DFT like every other candidate.
    # Go-ICP is a POINT-CLOUD method; STLs cross into its world through a
    # seeded area-uniform surface sample (5k pts, then down to n=250 for B&B).
    #
    # Trimming NOTE (measured, dome pair, 2026-09): trimFraction=0.2 per the
    # upstream README's outlier advice made the candidate WORSE (825 um vs
    # 248 um fine DFT) -- our pairs are fully overlapping surfaces, and
    # trimming discards real correspondences. Keep trimFraction=0.0 unless
    # registering partial-overlap scans.
    #
    # WARNING -- Go-ICP is PRONE TO WRONG POSES on near-symmetric geometry,
    # and on near-perfectly-symmetric parts branch-and-bound never prunes (hangs). The
    # fine-DFT referee + identity bay keep it from ever REGRESSING the
    # result, but treat its wins as advisory cross-checks, not truth.
    # Measured cost: 17 s (dome pair, n=250) to ~10 min (628k-vert bracket).
    # py_simple_go_icp_interface missing (e.g. CI) -> candidate is skipped
    # with a warning and accuracy falls back to the tuned solver alone.
    "accuracy": dict(_TUNED_PROFILE,
                     goicp_refine=True,
                     goicp_timeout_s=600,
                     teaser_refine=True,
                     symmetry_sweep=True,
                     save_per_sample=True,
                     voxel_pitch_mm=0.10),
}

DEFAULT_MODE = "default"


# --------------------------------------------------
# T1 area-uniform REPORTING default (docs/theory/t1-sampling-bias.md
# sections 7.1/7.3/7.4). For mesh-mesh pairs the reported headline MAE/p95
# are computed on the AREA measure (G_hat lumped re-weighting of the same
# per-vertex distances + an area-weighted p95 quantile) instead of the
# per-vertex mean: the per-vertex statistic is biased by
# -Cov_V(a_v, g_v)/mean(a_v) whenever the reference tessellation is
# non-uniform (measured gate-1 bias on a 10:1 graded tessellation: see
# docs/theory/impl-log.md 2026-09-18 T1 entry), and two tessellations of the
# SAME surface at the SAME pose reported different MAEs -- a comparability
# defect. Cloud inputs have no tessellation and always report per-point.
# Opt out per-call: run_pair(..., area_weighted=False) recovers the
# per-vertex statistic exactly. Arbitration (J2) is a SEPARATE knob
# (register(area_referee=True)) and is NOT governed by this default.
# --------------------------------------------------
AREA_WEIGHTED_REPORT = True


def preset(mode=DEFAULT_MODE, **overrides):
    """Resolve a mode preset dict; keyword overrides take precedence."""
    if mode not in MODE_PRESETS:
        raise ValueError(f"unknown mode {mode!r} (choose from {tuple(MODE_PRESETS)})")
    p = dict(MODE_PRESETS[mode])
    for k, v in overrides.items():
        if v is not None:
            p[k] = v
    return p
