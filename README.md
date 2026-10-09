# meshalign

Rigid mesh/point-cloud alignment and surface-error evaluation.
Built on Open3D's registration and raycasting primitives, with the policy
layer that calling Open3D naively doesn't give you:

- **Symmetry-safe registration** — bounded local ICP that cannot flip or tilt a near-symmetric shape into a wrong pose; explicit discrete yaw seeds (arbitrated on the real metric) for gross in-plane errors
- **No-regression guarantee** — the identity pose is always a candidate, so an already-correct mesh is never made worse
- **Determinism** — seeded Open3D RNG, exact point-to-triangle scoring via a BVH (`RaycastingScene`), not KD-trees
- **Metric-arbitrated candidates** — candidates are scored on the actual fine point-to-surface error you report, not on ICP's own fitness
- **Rigid only** — no scaling. A wrong-scale reconstruction is a genuine defect and must show up in the measurement
- **Meshes or point clouds, any combination** — STL/OBJ meshes and PLY/PCD/XYZ clouds register against each other; the referee adapts to the inputs

## Install

```bash
pip install "git+https://github.com/wpklab/meshalign.git"
# with extras (batch runner, metrics bundle, demo meshes):
pip install "git+https://github.com/wpklab/meshalign.git#egg=meshalign[all]"
```

Requires Python 3.11+, `open3d`, `numpy`, `matplotlib`.

## Quick start

Most common job: align a reconstruction against a ground-truth mesh, then
report the signed per-vertex surface error.

```python
import meshalign as ma

# one shot: align + measure + write CSVs/histogram/aligned STL
mae_um, p95_um = ma.run_pair_relative(
    "ground_truth.stl", "recon.stl", "out/", use_icp=True, histogram=True)

# the test side may be a raw scan cloud (PLY/PCD/XYZ):
mae_um, p95_um = ma.run_pair_relative(
    "ground_truth.stl", "scan.ply", "out/", use_icp=True)
```

Or step by step:

```python
import meshalign as ma
import numpy as np

ref = ma.load_mesh("ground_truth.stl")
test = ma.load_mesh("recon.stl")

T = ma.register(ref, test)           # rigid 4x4, routes local/global automatically
test.transform(T)

errs, delta_xyz = ma.surface_error(ref, test)   # signed, per ref vertex, mm
mae_um = float(np.mean(np.abs(errs))) * 1000
```

## CLI

```bash
meshalign run      gt.stl recon.stl --out out/ --relative --histogram
meshalign register gt.stl recon.stl -o aligned.stl            # auto local/global
meshalign register gt.stl recon.stl --global                  # force global RANSAC
meshalign measure  gt.stl aligned.stl --out out/              # no registration
meshalign chamfer  a.stl b.stl                                # symmetric Chamfer
meshalign batch    --refs ref_dir/ --tests test_dir/ --out results/  # parallel
```

## How the aligner routes

`register()` routes on the identity fit automatically:

- **already close** (< 3 mm mean point-to-surface) → bounded local ICP (coarse-to-fine ladder, 0°/180° yaw candidates, identity bay — cannot regress)
- **far off / raw** → global FPFH+RANSAC with seeded attempts + centroid-aligned yaw seeds at 0°/90°/180°/270°, then ICP refinement

## Modes

| Setting | `default` (tuned) | `fast` | `accuracy` |
|---|---|---|---|
| ICP sample points | 10k | 3k | 10k |
| RANSAC attempts | 3 | 2 | 3 |
| ICP ladder | coarse-to-fine, 3 rungs | 2 rungs | 3 rungs |
| Go-ICP audit candidate | — | — | yes (timeout 600 s) |
| voxel Dice/IoU pitch | 0.25 mm | 0.5 mm | 0.10 mm |

Use `default` for all mesh-to-mesh work. Use `fast` when speed matters more
than the last few microns on hard parts. Use `accuracy` only when you need
the Go-ICP cross-check on record — it is slower and the Go-ICP candidate is
prone to wrong poses on near-symmetric geometry (arbitrated so it can never
regress a result, but treat its wins as advisory).

**Scan-vs-scan pairs** (not mesh-vs-CAD): the tuned profile under-converges
there. Override:
```python
ma.register(ref, test, local_ladder=(4,3,2,1,0.5), icp_max_iter=200,
            global_ladder=(15,5,2,1,0.5))
```

## Metrics

```python
import meshalign as ma
errs, delta = ma.surface_error(ref, aligned)         # signed per-vertex, mm
bundle = ma.bundle(ref, aligned, pitch_mm=0.25)       # multi-metric bundle
iou, dice = ma.voxel_dice_iou(ref, aligned)           # voxel IoU (watertight)
gwn_iou, _, se = ma.gwn_iou(ref, aligned)             # GWN IoU (any mesh)
pe = ma.pose_recovery_error(verts, T_align, T_applied) # pose-recovery error
```

| Metric | What it measures | Best for |
|---|---|---|
| **Surface DFT** (default headline) | signed exact point-to-triangle distance, per ref vertex (µm) | accuracy vs ground truth |
| **Voxel IoU** | volume overlap on a voxel grid | watertight meshes, pose sanity |
| **GWN IoU** | Monte-Carlo volume overlap, works on any mesh | open meshes, point clouds |
| **Chamfer** | symmetric mean NN distance (mm) | cross-paper comparison |
| **Hausdorff p95/max** | worst-tail distance | spike/void detection |
| **Pose-recovery error** | displacement between estimated and ground-truth transform (mm + °) | benchmarking alignment quality |

## Web app (alignment console)

A browser GUI for running alignments and inspecting results — 3D error maps,
histograms, metric cards, provenance, artifact downloads.

First time setup (installs the package + webapp dependencies):

```bash
install.bat         # Windows (Miniconda/Anaconda)
./install.sh        # Linux / macOS (conda or venv)
```

Start the webapp:

```bash
./start.sh          # Linux / macOS
start.bat          # Windows
```

Then open http://localhost:8766 in a browser. Pick a reference and test mesh
(from the bundled demo or upload your own), press Run, and inspect the result.

**All alignments are saved** to `webapp/runs/` — each run has its own directory
with the aligned STL, error CSVs, histogram, metrics JSON, and viewer files.
The run history panel in the webapp lists every past run; click any run to
reload its results without re-running it. Runs survive server restarts.

Custom port: `./start.sh 9000` (Linux) or `start.bat 9000` (Windows).

See `webapp/README.md` for the full endpoint list, configuration, and tests.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

```
src/meshalign/
  config.py     constants, mode presets, determinism knobs
  io.py         mesh loading, seeding, PCA characteristic length
  align.py      register / align_local / align_global
  measure.py    surface_error, directional DFT, chamfer, area weighting
  metrics_bundle.py  multi-metric bundle (Chamfer, IoU, Dice, Hausdorff, GWN)
  pose_error.py pose-recovery error metric
  pipeline.py   run_pair, run_pair_relative (one-shot align + measure + write)
  batch.py      multiprocess batch runner
  output.py     histogram + CSV writers
  helpers.py    yaw_align, pick_flip, render views
  demo.py       synthetic demo meshes
  cli.py        command-line interface
webapp/         browser GUI (FastAPI + three.js)
data/mech_parts/      example dome pair for testing
tests/          pytest suite (synthetic fixtures, no external data needed)
examples/       runnable end-to-end example
```

## Papers this is grounded in

- Rusinkiewicz & Levoy, *Efficient Variants of the ICP Algorithm* (3DIM 2001)
- Yang et al., *Go-ICP* (ICCV 2013)
- Zhou et al., *Fast Global Registration* (ECCV 2016)
- Cignoni, Rocchini & Scopigno, *Metro* (EG 1998)
- Taha & Hanbury, *Metrics for evaluating 3D medical image segmentation* (2015)

## Gotchas baked into the defaults

1. **Point-to-point only, never point-to-plane.** Point-to-plane lets correspondences slide along the surface tangent and settles in tilted or flipped local minima on symmetric shapes.
2. **A yaw-90° error is invisible to ICP on a near-circular shape.** Discrete yaw seeds + arbitration on the fine metric handles it.
3. **A free global search on an already-close pair is dangerous.** Always compare against the identity baseline.
4. **Center by bounding box, not vertex mean.** Two meshes at different vertex densities have biased centroids.
5. **Rigid only.** A scale fit would hide a wrong-size reconstruction.
6. **KD-tree distances drift between processes.** BVH (`RaycastingScene`) queries are exact and deterministic.

## License

MIT
