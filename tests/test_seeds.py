"""Verify that all randomness in the meshalign package is seeded.

Greps the source for unseeded random calls and fails if any are found.
Also verifies that set_global_seed actually makes Open3D sampling
deterministic across calls.
"""
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src", "meshalign")


class TestSeededRandomness:
    """Every random call in src/ must use a seed (either explicit or via
    set_global_seed). This test greps for violations."""

    def _find_unseeded(self):
        violations = []
        for fn in os.listdir(SRC_DIR):
            if not fn.endswith(".py") or fn == "__pycache__":
                continue
            path = os.path.join(SRC_DIR, fn)
            for i, line in enumerate(open(path), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # np.random.randn / rand / randint / uniform / normal / choice without rng
                if re.search(r'np\.random\.(randn|rand|randint|uniform|normal|choice)\(', stripped):
                    # check if it's on an rng object (rng = default_rng(seed))
                    if "default_rng" in stripped or "rng." in stripped:
                        continue
                    violations.append(f"{fn}:{i}: {stripped}")
                # random.random() / random.uniform() from stdlib
                if re.search(r'(?<!np\.)random\.(random|uniform|randint)\(', stripped):
                    if "default_rng" in stripped or "rng." in stripped:
                        continue
                    violations.append(f"{fn}:{i}: {stripped}")
        return violations

    def test_no_unseeded_random_calls(self):
        violations = self._find_unseeded()
        assert violations == [], \
            f"Unseeded random calls found:\n" + "\n".join(violations)

    def test_open3d_sampling_deterministic(self):
        """Open3D's sample_points_uniformly must be reproducible given the
        same seed via o3d.utility.random.seed()."""
        import open3d as o3d
        import meshalign

        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=10.0, resolution=20)
        mesh.compute_vertex_normals()

        meshalign.set_global_seed(42)
        pts1 = np.asarray(mesh.sample_points_uniformly(number_of_points=100).points)

        meshalign.set_global_seed(42)
        pts2 = np.asarray(mesh.sample_points_uniformly(number_of_points=100).points)

        meshalign.set_global_seed(7)
        pts3 = np.asarray(mesh.sample_points_uniformly(number_of_points=100).points)

        assert np.allclose(pts1, pts2), "same seed should give identical samples"
        assert not np.allclose(pts1, pts3), "different seed should give different samples"

    def test_default_rng_deterministic(self):
        """np.random.default_rng with a fixed seed must be reproducible."""
        r1 = np.random.default_rng(0)
        v1 = r1.normal(0, 1, (10, 3))

        r2 = np.random.default_rng(0)
        v2 = r2.normal(0, 1, (10, 3))

        assert np.allclose(v1, v2)

    def test_chamfer_deterministic(self):
        """The chamfer metric uses seeded sampling — must be reproducible."""
        import open3d as o3d
        import meshalign

        m = o3d.geometry.TriangleMesh.create_sphere(radius=10.0, resolution=20)
        m.compute_vertex_normals()
        m2 = o3d.geometry.TriangleMesh(m)
        m2.translate([0.5, 0, 0])

        c1 = meshalign.chamfer(m, m2, sample=500, seed=42)
        c2 = meshalign.chamfer(m, m2, sample=500, seed=42)
        assert c1 == c2, f"chamfer not deterministic: {c1} != {c2}"
