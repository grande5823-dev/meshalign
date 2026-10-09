"""Tests for gwn_iou (generalized winding number IoU) and identity_control.

The GWN IoU handles non-watertight meshes that voxel_dice_iou cannot.
The identity control enforces protocol rule 1: self-vs-self must give 1.0.
"""
import os
import sys

import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from meshalign.metrics_bundle import gwn_iou, identity_control, voxel_dice_iou


def _sphere(radius=20.0, res=30):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=res)
    m.compute_vertex_normals()
    return m


def _box(extents=(10, 10, 10)):
    m = o3d.geometry.TriangleMesh.create_box(*extents)
    m.compute_vertex_normals()
    return m


class TestGwnIou:
    def test_identity_gives_one(self):
        s = _sphere()
        iou, dice, se = gwn_iou(s, s, n_samples=10000, seed=0)
        assert iou == pytest.approx(1.0, abs=0.001)
        assert dice == pytest.approx(1.0, abs=0.001)

    def test_disjoint_gives_zero(self):
        a = _sphere(radius=10)
        b = _sphere(radius=10)
        b.translate([100, 0, 0])  # far apart
        iou, dice, se = gwn_iou(a, b, n_samples=10000, seed=0)
        assert iou == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        a = _sphere(radius=10)
        b = _sphere(radius=10)
        b.translate([5, 0, 0])  # half overlapping
        iou, dice, se = gwn_iou(a, b, n_samples=20000, seed=0)
        assert 0.1 < iou < 0.9  # definitely not 0 or 1

    def test_agrees_with_voxel_iou_on_watertight(self):
        """GWN and voxel IoU should agree on watertight meshes."""
        s = _sphere(radius=20)
        t = _sphere(radius=20)
        t.translate([1.0, 0, 0])
        iou_gwn, _, _ = gwn_iou(s, t, n_samples=40000, seed=0)
        _, iou_vox = voxel_dice_iou(s, t, pitch_mm=1.0)
        # both should be in the same ballpark (within 0.05)
        assert abs(iou_gwn - iou_vox) < 0.05

    def test_open_mesh_does_not_crash(self):
        """GWN should handle an open surface (half a sphere's faces removed)."""
        s = _sphere(radius=20, res=20)
        f = np.asarray(s.triangles)
        # remove half the faces to make it open
        open_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(s.vertices)),
            o3d.utility.Vector3iVector(f[:len(f)//2]))
        open_mesh.compute_vertex_normals()
        t = o3d.geometry.TriangleMesh(open_mesh)
        t.translate([0.5, 0, 0])
        iou, dice, se = gwn_iou(open_mesh, t, n_samples=5000, seed=0)
        assert not np.isnan(iou)  # should produce a number, not crash

    def test_deterministic(self):
        s = _sphere(radius=10)
        t = _sphere(radius=10)
        t.translate([2, 0, 0])
        iou1, _, _ = gwn_iou(s, t, n_samples=10000, seed=42)
        iou2, _, _ = gwn_iou(s, t, n_samples=10000, seed=42)
        assert iou1 == iou2


class TestIdentityControl:
    def test_sphere_identity_passes(self):
        s = _sphere()
        passed, iou, detail = identity_control(s)
        assert passed
        assert iou == pytest.approx(1.0, abs=0.001)

    def test_box_identity_passes(self):
        b = _box()
        passed, iou, detail = identity_control(b)
        assert passed
        assert iou == pytest.approx(1.0, abs=0.001)

    def test_gwn_identity_passes(self):
        s = _sphere()
        passed, iou, detail = identity_control(s, metric_fn=gwn_iou,
                                                 n_samples=10000, seed=0)
        assert passed
        assert iou == pytest.approx(1.0, abs=0.001)

    def test_broken_scorer_fails(self):
        """A deliberately broken scorer should fail the identity control."""
        def broken_scorer(a, b, **kw):
            return (0.5, 0.5)  # wrong answer
        s = _sphere()
        passed, iou, detail = identity_control(s, metric_fn=broken_scorer)
        assert not passed
