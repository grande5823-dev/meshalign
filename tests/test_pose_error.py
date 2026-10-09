"""Tests for pose_recovery_error — pure geometry, no meshalign dependency."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from meshalign.pose_error import (pose_recovery_error, residual_transform,
                                   rotation_angle_deg, decompose_rigid)


def rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


VERTS = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10],
                  [5, 5, 5], [-5, -5, -5]], dtype=float)


class TestPerfectRecovery:
    def test_identity(self):
        T = np.eye(4)
        r = pose_recovery_error(VERTS, T, T)
        assert r["pose_err_mean_mm"] == 0.0
        assert r["pose_err_max_mm"] == 0.0
        assert r["pose_rot_deg"] == 0.0
        assert r["pose_trans_mm"] == 0.0

    def test_known_transform_exact_inverse(self):
        T_applied = make_T(rot_z(30), [5, -3, 2])
        T_align = np.linalg.inv(T_applied)
        r = pose_recovery_error(VERTS, T_align, T_applied)
        assert r["pose_err_mean_mm"] < 1e-10
        assert r["pose_rot_deg"] < 1e-10
        assert r["pose_trans_mm"] < 1e-10


class TestKnownErrors:
    def test_pure_rotation_1deg(self):
        T_applied = make_T(rot_z(30), [5, -3, 2])
        T_align = np.linalg.inv(T_applied)
        # add 1-degree residual rotation
        T_residual = make_T(rot_y(1.0), [0, 0, 0])
        T_align_imperfect = T_residual @ T_align
        r = pose_recovery_error(VERTS, T_align_imperfect, T_applied)
        assert r["pose_rot_deg"] == pytest.approx(1.0, abs=0.01)
        # max displacement: vertices at distance 10 from origin, 1 deg rotation
        # max disp = 2 * 10 * sin(0.5 deg) = 0.1745 mm
        assert r["pose_err_max_mm"] == pytest.approx(2 * 10 * np.sin(np.radians(0.5)), abs=0.001)

    def test_pure_translation(self):
        T_applied = make_T(rot_z(30), [5, -3, 2])
        T_align = np.linalg.inv(T_applied)
        T_align[0, 3] += 0.5  # 0.5 mm error in x
        r = pose_recovery_error(VERTS, T_align, T_applied)
        assert r["pose_trans_mm"] == pytest.approx(0.5, abs=0.01)
        assert r["pose_err_mean_mm"] == pytest.approx(0.5, abs=0.01)
        assert r["pose_err_max_mm"] == pytest.approx(0.5, abs=0.01)

    def test_combined_rotation_translation(self):
        T_applied = make_T(rot_z(45), [1, 2, 3])
        T_align = np.linalg.inv(T_applied)
        T_align = make_T(rot_y(2.0), [0.1, 0, 0]) @ T_align
        r = pose_recovery_error(VERTS, T_align, T_applied)
        assert r["pose_rot_deg"] == pytest.approx(2.0, abs=0.01)
        assert r["pose_trans_mm"] > 0


class TestResidualTransform:
    def test_identity_when_perfect(self):
        T = make_T(rot_z(90), [1, 2, 3])
        T_inv = np.linalg.inv(T)
        res = residual_transform(T_inv, T)
        assert np.allclose(res, np.eye(4), atol=1e-10)

    def test_nonidentity_residual(self):
        T = make_T(rot_z(90), [1, 2, 3])
        T_err = make_T(rot_z(5), [0, 0, 0])
        res = residual_transform(T_err @ np.linalg.inv(T), T)
        assert not np.allclose(res, np.eye(4))
        assert rotation_angle_deg(res[:3, :3]) == pytest.approx(5.0, abs=0.01)


class TestDecomposeRigid:
    def test_round_trip(self):
        R = rot_z(45)
        t = np.array([1, 2, 3])
        T = make_T(R, t)
        R_out, t_out = decompose_rigid(T)
        assert np.allclose(R_out, R)
        assert np.allclose(t_out, t)


class TestRotationAngle:
    def test_known_angles(self):
        assert rotation_angle_deg(np.eye(3)) == pytest.approx(0.0)
        assert rotation_angle_deg(rot_z(90)) == pytest.approx(90.0)
        assert rotation_angle_deg(rot_z(180)) == pytest.approx(180.0)
        assert rotation_angle_deg(rot_z(45)) == pytest.approx(45.0)
