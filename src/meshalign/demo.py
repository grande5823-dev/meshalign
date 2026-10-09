"""Synthetic demo meshes (hemi-dome reference + transformed noisy copy).

Used by the tests and to populate ``data/demo/`` so examples run out of the box.
The reference is a base-down hemi-ellipsoid with a small asymmetric bump at one
tip -- the bump breaks the 180deg in-plane rotational symmetry so registration
tests can actually distinguish the two halves.
"""
import os

import numpy as np


def _require_trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError as e:
        raise ImportError("meshalign.demo requires trimesh (pip install meshalign[helpers])") from e


def make_bracket(L=40.0, W=25.0, H=12.0, thick=2.0):
    """A machined bracket shape, mm units — flat plate with a raised tab
    on one side. This is the shape the tests use for pose-recovery
    (asymmetric, non-degenerate, not a dome or sphere)."""
    trimesh = _require_trimesh()
    plate = trimesh.creation.box([L, W, thick])
    plate.apply_translation([0, 0, thick / 2])
    tab = trimesh.creation.box([L * 0.25, W * 0.5, H * 0.8])
    tab.apply_translation([L * 0.30, 0, thick + H * 0.4])
    bracket = trimesh.util.concatenate([plate, tab])
    return bracket


def rotz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def make_pair(yaw_deg=35.0, translate_mm=(6.0, -3.0, 1.5), seed=0):
    """Reference bracket + a rigidly-transformed test copy (no noise — the
    benchmark deliberately tests clean meshes first; add noise with --noise
    if you want to model sensor uncertainty).

    Returns ``(ref_trimesh, test_trimesh, T_applied)`` where ``T_applied`` is the
    4x4 transform that maps the (noisy) reference onto the test mesh -- i.e. the
    inverse is what registration should recover (up to Open3D's src->dst
    convention: register() returns T that maps test back onto ref, ~ T_applied^-1).
    """
    ref = make_bracket()
    test = ref.copy()
    T_applied = np.eye(4)
    T_applied[:3, :3] = rotz(yaw_deg)[:3, :3]
    T_applied[:3, 3] = translate_mm
    test.apply_transform(T_applied)
    return ref, test, T_applied


def write_demo(outdir="data/mech_parts", seed=0):
    """Write ``Connector.stl`` + ``Connector_test.stl`` into ``outdir``;
    returns the paths. These are the reference and perturbed test mesh used
    in the tests and the webapp demo."""
    trimesh = _require_trimesh()
    os.makedirs(outdir, exist_ok=True)
    ref, test, T = make_pair(seed=seed)
    ref_path = os.path.join(outdir, "Connector.stl")
    test_path = os.path.join(outdir, "Connector_test.stl")
    ref.export(ref_path)
    test.export(test_path)
    return ref_path, test_path, T
