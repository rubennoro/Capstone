"""T2/T2b/T2c (D10b): exported poses equal the original ARKit poses up to one rigid world transform."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures_sim import *  # noqa: F401,F403
from sensor_rig.sim.compare import rel_pose_errors, world_transform_residual
from sensor_rig.transforms import T_YUP_ZUP, inv_T

pytestmark = pytest.mark.data

KS = [1, 6, 60, "all"]


def _check_rel(se, tol_rot, tol_trans):
    T_e = se.exported_poses()
    T_o = se.reference_poses()
    assert len(T_e) == len(T_o) > 60
    for k in KS:
        r, t = rel_pose_errors(T_e, T_o, k)
        assert r < tol_rot and t < tol_trans, f"{se.variant} k={k}: rot {r:.2e} rad, trans {t:.2e} m"


def test_T2_relative_poses_60fps(sim_export):
    se = sim_export("demo60")
    assert se.result.status == "exported", se.result
    assert se.result.frame_count == len(se.truth["t_ns"])  # nothing trimmed
    _check_rel(se, 1e-4, 1e-4)


def test_T2b_constant_world_transform_about_y(sim_export):
    se = sim_export("demo60")
    T_e, T_o = se.exported_poses(), se.reference_poses()
    D, rot, trans = world_transform_residual(T_e, T_o)
    assert rot < 1e-4 and trans < 1e-4
    D_i = T_e @ inv_T(T_o)
    assert np.abs(D_i - D_i[0]).max() < 1e-4
    assert np.abs(D[1, :3] - [0, 1, 0]).max() < 1e-4  # rotation about +y only (gravity preserved)
    # and it is exactly the composition the design predicts: T_off * A * C * A^-1
    C = se.truth["T_Wz_change"]
    D_pred = se.rig.export.T_world_offset @ T_YUP_ZUP @ C @ inv_T(T_YUP_ZUP)
    assert np.abs(D - D_pred).max() < 1e-4


def test_T2_1278_episode(sim_export):
    """Poses do not depend on the episode's camera K."""
    se = sim_export("demo1278")
    assert se.result.status == "exported", se.result
    _check_rel(se, 1e-4, 1e-4)


@pytest.mark.parametrize("variant,tol", [
    ("main30", 1e-4),            # fill grid; reference = originals interpolated at the 60 Hz grid times
    ("traj_ultrawide", 1e-4),    # trajectory of the ultrawide camera frame instead of the IMU
    ("traj_uw_rate", 2e-2),      # poses only at ~10 Hz: pure interpolation error of the real motion (see below)
    ("jitter", 2e-3),            # 0.5 ms stamp jitter -> poses sampled at slightly wrong times
])
def test_T2c_variants(sim_export, variant, tol):
    se = sim_export(variant)
    assert se.result.status == "exported", se.result
    if variant == "main30":
        assert se.result.grid_mode == "fill"
    _check_rel(se, tol, tol)


def test_T2c_uw_rate_equals_interpolated_originals(sim_export):
    """With ~10 Hz VIO poses the export must equal the originals sub-sampled at the same keys and interpolated
    (linear + SLERP) at the grid: the 1e-2-level error above is the motion itself, not the exporter."""
    from sensor_rig.pose.trajectory import Trajectory

    se = sim_export("traj_uw_rate")
    i0, n = se.kept_range()
    t = se.truth["t_ns"]
    key = se.src.uw_slots()
    key = key[key < len(t)]
    ref = Trajectory.from_T(t[key], se.truth["T_arkit"][key])
    T_ref, valid = ref.sample(se.grid_ns()[i0:i0 + n], max_gap_ns=10**12)
    assert valid.all()
    T_e = se.exported_poses()
    for k in KS:
        r, tr = rel_pose_errors(T_e, T_ref, k)
        assert r < 1e-4 and tr < 1e-4, f"k={k}: rot {r:.2e}, trans {tr:.2e}"
