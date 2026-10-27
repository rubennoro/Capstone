"""W10B: the eval runner drops the launcher's default-station TCP fill, so every launcher child re-derives it from the
full station (user sets such as extension / side change the TCP)."""
import types

import numpy as np

from franka_deploy.launch import launcher as L
from franka_sim.eval import runner as rn


def _args(tmp_path, sets):
    return types.SimpleNamespace(deploy_config=None, profile="sim-iphone", policy="fake", set=sets, out=str(tmp_path),
                                 seed=0, sim_stub=None)


def test_runner_sets_carry_no_stale_tcp_and_the_child_derives_the_full_station(tmp_path):
    sets = ["tooling.finger_adapter.extension_m=0.10", 'tooling.tcp_x_along_hand="-y"']
    R = rn.resolve(_args(tmp_path, sets))
    assert not any(str(x).startswith("tooling.F_T_TCP=") for x in R.sets), R.sets
    child = L.load_config(rn.DEFAULT_DEPLOY, "sim-iphone", station_set=R.user_sets)   # what a launcher child sees
    T = np.asarray(child["_sim_tcp"]["F_T_TCP"])
    assert abs(T[2, 3] - 0.1973) < 1e-3 and abs(np.degrees(np.arctan2(T[1, 0], T[0, 0])) + 135.0) < 0.1
    ok, detail = L.sim_tcp_check(child, L.station_from_deploy_cfg(child))
    assert ok, detail


def test_explicit_user_tcp_is_kept(tmp_path):
    good = L.load_config(rn.DEFAULT_DEPLOY, "sim-iphone")["_sim_tcp"]["set"]
    R = rn.resolve(_args(tmp_path, [good]))
    assert R.sets.count(good) == 1
