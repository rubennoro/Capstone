"""B9 finding 3 (fixed at W10): the Stage 2b / 3 launcher children get the run's station-group ids, so a sweep or
run_overrides `robot: fr3` runs the child on fr3 (it used to fall back to the profile's panda)."""
import types

from franka_deploy.launch import launcher as L
from franka_sim.eval import runner as rn
from franka_sim.eval import stage2b as s2b
from franka_sim.eval import stage3 as s3


def _args(tmp_path):
    return types.SimpleNamespace(deploy_config=None, profile="sim-iphone", policy="fake", set=[], out=str(tmp_path),
                                 seed=0, sim_stub=None)


def _child_robot(dep, profile):
    return L.load_config(str(dep), profile)["station"]["robot"]


def test_stage3_child_keeps_robot_override(tmp_path):
    R = rn.resolve(_args(tmp_path), run_overrides={"robot": "fr3"})
    assert R.ids["robot"] == "fr3"
    dep = s3.Driver(R, tmp_path, {}, 0, 10.0).derive(None)
    assert _child_robot(dep, R.profile) == "fr3"


def test_stage2b_child_keeps_robot_override(tmp_path):
    R = s2b.replay_run(rn.resolve(_args(tmp_path), run_overrides={"robot": "fr3"}))
    work = tmp_path / "demo_00" / "work"
    work.mkdir(parents=True)
    dep, prof = s2b.derive_deploy(R, 0, {"name": "x"}, work, tmp_path / "gt.npz", 0)
    assert prof["station"]["robot"] == "fr3" and _child_robot(dep, R.profile) == "fr3"


def test_replay_run_keeps_overrides_and_extra_sets(tmp_path):
    R = rn.resolve(_args(tmp_path), run_overrides={"robot": "fr3", "link": "wifi_bad"},
                   extra_sets=["tooling.finger_adapter.extension_m=0.1"])
    R2 = s2b.replay_run(R)
    assert R2.image_source in ("video_replay", "rig_raw_replay")
    assert R2.ids["robot"] == "fr3" and R2.run_ids["link"] == "wifi_bad" and R2.run_ids["sim"] != R.run_ids["sim"]
    assert "tooling.finger_adapter.extension_m=0.1" in R2.sets
