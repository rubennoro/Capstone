"""B10: the sim profiles no longer hardcode scratch paths; load_config fills work_root / python / checkpoint /
prompt_zarr / disk_budget_gb from the station `host` group when the PROFILE does not set them.

The golden files in tests/golden/host_paths/ are the resolved configs (Launcher(cfg).cfg dumped as deploy_resolved.yaml)
captured BEFORE the change, with the pc_laptop host values replaced by <HOST:key> tokens. They are a point-in-time
record: `_station_home` is left out of the comparison so later station edits do not break them.
"""
import copy
from pathlib import Path

import pytest
import yaml

from franka_deploy.launch.launcher import HOST_PATH_KEYS, Launcher, load_config, recorded_host_fills
from franka_deploy.station import station_from_deploy_cfg

PKG = Path(__file__).resolve().parents[1]
DEPLOY = PKG / "configs" / "deploy.yaml"
GOLDEN = Path(__file__).resolve().parent / "golden" / "host_paths"
HOST = yaml.safe_load((PKG.parent / "configs" / "station" / "host" / "pc_laptop.yaml").read_text())
JETSON = yaml.safe_load((PKG.parent / "configs" / "station" / "host" / "jetson_orin_nano.yaml").read_text())


def resolved(profile):
    d = yaml.safe_load(yaml.safe_dump(Launcher(load_config(DEPLOY, profile)).cfg))
    d.pop("_station_home", None)
    return d


@pytest.mark.parametrize("profile", ["sim-iphone-fake", "sim-iphone", "sim-rig-fake", "sim-rig", "station-mock"])
def test_resolved_config_equals_before(profile):
    gold = yaml.safe_load((GOLDEN / f"resolved_{profile}.yaml").read_text())
    gold.pop("_station_home", None)
    got = resolved(profile)
    got["config_path"] = "<CONFIG_PATH>"
    if profile != "station-mock":
        for k in HOST_PATH_KEYS:
            assert got[k] == HOST[k], f"{profile}.{k} is not the pc_laptop host value"
            got[k] = f"<HOST:{k}>"
    assert got == gold


def test_sim_profiles_hold_no_scratch_paths():
    raw = yaml.safe_load(DEPLOY.read_text())
    for name, prof in raw["profiles"].items():
        if name.startswith("sim-"):
            assert not [k for k in HOST_PATH_KEYS if k in prof], f"{name} still sets host paths"
    assert "/tmp/claude" not in "\n".join(l for l in DEPLOY.read_text().splitlines() if not l.lstrip().startswith("#"))


def test_station_mock_keeps_base_values():
    """station-mock is pinned to the deploy.yaml base (Jetson) values via anchors: not filled from pc_laptop."""
    cfg = load_config(DEPLOY, "station-mock")
    assert cfg["work_root"] == "/home/jetson/franka_runs" and cfg["python"]["bp"].startswith("/home/jetson")


def test_explicit_profile_value_is_kept(tmp_path):
    raw = yaml.safe_load(DEPLOY.read_text())
    raw["profiles"]["sim-iphone-fake"]["work_root"] = str(tmp_path / "mine")
    p = tmp_path / "deploy.yaml"
    p.write_text(yaml.safe_dump(raw))
    cfg = load_config(p, "sim-iphone-fake")
    assert cfg["work_root"] == str(tmp_path / "mine")                       # explicit: kept
    assert cfg["checkpoint"] == HOST["checkpoint"]                          # the others still come from the host


def test_set_host_override_reaches_the_fill():
    cfg = load_config(DEPLOY, "sim-iphone-fake", station_set=["host.work_root=/scratch/b10/runs"])
    assert cfg["work_root"] == "/scratch/b10/runs"
    assert "host.work_root=/scratch/b10/runs" in cfg["station_set"]


def test_fill_is_recorded_in_overrides_not_in_resolved(tmp_path):
    cfg = load_config(DEPLOY, "sim-iphone-fake", station_set=[f"host.work_root={tmp_path}/runs"])
    st = station_from_deploy_cfg(cfg)
    rec = {r["path"]: r for r in recorded_host_fills(cfg, st)}
    assert set(rec) == set(HOST_PATH_KEYS)
    assert rec["work_root"]["base"] == "/home/jetson/franka_runs" and rec["work_root"]["host"] == f"{tmp_path}/runs"
    assert not any(k.startswith("_host") for k in cfg)                      # nothing new in the resolved config


def test_real_profiles_unchanged_by_the_fill():
    """real-* use the jetson_orin_nano host, whose values equal the deploy.yaml base: nothing is substituted."""
    for prof in ("real-iphone", "real-rig"):
        cfg = load_config(DEPLOY, prof)
        base = {k: v for k, v in yaml.safe_load(DEPLOY.read_text()).items() if k != "profiles"}
        for k in HOST_PATH_KEYS:
            assert cfg[k] == base[k] == JETSON[k]
        assert recorded_host_fills(cfg, station_from_deploy_cfg(cfg)) == []


def test_station_less_profiles_never_read_host(monkeypatch):
    import franka_deploy.launch.launcher as L

    def boom(cfg):
        raise AssertionError("station_from_deploy_cfg called for a station-less profile")

    monkeypatch.setattr(L, "station_from_deploy_cfg", boom)
    for prof in ("jetson", "mock"):
        cfg = L.load_config(DEPLOY, prof)
        assert cfg["work_root"] == "/home/jetson/franka_runs"
    assert copy.deepcopy(cfg)["profile"] == "mock"


# ------------------------------------------------------------------------------------------------ --set rules (B10 review)
def test_real_profile_accepts_only_host_overrides():
    from franka_deploy.launch.launcher import ConfigUsageError

    cfg = load_config(DEPLOY, "real-iphone", station_set=["host.work_root=/scratch/x"])
    assert cfg["work_root"] == "/scratch/x"
    for bad in ("criteria.limits.v_max_m_s=5", "robot.fci.libfranka_version=1", "tooling.hand.force_n=70"):
        with pytest.raises(ConfigUsageError, match=bad.split("=")[0]):
            load_config(DEPLOY, "real-iphone", station_set=[bad])
    with pytest.raises(ConfigUsageError):
        load_config(DEPLOY, "real-rig", station_set=["criteria.limits.w_max_rad_s=9"])


def test_sim_profile_still_accepts_criteria_overrides():
    cfg = load_config(DEPLOY, "sim-iphone-fake", station_set=["criteria.limits.v_max_m_s=0.5"])
    assert "criteria.limits.v_max_m_s=0.5" in cfg["station_set"]


@pytest.mark.parametrize("profile", ["jetson", "mock"])
def test_set_on_station_less_profile_is_a_usage_error(profile):
    from franka_deploy.launch.launcher import ConfigUsageError

    with pytest.raises(ConfigUsageError, match="no station"):
        load_config(DEPLOY, profile, station_set=["criteria.limits.v_max_m_s=0.1"])


def test_cli_refuses_with_exit_2(capsys):
    from franka_deploy.__main__ import main

    assert main(["check-config", "--config", str(DEPLOY), "--profile", "real-iphone", "--set", "criteria.limits.v_max_m_s=5"]) == 2
    assert "criteria.limits.v_max_m_s=5" in capsys.readouterr().err
    assert main(["check-config", "--config", str(DEPLOY), "--profile", "jetson", "--set", "host.work_root=/x"]) == 2


def test_bridge_clamp_never_exceeds_bridge_yaml_outside_sim():
    from franka_deploy.launch.launcher import build_bridge_cfg

    own = yaml.safe_load((PKG / "configs" / "bridge.yaml").read_text())["safety"]
    for prof in ("real-iphone", "real-rig"):
        cfg = load_config(DEPLOY, prof)
        # even a station value set behind the CLI's back (a profile station_set) cannot raise the clamp
        cfg["station_set"] = list(cfg.get("station_set", [])) + ["criteria.limits.v_max_m_s=5", "criteria.limits.w_max_rad_s=9"]
        bridge, _ = build_bridge_cfg(cfg, station_from_deploy_cfg(cfg))
        assert bridge["safety"]["v_max"] <= own["v_max"] and bridge["safety"]["w_max"] <= own["w_max"], bridge["safety"]
    # lowering still works
    cfg = load_config(DEPLOY, "real-iphone")
    cfg["station_set"] = ["criteria.limits.v_max_m_s=0.1"]
    assert build_bridge_cfg(cfg, station_from_deploy_cfg(cfg))[0]["safety"]["v_max"] == 0.1


def test_sim_profiles_may_still_raise_the_clamp_for_sweeps():
    from franka_deploy.launch.launcher import build_bridge_cfg

    cfg = load_config(DEPLOY, "sim-iphone-fake", station_set=["criteria.limits.v_max_m_s=0.5"])
    assert build_bridge_cfg(cfg, station_from_deploy_cfg(cfg))[0]["safety"]["v_max"] == 0.5
