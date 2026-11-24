"""Every row of `control_deploy check` with injected inputs (control_deploy/STRUCTURE.md 6.3, 9.2)."""
from __future__ import annotations

import resource
from pathlib import Path

import deploy_contracts
import pytest
from helpers import write_fake_binary, write_station

from control_deploy import paths
from control_deploy.binary import BinaryError, CheckConfigResult, VersionInfo
from control_deploy.check_result import CheckResult, exit_code
from control_deploy.checks import PcChecks, Probes, todo_keys
from control_deploy.firewall import LiveRules, LiveTable

HEAD = "1" * 40
HERE = paths.contracts_dir().resolve()
GOOD_CONTRACTS = {"contracts_dir": str(HERE), "limits_file": str(HERE / "limits.yaml"),
                  "frames_file": str(HERE / "frames.yaml"), "link_file": str(HERE / "link.yaml")}
GOOD_FIREWALL = LiveRules(loopback_ports={5620, 5630}, accepted={"192.168.50.20": {5620, 5630}},
                          dropped_ports={5620, 5630})


class FakeBinary:
    def __init__(self, version: VersionInfo | Exception, config: CheckConfigResult) -> None:
        self._version, self._config = version, config

    def version(self) -> VersionInfo:
        if isinstance(self._version, Exception):
            raise self._version
        return self._version

    def check_config(self, station, mock, local) -> CheckConfigResult:
        return self._config


def version(commit=HEAD, dirty=False, libfranka="0.9.2") -> VersionInfo:
    return VersionInfo(commit, dirty, libfranka, f"franka_control commit={commit} dirty={'yes' if dirty else 'no'} "
                                                 f"libfranka={libfranka}")


def config(contracts=None, **fc) -> CheckConfigResult:
    franka = {"robot_ip": "172.16.0.2", "realtime": "ignore", "load": {"mass_kg": 0.0}}
    franka.update(fc)
    return CheckConfigResult(ok=True, config={"contracts": contracts or GOOD_CONTRACTS, "franka_control": franka})


def rows(tmp_path: Path, mock=False, local=False, station_fc=None, ver=None, cfg=None, **probes) -> dict:
    fake = write_fake_binary(tmp_path)
    fc = {"binary": str(fake)}
    fc.update(station_fc or {})
    station = write_station(tmp_path, franka_control=fc)
    p = Probes(
        git_head=lambda: HEAD, git_dirty=lambda: False,
        binary=lambda path: FakeBinary(ver or version(), cfg or config()),
        local_addresses=lambda: {"lo": ["127.0.0.1"], "wlp2s0": ["192.168.50.10"], "enp3s0": ["172.16.0.1"]},
        port_free=lambda ep: CheckResult("control_deploy", "port", "pass", f"{ep} is free"),
        read_firewall=lambda: LiveTable("ok", GOOD_FIREWALL, ""),
        ufw_enabled=lambda: False,
        host_reachable=lambda h, port, t: CheckResult("control_deploy", "reach", "pass", "opened"),
        rt_flag=lambda: "1", rtprio_limit=lambda: 99)
    for k, v in probes.items():
        setattr(p, k, v)
    results = PcChecks(station, mock, local, p).run()
    out: dict = {}
    for r in results:
        out.setdefault(r.name, []).append(r)
    out["_all"] = results
    return out


def one(rs: dict, name: str) -> CheckResult:
    assert name in rs, f"no row {name}: {[r.name for r in rs['_all']]}"
    return rs[name][0]


def test_everything_good_passes(tmp_path):
    rs = rows(tmp_path, station_fc={"realtime": "enforce", "load": {"mass_kg": 0.4}},
              cfg=config(realtime="enforce", load={"mass_kg": 0.4}))
    bad = [r for r in rs["_all"] if r.status == "fail"]
    assert not bad, bad
    assert one(rs, "rt kernel").status == "pass" and one(rs, "rtprio").status == "pass"
    assert one(rs, "payload").status == "pass"
    assert exit_code(rs["_all"]) == 0


def test_env_from_another_checkout_fails(tmp_path):
    rs = rows(tmp_path, contracts_dir=lambda: Path("/elsewhere/deployment/contracts"))
    assert one(rs, "env").status == "fail"
    assert "uv pip install -e deployment/contracts" in one(rs, "env").detail


def test_a_commit_mismatch_fails(tmp_path):
    r = one(rows(tmp_path, ver=version(commit="2" * 40)), "version")
    assert r.status == "fail" and f"built from {'2' * 40}, checkout is {HEAD}: rebuild after every pull" in r.detail


def test_dirty_and_unknown_warn(tmp_path):
    assert one(rows(tmp_path, ver=version(dirty=True)), "version").status == "warn"
    r = one(rows(tmp_path, git_dirty=lambda: True), "version")
    assert r.status == "warn" and "the laptop's hello check sees only the commit" in r.detail
    assert one(rows(tmp_path, ver=version(commit="unknown")), "version").status == "warn"


def test_libfranka_none_fails_but_is_info_with_mock(tmp_path):
    assert one(rows(tmp_path, ver=version(libfranka="none")), "version").status == "fail"
    assert one(rows(tmp_path, mock=True, ver=version(libfranka="none")), "version").status == "info"


def test_an_unrunnable_binary_fails_version(tmp_path):
    r = one(rows(tmp_path, ver=BinaryError("unexpected --version output 'x'")), "version")
    assert r.status == "fail" and "build.sh" in r.detail


def test_a_missing_binary_fails_the_binary_row(tmp_path):
    station = write_station(tmp_path, franka_control={"binary": str(tmp_path / "nope")})
    results = PcChecks(station, True, True, Probes(binary=lambda p: pytest.fail("not run"))).run()
    r = [x for x in results if x.name == "binary"][0]
    assert r.status == "fail" and "build it: control_deploy/franka_control/build.sh" in r.detail
    assert not [x for x in results if x.name in ("version", "config")]


def test_baked_paths_moved_fails(tmp_path):
    moved = {k: v.replace(str(HERE), "/old/clone/deployment/contracts") for k, v in GOOD_CONTRACTS.items()}
    r = one(rows(tmp_path, cfg=config(contracts=moved)), "baked paths")
    assert r.status == "fail" and "the checkout moved since the build" in r.detail


def test_config_problems_fail_with_the_count(tmp_path):
    cfg = CheckConfigResult(ok=False, problems=["link.yaml: pc.address: bad", "station.yaml: x: y"])
    rs = rows(tmp_path, cfg=cfg)
    assert one(rs, "config").status == "fail" and one(rs, "config").detail == "link.yaml: pc.address: bad (2 problems)"
    assert one(rs, "baked paths").status == "info"


def test_tcp_frame_info_while_not_settled(tmp_path):
    r = one(rows(tmp_path), "tcp frame")
    assert r.status == "info" and "settled: false" in r.detail


def test_todo_markers_warn_unless_the_station_overrides_them(tmp_path):
    r = one(rows(tmp_path), "todo")
    assert r.status == "warn" and "safety.workspace_min_m" in r.detail and "load.mass_kg" in r.detail
    keys = todo_keys(paths.config_file().read_text())
    over: dict = {}
    for k in keys:
        node = over
        parts = k.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = 1
    assert one(rows(tmp_path, station_fc=over), "todo").status == "pass"


def test_todo_keys_are_dotted_paths():
    text = "a: 1  # TODO: x\nsafety:\n  box: [1]   # TODO\n  ok: 2\n# a TODO comment line\nload:\n  m: 0 # TODO\n"
    assert todo_keys(text) == ["a", "safety.box", "load.m"]


def test_payload_warns_at_zero(tmp_path):
    r = one(rows(tmp_path), "payload")
    assert r.status == "warn" and "no payload set" in r.detail


def test_link_address_absent_fails(tmp_path):
    rs = rows(tmp_path, local_addresses=lambda: {"lo": ["127.0.0.1"], "wlp2s0": ["192.168.50.37"]})
    r = one(rs, "link address")
    assert r.status == "fail"
    assert r.detail == ("link.yaml pc.address 192.168.50.10 is not on this PC (wlp2s0 has 192.168.50.37): reserve the "
                        "address on ASUS_5g or update link.yaml")


def test_port_busy_fails(tmp_path):
    def busy(ep):
        return CheckResult("control_deploy", "port", "fail" if ep.endswith(":5630") else "pass", f"{ep} in use")
    rs = rows(tmp_path, port_free=busy)
    assert [r.status for r in rs["ports"]] == ["pass", "fail"]
    assert rs["ports"][1].detail.startswith("tcp://192.168.50.10:5630")


def test_ports_bind_127_with_local(tmp_path):
    seen = []
    rows(tmp_path, local=True, port_free=lambda ep: seen.append(ep) or CheckResult("c", "p", "pass", ep))
    assert seen == ["tcp://127.0.0.1:5620", "tcp://127.0.0.1:5630"]


@pytest.mark.parametrize("table, status, text", [
    (LiveTable("ok", GOOD_FIREWALL, ""), "pass", "matches"),
    (LiveTable("ok", LiveRules({5620, 5630}, {"192.168.50.99": {5620, 5630}}, {5620, 5630}), ""), "fail",
     "not accepted: 192.168.50.20"),
    (LiveTable("absent", None, "No such file or directory"), "fail", "print it with control_deploy firewall"),
    (LiveTable("unreadable", None, "Operation not permitted"), "warn", "cannot read the firewall without root"),
])
def test_firewall(tmp_path, table, status, text):
    r = one(rows(tmp_path, read_firewall=lambda: table), "firewall")
    assert r.status == status and text in r.detail


def test_ufw_enabled_warns(tmp_path):
    r = one(rows(tmp_path, ufw_enabled=lambda: True), "firewall")
    assert r.status == "warn" and "ufw is enabled: also allow the laptop there" in r.detail


def test_rt_kernel_and_rtprio(tmp_path):
    rs = rows(tmp_path, station_fc={"realtime": "enforce"}, cfg=config(realtime="enforce"), rt_flag=lambda: "",
              rtprio_limit=lambda: 0)
    assert one(rs, "rt kernel").status == "fail" and "PREEMPT_RT" in one(rs, "rt kernel").detail
    assert one(rs, "rtprio").status == "fail" and "rtprio limit 0 < 99" in one(rs, "rtprio").detail
    rs = rows(tmp_path, station_fc={"realtime": "enforce"}, cfg=config(realtime="enforce"),
              rtprio_limit=lambda: resource.RLIM_INFINITY)
    assert one(rs, "rtprio").detail == "rtprio limit unlimited"
    assert one(rows(tmp_path), "realtime").status == "warn"


def test_fci_network_and_arm(tmp_path):
    rs = rows(tmp_path, local_addresses=lambda: {"wlp2s0": ["192.168.50.10"]},
              host_reachable=lambda h, p, t: CheckResult("c", "r", "fail", "no"))
    assert one(rs, "fci network").detail == "no 172.16.0.x address: sudo nmcli con up franka"
    assert one(rs, "arm reachable").detail.startswith("arm not reachable at 172.16.0.2:1337")


def test_mock_skips_arm_realtime_payload_fci_todo(tmp_path):
    rs = rows(tmp_path, mock=True)
    for name in ("arm reachable", "rt kernel", "rtprio", "realtime", "payload", "fci network", "todo"):
        assert name not in rs, name
    assert one(rs, "backups").detail.startswith("arm: FakeArm, gripper: FakeGripper (--mock)")


def test_local_skips_address_and_firewall(tmp_path):
    rs = rows(tmp_path, local=True, read_firewall=lambda: pytest.fail("not read"))
    assert "link address" not in rs and "firewall" not in rs
    assert one(rs, "backups").detail.endswith("binds 127.0.0.1")


def test_backups_real(tmp_path):
    assert one(rows(tmp_path), "backups").detail == "arm: LibfrankaArm at 172.16.0.2, realtime ignore; binds " \
                                                   "192.168.50.10"


def test_a_bad_station_fails(tmp_path):
    station = tmp_path / "station.yaml"
    station.write_text("station_version: 1\nsession_root: x\nendpoints: {}\n")
    results = PcChecks(station, True, True, Probes(binary=lambda p: pytest.fail("x"))).run()
    r = [x for x in results if x.name == "station"][0]
    assert r.status == "fail" and "endpoints" in r.detail


def test_session_root_not_writable(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        fake = write_fake_binary(tmp_path)
        station = tmp_path / "station.yaml"
        station.write_text(f"station_version: 1\nsession_root: {ro / 'sessions'}\nfranka_control:\n  binary: {fake}\n")
        p = Probes(binary=lambda path: FakeBinary(version(), config()), git_head=lambda: HEAD, git_dirty=lambda: False,
                   port_free=lambda ep: CheckResult("c", "p", "pass", ep))
        results = PcChecks(station, True, True, p).run()
        r = [x for x in results if x.name == "session root"][0]
        assert r.status == "fail"
    finally:
        ro.chmod(0o700)


def test_real_contracts_dir_is_this_checkout():
    assert deploy_contracts.paths.contracts_dir().resolve() == HERE
