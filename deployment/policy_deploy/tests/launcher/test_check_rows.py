"""Every row of STRUCTURE.md 9.2 with a passing and a failing input (12.2). The process rows' details are in
test_processes.py; here: the station rows, the whole table in order, and --mock skipping the Pi and PC rows."""
from __future__ import annotations

import collections
import dataclasses
import json
import subprocess
from pathlib import Path

import pytest
from deploy_contracts import ContractLimits, LinkSettings, TcpFrame
from deploy_contracts import paths as contract_paths
from launcher_fakes import check_line, write_fake_python

import policy_deploy.launcher.launcher as launcher_module
from policy_deploy.contracts import Station
from policy_deploy.launcher.launcher import Launcher

ANSWERS = {
    "policy_deploy.frames.rig_file": (json.dumps({"ok": True, "rig_id": "r", "cameras": {
        "main": {"sensor": "imx519", "width": 640, "height": 480, "fps": 30.0},
        "ultrawide": {"sensor": "imx219", "width": 640, "height": 480, "fps": 30.0}}}) + "\n", 0),
    "policy_deploy.policy.self_check": (check_line("settings", "pass", "ok") + "\n", 0),
    "policy_deploy.manager.self_check": (check_line("worst gap", "pass", "0.57 s < 1.0 s") + "\n", 0),
    "policy_deploy.manager.pc_probe": (check_line("C2", "pass", "100 Hz") + "\n", 0),
}


@pytest.fixture
def py(tmp_path):
    return str(write_fake_python(tmp_path / "fake_python", ANSWERS, tmp_path / "calls.jsonl"))


@pytest.fixture
def episode(tmp_path):
    folder = tmp_path / "episode"
    folder.mkdir()
    (folder / "meta.json").write_text("{}")
    return folder


@pytest.fixture
def local_link(monkeypatch):
    """link.yaml with free C2/C3 ports and the laptop address 127.0.0.1 (which is on every machine)."""
    from station_helpers import free_port
    real = LinkSettings.load()
    fake = dataclasses.replace(real, state_port=free_port(), commands_port=free_port(),
                               laptop_addresses=("127.0.0.1",))
    monkeypatch.setattr(LinkSettings, "load", classmethod(lambda cls, path=None: fake))
    return fake


@pytest.fixture
def nmcli(monkeypatch):
    """Answer `nmcli` with the given lines; every other subprocess runs for real."""
    real_run = subprocess.run
    state = {"stdout": "yes:ASUS_5g\nno:Other\n", "missing": False}

    def run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "nmcli":
            if state["missing"]:
                raise FileNotFoundError("nmcli")
            return subprocess.CompletedProcess(cmd, 0, stdout=state["stdout"], stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(launcher_module.subprocess, "run", run)
    return state


def rows(launcher):
    return {(r.process, r.name): r for r in launcher._results()}


def station_rows(launcher):
    return {r.name: r for r in launcher.station_checks()}


def make(make_station, mock, script, **sections):
    sections.setdefault("manager", {"operator_script": str(script)})
    return Launcher(Station.load(make_station(**sections), mock), mock)


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "script.yaml"
    path.write_text("- {key: q}\n")
    return path


# ---------------------------------------------------------------------------------------------------- station


def test_station_rows_pass(make_station, script, local_link, nmcli):
    r = station_rows(make(make_station, False, script))
    assert list(r) == ["station.yaml", "deploy_contracts", "contracts/link.yaml", "contracts/frames.yaml",
                       "contracts/limits.yaml", "laptop address", "Wi-Fi", "commit", "disk"]
    for name in ("station.yaml", "deploy_contracts", "contracts/link.yaml", "contracts/frames.yaml",
                 "contracts/limits.yaml", "laptop address", "Wi-Fi"):
        assert r[name].status == "pass", (name, r[name].detail)
    assert r["commit"].status == "info" and r["commit"].detail.startswith("laptop commit ")
    assert r["disk"].status in ("pass", "warn") and "GB free under" in r["disk"].detail
    assert "settled: false" in r["contracts/frames.yaml"].detail


def test_deploy_contracts_from_another_checkout(make_station, script, monkeypatch, tmp_path):
    monkeypatch.setattr(contract_paths, "contracts_dir", lambda: tmp_path)
    r = station_rows(make(make_station, True, script))["deploy_contracts"]
    assert r.status == "fail" and "not this checkout's" in r.detail and "uv pip install -e deployment/contracts" in \
        r.detail

    def missing():
        raise FileNotFoundError("deploy_contracts must be installed editable from deployment/contracts")

    monkeypatch.setattr(contract_paths, "contracts_dir", missing)
    r = station_rows(make(make_station, True, script))["deploy_contracts"]
    assert r.status == "fail" and "installed editable" in r.detail


@pytest.mark.parametrize("cls,label", [(LinkSettings, "contracts/link.yaml"), (TcpFrame, "contracts/frames.yaml"),
                                       (ContractLimits, "contracts/limits.yaml")])
def test_contract_file_invalid(make_station, script, monkeypatch, cls, label):
    launcher = make(make_station, True, script)

    def broken(path=None):
        raise ValueError(f"{label}: timing.c3_timeout_s: must be a finite number")

    monkeypatch.setattr(cls, "load", staticmethod(broken))
    r = station_rows(launcher)[label]
    assert r.status == "fail" and "ValueError" in r.detail


def test_laptop_address_not_on_this_laptop(make_station, script, local_link, nmcli, monkeypatch):
    fake = dataclasses.replace(local_link, laptop_addresses=("192.0.2.20", "192.0.2.21"))
    monkeypatch.setattr(LinkSettings, "load", classmethod(lambda cls, path=None: fake))
    r = station_rows(make(make_station, False, script))["laptop address"]
    assert r.status == "fail" and "192.0.2.20, 192.0.2.21" in r.detail and "firewall" in r.detail


def test_wifi_rows(make_station, script, local_link, nmcli):
    launcher = make(make_station, False, script)
    nmcli["stdout"] = "yes:LabGuest\n"
    r = station_rows(launcher)["Wi-Fi"]
    assert r.status == "warn" and "not on ASUS_5g" in r.detail and "LabGuest" in r.detail
    nmcli["missing"] = True
    assert station_rows(launcher)["Wi-Fi"].status == "warn"


def test_disk_low_warns(make_station, script, monkeypatch):
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(launcher_module.shutil, "disk_usage", lambda p: usage(10**12, 10**12, 10**9))
    r = station_rows(make(make_station, True, script))["disk"]
    assert r.status == "warn" and "below 2 GB" in r.detail


def test_commit_unknown_without_git(make_station, script, monkeypatch):
    def no_git(cmd, *a, **k):
        raise FileNotFoundError("git")

    launcher = make(make_station, True, script)
    monkeypatch.setattr(launcher_module.subprocess, "run", no_git)
    r = station_rows(launcher)["commit"]
    assert (r.status, r.detail) == ("info", "laptop commit unknown")


# ------------------------------------------------------------------------------------------------ whole table


def test_mock_table_skips_pi_and_pc_rows(make_station, script, py, episode, local_link, tmp_path):
    launcher = make(make_station, True, script, python={"light": py}, frames={"mock_episode": str(episode)})
    table = rows(launcher)
    assert [k for k, r in table.items() if r.status == "fail"] == []
    processes = [p for p, _ in table]
    assert processes.index("station") < processes.index("fake_pc") < processes.index("frames") < \
        processes.index("policy") < processes.index("manager")
    names = {n for _, n in table}
    assert "laptop address" not in names and "Wi-Fi" not in names and "Pi reachable" not in names
    assert table[("manager", "PC")].detail == "fake PC on 127.0.0.1, started by run"
    assert ("fake_pc", "C2 port") in table and ("frames", "mock episode") in table
    calls = (tmp_path / "calls.jsonl").read_text()
    assert "policy_deploy.manager.pc_probe" not in calls


def test_mock_cross_test_has_no_fake_pc_rows(make_station, script, py, episode):
    launcher = make(make_station, True, script, python={"light": py}, frames={"mock_episode": str(episode)},
                    launcher={"fake_pc": False})
    table = rows(launcher)
    assert not any(p == "fake_pc" for p, _ in table)
    assert table[("manager", "PC")].detail == "PC on 127.0.0.1 started outside the launcher (cross test)"


def test_real_table_has_pi_settled_and_pc_rows(make_station, script, py, local_link, nmcli):
    from station_helpers import free_port
    launcher = make(make_station, False, script, python={"light": py}, pi={"address": "127.0.0.1",
                                                                          "port": free_port()})
    table = rows(launcher)
    assert ("frames", "Pi reachable") in table and ("policy", "settled") in table and ("manager", "C2") in table
    assert not any(p == "fake_pc" for p, _ in table)
    assert table[("policy", "settled")].status == "pass"                  # hold_still is allowed unsettled


def test_check_prints_the_table_and_exits_2_on_a_fail(make_station, script, py, episode, local_link, capsys,
                                                      monkeypatch):
    launcher = make(make_station, True, script, python={"light": py}, frames={"mock_episode": str(episode)})
    assert launcher.check() == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["process", "check", "status", "detail"]
    assert out.splitlines()[-1].startswith("policy_deploy check: ready (")
    monkeypatch.setattr(contract_paths, "contracts_dir", lambda: Path("/elsewhere"))
    assert launcher.check() == 2
    assert capsys.readouterr().out.splitlines()[-1] == "policy_deploy check: 1 check(s) failed"
