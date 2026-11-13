"""launch.py (deploy/STRUCTURE.md §9): program(), checks(), the mock choice."""

from __future__ import annotations

import socket
import sys

import pytest

deploy = pytest.importorskip("deploy")

from conftest import SMALL_RIG, write_station  # noqa: E402
from process_frames import FOLDER  # noqa: E402


@pytest.fixture(scope="module")
def process_class():
    return deploy.load_process_class(FOLDER)


def make(process_class, tmp_path, mock, process_frames=None, endpoints=None):
    ep = {"pi_frames": "tcp://127.0.0.1:1", "pi_time_sync": "tcp://127.0.0.1:2",
          "mock_pi_frames": f"tcp://127.0.0.1:{free_port()}", "mock_pi_time_sync": f"tcp://127.0.0.1:{free_port()}",
          "frames": f"tcp://127.0.0.1:{free_port()}"}
    ep.update(endpoints or {})
    station = write_station(tmp_path / "station.yaml", endpoints=ep, process_frames=process_frames,
                            extra={"python": {"light": sys.executable, "policy": sys.executable}})
    return process_class(deploy.Station.load(station, mock=mock), mock, None)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def by_name(results):
    return {r.name: r for r in results}


def test_name_and_program(process_class, tmp_path):
    assert process_class.NAME == "process_frames"
    p = make(process_class, tmp_path, mock=False)
    assert p.program() == [sys.executable, "-m", "process_frames"]
    assert "--mock" in make(process_class, tmp_path, mock=True).command()


def test_checks_with_the_default_640_rig(process_class, tmp_path):
    r = by_name(make(process_class, tmp_path, mock=False).checks())
    assert r["rig file"].status == "pass"
    assert r["rig file"].detail == "rig_pi_v2_640: main imx519 640x480@30, ultrawide imx219 640x480@30"
    assert "rig stand-in" not in r
    assert r["imports"].status == "pass"
    assert r["decoder"].status == "info" and r["decoder"].detail.startswith("jpeg (stream pi_v1")
    assert r["Pi reachable"].status == "fail" and "is the Pi on" in r["Pi reachable"].detail
    assert r["camera source"].detail == "Pi at 127.0.0.1"


def test_another_rig_file_is_summarised(process_class, tmp_path):
    r = by_name(make(process_class, tmp_path, mock=False, process_frames={"rig_file": str(SMALL_RIG)}).checks())
    assert r["rig file"].status == "pass"
    assert r["rig file"].detail == "rig_small: main imx519 160x90@30, ultrawide imx219 128x96@30"


def test_pi_reachable_passes_against_a_listening_port(process_class, tmp_path):
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        port = srv.getsockname()[1]
        p = make(process_class, tmp_path, mock=False, endpoints={"pi_frames": f"tcp://127.0.0.1:{port}"},
                 process_frames={"rig_file": str(SMALL_RIG), "stream": "c0", "decoder": "jpeg"})
        r = by_name(p.checks())
    assert r["Pi reachable"].status == "pass"
    assert r["decoder"].detail == "jpeg (stream c0, backup, PLAN §9)"


def test_broken_station_section_and_missing_rig_file(process_class, tmp_path):
    p = make(process_class, tmp_path, mock=False, process_frames={"bogus": 1})
    assert p.base_checks()[0].status == "fail"
    assert [r.status for r in p.checks()] == ["fail"]
    p = make(process_class, tmp_path, mock=False, process_frames={"rig_file": "nope.yaml"})
    r = by_name(p.checks())
    assert r["rig file"].status == "fail" and "nope.yaml" in r["rig file"].detail


def test_mock_skips_the_pi_check_and_checks_mock_episode(process_class, tmp_path, pi_episode):
    r = by_name(make(process_class, tmp_path, mock=True).checks())
    assert "Pi reachable" not in r
    assert r["mock episode"].status == "fail" and r["mock episode"].detail == "process_frames.mock_episode is not set"
    r = make(process_class, tmp_path, mock=True, process_frames={"mock_episode": str(tmp_path)}).checks()
    assert by_name(r)["mock episode"].status == "fail"
    r = make(process_class, tmp_path, mock=True, process_frames={"mock_episode": str(pi_episode)}).checks()
    d = by_name(r)
    assert d["mock episode"].status == "pass"
    assert "FakePiSender" in d["camera source"].detail
    assert [x.status for x in r if x.name.startswith("port ")] == ["pass", "pass", "pass"]
    assert not [x for x in r if x.status == "fail"]
