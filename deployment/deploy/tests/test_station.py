from __future__ import annotations

import sys
from pathlib import Path

import pytest

import deploy
from deploy import Station, StationError, parse_endpoint
from helpers import REAL_ROOT, STATION, station_data, write_yaml

DATA = Path(__file__).parent / "data"


def problems_of(path: Path, mock: bool = False) -> list[str]:
    with pytest.raises(StationError) as info:
        Station.load(path, mock)
    return info.value.problems


@pytest.mark.parametrize("mock", [False, True])
def test_example_station_loads(mock):
    station = Station.load(DATA / "station_example.yaml", mock)
    assert station.endpoint("commands") == "tcp://127.0.0.1:5630"
    assert station.section("policy") == {"kind": "hold_still"}
    assert station.section("message_comms") == {}
    assert station.session_root == Path("~/deploy_sessions").expanduser()
    assert station.python("policy") == str(Path("~/miniconda3/envs/behavior_prompting/bin/python").expanduser())


@pytest.mark.parametrize("mock", [False, True])
def test_real_station_loads(mock):
    station = Station.load(None, mock)
    assert station.path == REAL_ROOT / "station.yaml"


def test_launcher_defaults(make_station):
    station = Station.load(make_station(), False)
    assert dict(station.ready_timeout_s) == {"franka_control": 30, "process_frames": 30, "policy": 300,
                                             "message_comms": 30}
    assert (station.stop_timeout_s, station.poll_s) == (5.0, 0.1)
    assert (station.tcp_check_max_mm, station.tcp_check_max_deg) == (2.0, 0.5)


def test_launcher_values_override_defaults(make_station):
    station = Station.load(make_station(launcher={"ready_timeout_s": {"policy": 12}, "poll_s": 0.5}), False)
    assert station.ready_timeout_s["policy"] == 12.0
    assert station.ready_timeout_s["franka_control"] == 30.0
    assert station.poll_s == 0.5


def test_every_problem_listed_at_once(make_station):
    data = station_data(station_version=2, bogus=1, launcher={"poll_s": -1, "ready_timeout_s": {"polcy": 3}})
    data["endpoints"]["policy"] = "localhost:5640"
    del data["endpoints"]["state"]
    data["python"]["light"] = 5
    path = write_yaml(make_station().parent / "broken.yaml", data)
    problems = problems_of(path)
    assert 'broken.yaml: endpoints.policy: expected "tcp://host:port", got \'localhost:5640\'' in problems
    expected = ["broken.yaml: station_version: expected 1, got 2", "broken.yaml: bogus: unknown key",
                "broken.yaml: endpoints.state: required", "broken.yaml: python.light: expected a path",
                "broken.yaml: launcher.poll_s: expected a number > 0", "broken.yaml: launcher.ready_timeout_s.polcy"]
    for start in expected:
        assert any(p.startswith(start) for p in problems), (start, problems)
    assert len(problems) == 7


@pytest.mark.parametrize("mock", [False, True])
def test_session_root_and_every_endpoint_required(make_station, mock):
    data = station_data(session_root=None, endpoints={})
    problems = problems_of(write_yaml(make_station().parent / "s.yaml", data), mock)
    assert "s.yaml: session_root: required" in problems
    required = ["mock_pi_frames", "mock_pi_time_sync", "frames", "state", "commands", "policy"]
    if not mock:
        required += ["pi_frames", "pi_time_sync"]
    for name in required:
        assert any(p.startswith(f"s.yaml: endpoints.{name}: required") for p in problems), name
    assert len(problems) == 1 + len(required)


def test_station_version_required(make_station):
    assert problems_of(make_station(station_version=None)) == [f"station_{0}.yaml: station_version: required"]


def test_mock_needs_no_policy_python_or_pi_endpoints(make_station):
    endpoints = {k: v for k, v in STATION["endpoints"].items() if not k.startswith("pi_")}
    path = make_station(python={"light": "/usr/bin/python3"}, endpoints=endpoints)
    station = Station.load(path, mock=True)
    with pytest.raises(StationError):
        station.python("policy")
    with pytest.raises(StationError):
        station.endpoint("pi_frames")
    problems = problems_of(path, mock=False)
    assert sorted(p.split(": ")[1] for p in problems) == ["endpoints.pi_frames", "endpoints.pi_time_sync",
                                                          "python.policy"]


def test_python_light_defaults_to_running_python(make_station):
    station = Station.load(make_station(python={"policy": "/opt/p/bin/python"}), False)
    assert station.python("light") == sys.executable


def test_paths_resolved_from_station_folder(make_station):
    path = make_station(session_root="runs/here", python={"light": "~/py", "policy": "envs/p/bin/python"})
    station = Station.load(path, False)
    assert station.session_root == path.parent / "runs" / "here"
    assert station.python("light") == str(Path.home() / "py")
    assert station.python("policy") == str(path.parent / "envs/p/bin/python")
    assert station.resolve("/abs/x") == Path("/abs/x")


def test_relative_station_path_is_made_absolute(make_station, monkeypatch):
    path = make_station()
    monkeypatch.chdir(path.parent)
    assert Station.load(Path(path.name), False).path == path


def test_missing_file_and_non_map(tmp_path):
    assert problems_of(tmp_path / "nope.yaml")[0].startswith("nope.yaml: not found")
    (tmp_path / "list.yaml").write_text("- 1\n")
    assert "expected a map" in problems_of(tmp_path / "list.yaml")[0]


def test_section_must_be_a_map(make_station):
    assert problems_of(make_station(policy=[1])) == ["station_0.yaml: policy: expected a map, got [1]"]


def test_default_path_follows_deployment_root(fake_root):
    assert Station.load(None, False).path == fake_root / "station.yaml"
    assert deploy.DEPLOYMENT_ROOT == fake_root


@pytest.mark.parametrize("text,ok", [("tcp://127.0.0.1:5630", ("127.0.0.1", 5630)), ("tcp://*:1", ("*", 1)),
                                     ("localhost:5640", None), ("tcp://h:0", None), ("tcp://h:70000", None),
                                     ("tcp://h", None), (5, None)])
def test_parse_endpoint(text, ok):
    if ok is None:
        with pytest.raises(ValueError):
            parse_endpoint(text)
    else:
        assert parse_endpoint(text) == ok
