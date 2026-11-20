# adapted from rnoro/deployment:deployment/message_comms/tests/test_flow.py @ 91b178b
# changes: the real manager main() against an in-process FakePc (instead of FakeFrankaControl) and fake frames and
# policy on real sockets, with a script; --mock (hello says mock); the endpoints are patched onto the settings; a
# shorter clock warm-up (clock.min_samples_s 0.2) keeps the runs short; the new FATAL texts.
"""PLAN 4 "one policy step": the real manager main() against fakes on real sockets, with a script."""
import dataclasses
import json
import threading

import pytest
from manager_servers import FakePolicy, FakeProcessFrames, make_fake_pc
from station_helpers import write_station

from policy_deploy.manager.__main__ import main
from policy_deploy.manager.settings import Settings

EPISODE = "- {wait_phase: idle}\n- {key: c}\n- {wait_steps: %d}\n- {key: %s}\n"
QUIT = "- {wait_phase: idle}\n- {key: q}\n"
FAST_CLOCK = {"min_samples_s": 0.2, "min_samples": 5}


class Stack:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.frames = FakeProcessFrames()
        self.policy = FakePolicy()
        self.pc = make_fake_pc(tmp_path / "pc", home_duration_s=0.3)
        self.session = tmp_path / "session"
        original = Settings.load.__func__

        def load(cls, station_path, mock, link=None, frame=None):
            s = original(cls, station_path, mock, link, frame)
            return dataclasses.replace(s, frames_endpoint=self.frames.endpoint, policy_endpoint=self.policy.endpoint,
                                       state_endpoint=self.pc.state_endpoint,
                                       commands_endpoint=self.pc.commands_endpoint)
        monkeypatch.setattr(Settings, "load", classmethod(load))

    def run(self, script: str, capsys, **overrides):
        (self.tmp / "script.yaml").write_text(script)
        settings = {"operator_script": "script.yaml", "timeouts": {"startup_s": 5.0, "policy_hang_s": 0.5},
                    "clock": FAST_CLOCK}
        settings.update(overrides)
        station = write_station(self.tmp, manager=settings)
        code = main(["--station", str(station), "--session", str(self.session), "--mock"])
        err = capsys.readouterr().err
        return code, [line for line in err.splitlines() if line.strip()]

    def jsonl(self, name):
        path = self.session / "manager" / name
        return [json.loads(x) for x in path.read_text().splitlines()]

    def names(self):
        return [n.split(".")[1].split("/")[0] for n in self.pc.request_names()]

    def sets(self):
        return self.names().count("set_waypoints")

    def close(self):
        for fake in (self.frames, self.policy, self.pc):
            fake.close()


@pytest.fixture
def stack(tmp_path, monkeypatch):
    s = Stack(tmp_path, monkeypatch)
    yield s
    s.close()


def stop_after_last(stack, name):
    names = stack.names()
    positions = [i for i, n in enumerate(names) if n == name]
    return bool(positions) and "stop" in names[positions[-1]:]


def when(condition, action):
    """Run action once condition() turns true (a watcher thread; condition timeouts, no fixed sleeps)."""
    def watch():
        ev = threading.Event()
        for _ in range(4000):
            if condition():
                action()
                return
            ev.wait(0.005)
    threading.Thread(target=watch, daemon=True).start()


# ---- the normal step ----------------------------------------------------------------------------------------------

def test_one_episode(stack, capsys):
    code, err = stack.run(EPISODE % (10, "y") + QUIT, capsys)
    assert code == 0, err[-3:]
    assert not any(line.startswith("FATAL") for line in err)
    names = stack.names()
    assert names.index("hello") < names.index("get_state") < names.index("start")    # the start-up order
    assert names[-1] == "stop"                          # stop is sent first at exit
    beats = [t for t, n in stack.pc.requests if n in ("c3.heartbeat/v1", "c3.hello/v1")]
    assert len(beats) > 10 and max(b - a for a, b in zip(beats, beats[1:])) < 570_000_000   # the worst gap
    episodes = stack.jsonl("episodes.jsonl")
    assert [e["outcome"] for e in episodes] == ["success"] and episodes[0]["steps_sent"] >= 10
    steps = [s for s in stack.jsonl("steps.jsonl") if s["sent"]]
    assert all(s["rejected"] == 0 for s in steps) and sum(s["accepted"] for s in steps) > 0
    assert stack.policy.calls[:2] == ["c4.config/v1", "c4.reset/v1"]
    assert "c4.prompt/v1" not in stack.policy.calls
    clock = stack.jsonl("clock.jsonl")[0]
    true_offset = stack.pc.clock.at(clock["t_ns"]) - clock["t_ns"]           # laptop x (1 + 30 ppm) + 1000 s
    assert true_offset > 1000e9 and clock["offset_ns"] == pytest.approx(true_offset, abs=5e6)


# ---- slow cases: keep running, exit 0 -----------------------------------------------------------------------------

def test_camera_stale_then_resume(stack, capsys):
    def stale():
        stack.frames.mode = "stale"
        threading.Timer(0.3, lambda: setattr(stack.frames, "mode", "ok")).start()
    when(lambda: stack.sets() >= 3, stale)
    code, err = stack.run(EPISODE % (12, "y") + QUIT, capsys)
    assert code == 0, err[-3:]
    steps = stack.jsonl("steps.jsonl")
    stale_steps = [i for i, s in enumerate(steps) if not s["sent"] and s["skip_reason"].startswith("camera_stale")]
    assert stale_steps and any(s["sent"] for s in steps[stale_steps[-1]:])     # steps stop, then start again
    assert stack.jsonl("episodes.jsonl")[0]["skips"]["camera_stale"] == len(stale_steps)


def test_odd_chunk_skipped_counted_next_sent(stack, capsys):
    stack.policy.faults[5] = "odd"
    code, err = stack.run(EPISODE % (10, "y") + QUIT, capsys)
    assert code == 0, err[-3:]
    episode = stack.jsonl("episodes.jsonl")[0]
    assert episode["odd_chunks"] == 1 and episode["width_clips"] == 2
    steps = stack.jsonl("steps.jsonl")
    odd = [i for i, s in enumerate(steps) if s["skip_reason"].startswith("odd_chunk")]
    assert len(odd) == 1 and steps[odd[0] + 1]["sent"]


# ---- operator keys ------------------------------------------------------------------------------------------------

def test_s_then_c(stack, capsys):
    script = (EPISODE % (3, "s") + "- {wait_phase: idle}\n- {wait_s: 0.1}\n" + EPISODE % (3, "y") + QUIT)
    code, err = stack.run(script, capsys)
    assert code == 0, err[-3:]
    assert [e["outcome"] for e in stack.jsonl("episodes.jsonl")] == ["stopped", "success"]
    names = stack.names()
    assert names.count("home") == 0 and names.count("start") == 2
    assert stack.policy.calls.count("c4.reset/v1") == 2


def test_q_while_running(stack, capsys):
    code, err = stack.run("- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 3}\n- {key: q}\n", capsys)
    assert code == 0 and stack.names()[-1] == "stop"
    assert stack.jsonl("episodes.jsonl")[0]["outcome"] == "stopped"


# ---- concludes: exit 1, FATAL line last, stop sent first ----------------------------------------------------------

CONCLUDES = {
    "camera down": ("frames_down", "FATAL manager: main camera down: check the Pi and ASUS_5g"),
    "policy hangs": ("policy:hang", "FATAL manager: policy: no reply to infer within 0.5 s: "
                                    "check the policy log and the GPU"),
    "policy error": ("policy:error", "FATAL manager: policy: CUDA error: device lost: check the policy log"),
    "broken chunk": ("policy:broken", "FATAL manager: policy: invalid chunk: eef_rot6d"),
    "arm fault": ("pc:fault", "FATAL manager: the PC reports fault: cartesian_reflex: the PC holds; Ctrl-C and "
                              "restart control_deploy run on the PC"),
    "stops answering": ("frames_silent", "FATAL manager: frames: no reply to"),
    "schema version": ("policy:version", "FATAL manager: c4.chunk/v2: "),
}


@pytest.mark.parametrize("name", list(CONCLUDES))
def test_concludes(stack, capsys, name):
    how, fatal = CONCLUDES[name]
    trigger_at = 4
    if how.startswith("policy:"):
        stack.policy.faults[trigger_at] = how.split(":")[1]
    elif how == "pc:fault":
        when(lambda: stack.sets() >= trigger_at, lambda: stack.pc.inject_fault("cartesian_reflex"))
    else:
        mode = "down" if how == "frames_down" else "silent"
        when(lambda: stack.sets() >= trigger_at, lambda: setattr(stack.frames, "mode", mode))
    code, err = stack.run(EPISODE % (200, "y") + QUIT, capsys)
    assert code == 1
    assert err[-1].startswith(fatal), err[-3:]
    assert sum(line.startswith("FATAL") for line in err) == 1
    assert stop_after_last(stack, "set_waypoints")                 # stop sent after the trigger
    assert stack.jsonl("episodes.jsonl")[0]["outcome"] == "aborted"
