"""PLAN section 7 "message flow": the real message_comms main() against fake process_frames, policy and
franka_control on real sockets, with a script. The normal step and every row of STRUCTURE.md section 7."""
import json

import pytest

from conftest import write_station
from fake_servers import FakeFrankaControl, FakePolicy, FakeProcessFrames
from message_comms.__main__ import main

EPISODE = "- {wait_phase: idle}\n- {key: c}\n- {wait_steps: %d}\n- {key: %s}\n"
QUIT = "- {wait_phase: idle}\n- {key: q}\n"


class Stack:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.frames = FakeProcessFrames()
        self.policy = FakePolicy()
        self.franka = FakeFrankaControl()
        self.session = tmp_path / "session"

    def run(self, script: str, capsys, **overrides):
        (self.tmp / "script.yaml").write_text(script)
        endpoints = {"frames": self.frames.endpoint, "state": self.franka.state_endpoint,
                     "commands": self.franka.commands_endpoint, "policy": self.policy.endpoint}
        settings = {"operator_script": "script.yaml", "timeouts": {"startup_s": 5.0, "policy_hang_s": 0.5}}
        settings.update(overrides)
        station = write_station(self.tmp, settings, endpoints)
        code = main(["--station", str(station), "--session", str(self.session)])
        err = capsys.readouterr().err
        return code, [line for line in err.splitlines() if line.strip()]

    def jsonl(self, name):
        path = self.session / "message_comms" / name
        return [json.loads(x) for x in path.read_text().splitlines()]

    def close(self):
        for fake in (self.frames, self.policy, self.franka):
            fake.close()


@pytest.fixture
def stack(tmp_path):
    s = Stack(tmp_path)
    yield s
    s.close()


def stop_after_last(stack, name):
    """True if a stop request came after the last `name` request."""
    names = stack.franka.names()
    positions = [i for i, n in enumerate(names) if n == name]
    return bool(positions) and "stop" in names[positions[-1]:]


# ---- the normal step ----------------------------------------------------------------------------------------------

def test_one_episode(stack, capsys):
    code, err = stack.run(EPISODE % (10, "y") + QUIT, capsys)
    assert code == 0
    assert not any(line.startswith("FATAL") for line in err)
    ids = [s.chunk_id for s in stack.franka.sets]
    assert len(ids) >= 10 and ids == sorted(set(ids))
    beats = [t for name, t in stack.franka.requests if name == "heartbeat"]
    assert len(beats) > 10 and max(b - a for a, b in zip(beats, beats[1:])) < 420_000_000   # under the watchdog
    names = stack.franka.names()
    assert names[-1] == "stop"                          # stop is sent first at exit
    episodes = stack.jsonl("episodes.jsonl")
    assert [e["outcome"] for e in episodes] == ["success"] and episodes[0]["steps_sent"] >= 10
    assert stack.policy.calls[:2] == ["c4.config/v1", "c4.reset/v1"]
    assert "c4.prompt/v1" not in stack.policy.calls


# ---- slow cases: keep running, exit 0 -----------------------------------------------------------------------------

def test_camera_stale_then_resume(stack, capsys):
    import threading
    steps_at_stale = {}

    def watch():
        while len(stack.franka.sets) < 3:
            threading.Event().wait(0.005)
        stack.frames.mode = "stale"
        steps_at_stale["n"] = len(stack.franka.sets)
        threading.Event().wait(0.3)
        stack.frames.mode = "ok"
    threading.Thread(target=watch, daemon=True).start()
    code, err = stack.run(EPISODE % (12, "y") + QUIT, capsys)
    assert code == 0
    steps = stack.jsonl("steps.jsonl")
    stale = [i for i, s in enumerate(steps) if not s["sent"] and s["skip_reason"].startswith("camera_stale")]
    assert stale and any(s["sent"] for s in steps[stale[-1]:])     # steps stop, then start again
    assert stack.jsonl("episodes.jsonl")[0]["skips"]["camera_stale"] == len(stale)


def test_state_gap_skips_that_step(stack, capsys):
    stack.franka.faults["gap_after_sets"] = 3
    code, err = stack.run(EPISODE % (12, "y") + QUIT, capsys)
    assert code == 0
    reasons = [s["skip_reason"] for s in stack.jsonl("steps.jsonl") if not s["sent"]]
    assert any(r.startswith("state_gap") for r in reasons), reasons


def test_odd_chunk_skipped_counted_next_sent(stack, capsys):
    stack.policy.faults[5] = "odd"
    code, err = stack.run(EPISODE % (10, "y") + QUIT, capsys)
    assert code == 0
    episode = stack.jsonl("episodes.jsonl")[0]
    assert episode["odd_chunks"] == 1 and episode["width_clips"] == 2
    steps = stack.jsonl("steps.jsonl")
    odd = [i for i, s in enumerate(steps) if s["skip_reason"].startswith("odd_chunk")]
    assert len(odd) == 1 and steps[odd[0] + 1]["sent"]


# ---- operator keys ------------------------------------------------------------------------------------------------

def test_s_then_h_then_c(stack, capsys):
    script = (EPISODE % (3, "s") + "- {wait_phase: idle}\n- {key: h}\n- {wait_s: 0.1}\n" + EPISODE % (3, "y") + QUIT)
    code, err = stack.run(script, capsys)
    assert code == 0
    assert [e["outcome"] for e in stack.jsonl("episodes.jsonl")] == ["stopped", "success"]
    names = stack.franka.names()
    assert names.count("home") == 1 and names.count("start") == 2
    assert stack.policy.calls.count("c4.reset/v1") == 2


def test_q_while_running(stack, capsys):
    code, err = stack.run("- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 3}\n- {key: q}\n", capsys)
    assert code == 0 and stack.franka.names()[-1] == "stop"
    assert stack.jsonl("episodes.jsonl")[0]["outcome"] == "stopped"


# ---- concludes: exit 1, FATAL line last, stop sent first ----------------------------------------------------------

CONCLUDES = {
    "camera down": ("frames_down", "FATAL message_comms: main camera down: check the Pi stream and cable"),
    "policy hangs": ("policy:hang", "FATAL message_comms: policy: no reply to infer within 0.5 s: "
                                    "check the policy log and the GPU"),
    "policy error": ("policy:error", "FATAL message_comms: policy: CUDA error: device lost: check the policy log"),
    "broken chunk": ("policy:broken", "FATAL message_comms: policy: invalid chunk: eef_rot6d"),
    "arm fault": ("franka:fault_after_sets", "FATAL message_comms: franka_control fault: cartesian_reflex: "
                                             "check the arm (reflex, connection)"),
    "gate rejection": ("franka:reject_at_set", "FATAL message_comms: franka_control: gate rejected waypoint 7: "
                                               "outside the box by 3 cm: check the workspace box and the policy"),
    "stops answering": ("frames_silent", "FATAL message_comms: process_frames: no reply to"),
    "schema version": ("policy:version", "FATAL message_comms: c4.chunk/v2: "),
    "broken C2 message": ("franka:broken_c2_after_sets", "FATAL message_comms: c2.state/v1: field 'O_T_TCP': "),
}


@pytest.mark.parametrize("name", list(CONCLUDES))
def test_concludes(stack, capsys, name):
    import threading
    how, fatal = CONCLUDES[name]
    trigger_at = 4
    if how.startswith("policy:"):
        stack.policy.faults[trigger_at] = how.split(":")[1]
    elif how.startswith("franka:"):
        stack.franka.faults[how.split(":")[1]] = trigger_at
    else:
        def watch():
            while len(stack.franka.sets) < trigger_at:
                threading.Event().wait(0.005)
            stack.frames.mode = "down" if how == "frames_down" else "silent"
        threading.Thread(target=watch, daemon=True).start()
    code, err = stack.run(EPISODE % (200, "y") + QUIT, capsys)
    assert code == 1
    assert err[-1].startswith(fatal), err[-3:]
    assert sum(line.startswith("FATAL") for line in err) == 1
    assert stop_after_last(stack, "set_waypoints")                 # stop sent after the trigger
    episode = stack.jsonl("episodes.jsonl")[0]
    assert episode["outcome"] == "aborted"
