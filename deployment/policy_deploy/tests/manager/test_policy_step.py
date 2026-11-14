# adapted from rnoro/deployment:deployment/message_comms/tests/test_policy_step.py @ 91b178b
# changes: states on the PC clock (laptop + 1000 s), a real PC clock map; the anchor and waypoints carry PC time;
# + one snapshot for both rows and now, CLOCK_STALE, the state age on the mapped now (state_stale_s); imports.
import numpy as np
import pytest

from deploy_contracts import ContractError, ErrorReply, RemoteError, TransportTimeout
from manager_fakes import (PC_OFFSET_NS, FakeClock, FakeFramesClient, FakePolicyClient, health, make_chunk, make_state,
                           pose, rz)

from policy_deploy.manager.pc_clock import HelloSample, make_pc_clock
from policy_deploy.manager.policy_step import NoNewFrame, PolicyStep, StepDone, StepJob, StepSkipped
from policy_deploy.manager.pose_converter import PoseConverter
from policy_deploy.manager.problems import ProblemCode
from policy_deploy.manager.state_history import StateHistory

MS = 1_000_000


class Rig:
    def __init__(self, settings, T=None, target=None):
        self.clock = FakeClock()
        self.history = StateHistory(settings.history_s)
        self.frames = FakeFramesClient(self.clock)
        self.policy = FakePolicyClient()
        self.pc_clock = make_pc_clock("round_trip", settings.clock, self.clock)
        self.warm()
        self.step = PolicyStep(self.frames, self.policy, self.history,
                               PoseConverter(settings.exec_h, settings.lookahead_ms), settings, self.clock,
                               self.pc_clock)
        self.seq = 0
        self.fill(T, target)

    def warm(self, offset_ns=PC_OFFSET_NS, seconds=2.5, end_ns=None):
        """Hello samples every 50 ms up to `end_ns` (default now): the PC clock = laptop + offset_ns."""
        end = self.clock() if end_ns is None else end_ns
        n = int(seconds / 0.05) + 1
        for i in range(n):
            t = end - (n - 1 - i) * 50 * MS
            self.pc_clock.add_hello(HelloSample(t - MS, t + offset_ns, t + MS))

    def fill(self, T=None, target=None, n=50, skip_seq_at=None):
        """n states 10 ms apart up to now, on the PC clock (laptop + PC_OFFSET_NS)."""
        start = self.clock() - (n - 1) * 10 * MS
        for i in range(n):
            self.seq += 2 if skip_seq_at is not None and i == skip_seq_at else 1
            self.history.add(make_state(self.seq, start + i * 10 * MS + PC_OFFSET_NS, T=T, target=target))

    def job(self, last=None, chunk_id=1):
        return StepJob(1, chunk_id, pose(), last)


def test_normal_step(settings):
    rig = Rig(settings, T=pose((0.5, 0, 0.3), rz(90)))
    out = rig.step.run(rig.job(chunk_id=4))
    assert isinstance(out, StepDone)
    t_a = rig.clock() - 40 * MS
    assert out.anchor.t_ns == t_a and out.anchor.source == "target"
    assert out.anchor.t_pc_ns == t_a + PC_OFFSET_NS                         # the waypoints are on the PC clock
    assert out.waypoints.chunk_id == 4 and len(out.waypoints.t_ns) == settings.exec_h
    assert out.waypoints.t_ns[0] == t_a + PC_OFFSET_NS
    assert list(out.main_t_pc_ns) == [t_a - 50 * MS + PC_OFFSET_NS, t_a + PC_OFFSET_NS]
    assert np.allclose(out.waypoints.O_T_TCP[0], pose((0.5, 0, 0.3), rz(90)), atol=1e-6)
    assert out.timing.image_age_ms == pytest.approx(40.0)
    assert out.anchor_dev_m == 0.0
    assert rig.policy.calls == ["infer"]


@pytest.mark.parametrize("main, ultrawide, code", [("stale", "ok", ProblemCode.CAMERA_STALE),
                                                     ("ok", "stale", ProblemCode.CAMERA_STALE),
                                                     ("down", "stale", ProblemCode.CAMERA_DOWN)])
def test_camera_not_ok(settings, main, ultrawide, code):
    rig = Rig(settings)
    rig.frames.report = health(main, ultrawide)
    out = rig.step.run(rig.job())
    assert isinstance(out, StepSkipped) and out.problem.code is code and out.width_clipped == 0
    assert rig.policy.calls == []


def test_same_anchor_as_last_is_no_new_frame(settings):
    rig = Rig(settings)
    out = rig.step.run(rig.job(last=rig.clock() - 40 * MS))
    assert isinstance(out, NoNewFrame)


def test_old_image(settings):
    rig = Rig(settings)
    rig.frames.latency_ms = settings.max_image_age_ms + 10
    out = rig.step.run(rig.job())
    assert out.problem.code is ProblemCode.IMAGE_OLD


def test_old_state(settings):
    rig = Rig(settings)
    rig.clock.advance(0.15)
    rig.frames.latency_ms = 190           # images still inside the history; the newest state is 150 ms old (mapped)
    out = rig.step.run(rig.job())
    assert out.problem.code is ProblemCode.STATE_OLD and "mapped" in out.problem.detail


def test_state_age_uses_the_mapped_now_with_state_stale_s(settings):
    assert settings.state_stale_s == 0.1                                    # link.yaml
    rig = Rig(settings)
    rig.clock.advance(0.09)               # newest state 90 ms old: not stale
    rig.frames.latency_ms = 130
    assert isinstance(rig.step.run(rig.job()), StepDone)
    rig = Rig(settings)
    rig.pc_clock = rig.step._pc_clock = make_pc_clock("round_trip", settings.clock, rig.clock)
    rig.warm(offset_ns=PC_OFFSET_NS + 200 * MS)     # a map 200 ms ahead: the mapped now is 200 ms past the newest C2
    out = rig.step.run(rig.job())
    assert isinstance(out, StepSkipped) and out.problem.code is ProblemCode.STATE_OLD


def test_clock_not_ready_is_clock_stale(settings):
    rig = Rig(settings)
    rig.clock.advance(settings.clock.max_sample_age_s + 0.1)   # no hello for longer than max_sample_age_s
    rig.fill()
    out = rig.step.run(rig.job())
    assert isinstance(out, StepSkipped) and out.problem.code is ProblemCode.CLOCK_STALE
    assert rig.policy.calls == []


def test_one_snapshot_maps_both_rows_and_now(settings, monkeypatch):
    rig = Rig(settings)
    calls = []
    original = rig.pc_clock.snapshot

    def counting():
        calls.append(1)
        return original()
    monkeypatch.setattr(rig.pc_clock, "snapshot", counting)
    out = rig.step.run(rig.job())
    assert isinstance(out, StepDone) and len(calls) == 1


def test_seq_gap(settings):
    rig = Rig(settings)
    rig.history = StateHistory(1.0)
    rig.step._history = rig.history
    rig.fill(n=50, skip_seq_at=46)        # the gap is between the samples 40 and 30 ms before now
    rig.frames.latency_ms = 35
    out = rig.step.run(rig.job())
    assert out.problem.code is ProblemCode.STATE_GAP


def test_target_far_from_measured_falls_back(settings):
    rig = Rig(settings, T=pose((0.5, 0, 0.3)), target=pose((0.53, 0, 0.3)))
    out = rig.step.run(rig.job())
    assert isinstance(out, StepDone)
    assert out.anchor.source == "measured" and out.anchor_dev_m == pytest.approx(0.03)
    assert np.allclose(out.anchor.T, pose((0.5, 0, 0.3)))


def test_target_close_to_measured_stays_target(settings):
    rig = Rig(settings, T=pose((0.5, 0, 0.3)), target=pose((0.51, 0, 0.3)))
    out = rig.step.run(rig.job())
    assert out.anchor.source == "target" and np.allclose(out.anchor.T, pose((0.51, 0, 0.3)))
    assert out.anchor_dev_m == pytest.approx(0.01)


def chunk_step0(pos_m=0.0, rot_deg=0.0, clipped=0):
    pos = np.zeros((16, 3))
    pos[0, 0] = pos_m
    rot = np.tile(np.eye(3), (16, 1, 1))
    rot[0] = rz(rot_deg)
    return make_chunk(pos=pos, rot=rot, width_clipped=clipped)


@pytest.mark.parametrize("pos_m, rot_deg, odd", [(0.025, 0, True), (0, 6.0, True), (0.019, 4.9, False)])
def test_odd_chunk(settings, pos_m, rot_deg, odd):
    rig = Rig(settings)
    rig.policy.chunk = lambda obs: chunk_step0(pos_m, rot_deg, clipped=2)
    out = rig.step.run(rig.job())
    if odd:
        assert isinstance(out, StepSkipped) and out.problem.code is ProblemCode.ODD_CHUNK
        assert out.width_clipped == 2
    else:
        assert isinstance(out, StepDone)


@pytest.mark.parametrize("error", [
    TransportTimeout("policy", "tcp://x", "c4.infer/v1", 5.0),
    RemoteError("policy", ErrorReply(request="c4.infer/v1", error="boom", field="")),
    ContractError("c4.chunk/v1", "eef_rot6d", "value", "not unit"),
])
def test_policy_errors_propagate(settings, error):
    rig = Rig(settings)
    rig.policy.error = error
    with pytest.raises(type(error)):
        rig.step.run(rig.job())
    assert rig.step.last_call == ("policy", "c4.infer/v1")


def test_frames_error_propagates(settings):
    rig = Rig(settings)
    rig.frames.error = TransportTimeout("frames", "tcp://x", "c1.get_frames/v1", 1.0)
    with pytest.raises(TransportTimeout):
        rig.step.run(rig.job())
    assert rig.step.last_call == ("frames", "c1.get_frames/v1")
