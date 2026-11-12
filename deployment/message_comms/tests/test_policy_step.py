import numpy as np
import pytest

from contracts import ContractError, ErrorReply, RemoteError, TransportTimeout
from fakes import FakeClock, FakeFramesClient, FakePolicyClient, health, make_chunk, make_state, pose, rz
from message_comms.policy_step import NoNewFrame, PolicyStep, StepDone, StepJob, StepSkipped
from message_comms.pose_converter import PoseConverter
from message_comms.problems import ProblemCode
from message_comms.state_history import StateHistory

MS = 1_000_000


class Rig:
    def __init__(self, settings, T=None, target=None):
        self.clock = FakeClock()
        self.history = StateHistory(settings.history_s)
        self.frames = FakeFramesClient(self.clock)
        self.policy = FakePolicyClient()
        self.step = PolicyStep(self.frames, self.policy, self.history,
                               PoseConverter(settings.exec_h, settings.lookahead_ms), settings, self.clock)
        self.seq = 0
        self.fill(T, target)

    def fill(self, T=None, target=None, n=50, skip_seq_at=None):
        start = self.clock() - (n - 1) * 10 * MS
        for i in range(n):
            self.seq += 2 if skip_seq_at is not None and i == skip_seq_at else 1
            self.history.add(make_state(self.seq, start + i * 10 * MS, T=T, target=target))

    def job(self, last=None, chunk_id=1):
        return StepJob(1, chunk_id, pose(), last)


def test_normal_step(settings):
    rig = Rig(settings, T=pose((0.5, 0, 0.3), rz(90)))
    out = rig.step.run(rig.job(chunk_id=4))
    assert isinstance(out, StepDone)
    t_a = rig.clock() - 40 * MS
    assert out.anchor.t_ns == t_a and out.anchor.source == "target"
    assert out.waypoints.chunk_id == 4 and len(out.waypoints.t_ns) == settings.exec_h
    assert out.waypoints.t_ns[0] == t_a
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
    rig.clock.advance(0.1)
    rig.frames.latency_ms = 140           # images still inside the history; the newest state is 100 ms old
    out = rig.step.run(rig.job())
    assert out.problem.code is ProblemCode.STATE_OLD


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
    rig.frames.error = TransportTimeout("process_frames", "tcp://x", "c1.get_frames/v1", 1.0)
    with pytest.raises(TransportTimeout):
        rig.step.run(rig.job())
    assert rig.step.last_call == ("process_frames", "c1.get_frames/v1")
