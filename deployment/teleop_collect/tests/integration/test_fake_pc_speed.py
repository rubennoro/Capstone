"""The fake PC follows teleop's waypoints at the commanded speed (E7 seam fix in fake_pc.py, PLAN 5.2).

Teleop replaces the PC's plan every 20 ms with waypoints on the line base + v (t - t_base), the base being the newest
C2 target. The real franka_control evaluates its plan at 1 kHz and starts a new plan from plan.at(now), so the target
moves at v (measured 50.0 mm/s for v_max 50 mm/s, test_cross_franka_control.py). The fake PC ticks at 100 Hz; before
the fix an accepted plan started from the last tick's target stamped as "now", so every replacement lost the time
since that tick (here 7 of every 20 ms: about 65 % of v)."""
from __future__ import annotations

import numpy as np
from deploy_contracts import SetWaypoints, Start
from launcher_helpers import FakeLaptopClock, make_fake_pc

V = 0.05                       # m/s along +x (v_max)
K, SPACING_NS = 6, 50_000_000  # teleop.yaml waypoints (6 since wave 3) and spacing_ms


class _Recorder:
    def __init__(self) -> None:
        self.states = []

    def publish(self, msg) -> None:
        self.states.append(msg)

    def close(self) -> None:
        pass


def test_a_plan_replaced_every_20_ms_moves_the_target_at_the_commanded_speed(tmp_path):
    clock = FakeLaptopClock()
    fake = make_fake_pc(tmp_path, laptop_clock=clock)
    fake._pub = _Recorder()
    fake.tick()
    assert fake.handle(Start()).ok
    xs, ts = [], []
    for ms in range(1, 1201):
        clock.advance(0.001)
        if ms % 10 == 0:                                  # the fake PC's 100 Hz C2 tick
            fake.tick()
            st = fake._pub.states[-1]
            xs.append(float(st.O_T_TCP_target[0, 3]))
            ts.append(st.t_ns)
        if ms % 20 == 7:                                  # teleop's 50 Hz set, 7 ms after a tick
            base = fake._pub.states[-1]
            now_pc = fake.clock()
            t = np.array([now_pc + k * SPACING_NS for k in range(1, K + 1)], dtype=np.int64)
            poses = np.repeat(np.asarray(base.O_T_TCP_target)[None], K, axis=0)
            poses[:, 0, 3] += V * (t - base.t_ns) / 1e9
            reply = fake.handle(SetWaypoints(chunk_id=ms, t_ns=t, O_T_TCP=poses,
                                             gripper_width_m=np.full(K, base.gripper_width_m)))
            assert reply.accepted == K, reply
    t_s = (np.array(ts[20:]) - ts[20]) / 1e9                 # after the first 0.2 s
    slope = np.polyfit(t_s, np.array(xs[20:]), 1)[0]
    assert 0.98 * V <= slope <= 1.01 * V, slope
