"""FrankaHandDriver: the controller's gripper driver for a Franka Hand, NON-BLOCKING (I-HAND client).

Why: FrankaCartesian calls ``get_width()`` at 200 Hz and ``set_width()`` at 50 Hz inside its control loop, while
libfranka's Gripper calls block for 0.3-1.5 s. So ``set_width`` only stores the latest training width and
``get_width`` returns a cached value; a background thread with its OWN REQ socket (lazy pirate BridgeClient) does,
every 1 / rate_hz: ``hand_state`` -> HandCommandPolicy.update(latest width) -> ``hand_cmd`` (increasing seq).
RPC timeouts and errors are counted (``status()``), never raised into the controller. The code is identical in
sim and on hardware: the endpoint is the bridge (MuJoCo serves hand_*) or hand_server (5566).

Config (the merged controller ``gripper`` dict; servo keys such as calibration/squeeze_*/current_limit are ignored):
  type: franka_hand, endpoint, width_map {a, b, gap_min_m, gap_max_m}, speed_m_s, force_n, force_max_n,
  grasp_epsilon_inner_m, grasp_epsilon_outer_m, optional policy {close_below_train_m, open_above_train_m,
  move_deadband_m, min_cmd_interval_s}, optional rate_hz (50), rpc_timeout_s (0.1), stale_s (0.2),
  first_state_wait_s (1.0).
"""

from __future__ import annotations

import threading
import time

from franka_deploy.gripper.driver import GripperDriver, WidthCalibration
from franka_deploy.gripper.hand_logic import HandCommandPolicy, HandPolicyConfig, WidthMap


class FrankaHandDriver(GripperDriver):
    def __init__(self, endpoint: str, width_map: WidthMap, policy: HandPolicyConfig, rate_hz: float = 50.0,
                 rpc_timeout_s: float = 0.1, stale_s: float = 0.2, first_state_wait_s: float = 1.0):
        # identity calibration: training width in, training width out (existing .cal users, e.g. trace_extra)
        super().__init__(WidthCalibration([0.0, 0.2], [0.0, 0.2]), squeeze_delta_m=0.0, min_width_m=0.0,
                         max_width_m=0.2)
        self.endpoint = endpoint
        self.wm = width_map
        self.policy = HandCommandPolicy(policy, width_map)
        self.period = 1.0 / float(rate_hz)
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.stale_s = float(stale_s)
        self._lock = threading.Lock()
        self._w_cmd = None                       # latest training width from set_width
        self._width = width_map.train(width_map.gap_max_m)
        self._state = None
        self._state_t = None                     # local monotonic of the last good hand_state reply
        self._target_gap = None
        self._seq = 0
        self.counts = {"cmds_sent": 0, "rpc_timeouts": 0, "rpc_errors": 0, "stale_reads": 0, "loops": 0}
        self.last_error = None
        self._run = True
        self._first = threading.Event()
        self._th = threading.Thread(target=self._loop, name="franka-hand", daemon=True)
        self._th.start()
        if first_state_wait_s > 0 and not self._first.wait(first_state_wait_s):
            print(f"[franka_hand] no hand_state from {endpoint} within {first_state_wait_s:.1f} s; get_width() "
                  f"returns {self._width:.4f} (open) until it answers", flush=True)

    @classmethod
    def from_config(cls, cfg: dict) -> "FrankaHandDriver":
        if not cfg.get("endpoint"):
            raise ValueError("gripper type franka_hand needs 'endpoint' (the bridge in sim, hand_server 5566 on hardware)")
        return cls(str(cfg["endpoint"]), WidthMap.from_config(cfg.get("width_map")), HandPolicyConfig.from_config(cfg),
                   rate_hz=float(cfg.get("rate_hz", 50.0)), rpc_timeout_s=float(cfg.get("rpc_timeout_s", 0.1)),
                   stale_s=float(cfg.get("stale_s", 0.2)), first_state_wait_s=float(cfg.get("first_state_wait_s", 1.0)))

    # ------------------------------------------------------------------------------- controller API (no I/O)
    def set_width(self, width_m: float) -> None:
        w = float(width_m)
        self._commanded = w
        self._w_cmd = w                          # one attribute store: atomic under the GIL

    def commanded_width(self):
        return self._commanded

    def get_width(self) -> float:
        t = self._state_t
        if t is None or time.monotonic() - t > self.stale_s:
            self.counts["stale_reads"] += 1
        return self._width

    def target_position(self) -> float:
        """Training width of the last command sent (identity cal: trace width_servo_target = the hand's target)."""
        g = self._target_gap
        if g is None:
            return self._width
        return self.wm.train(g)

    def set_current_limit(self, limit: float) -> None:     # servo concept; the Hand grasps with a force
        pass

    def status(self) -> dict:
        st = self._state or {}
        t = self._state_t
        return {**self.counts, "clips": self.policy.clips, "regrasps": self.policy.regrasps, "last_result": st.get("last_result"),
                "grasped": st.get("grasped"), "mode": st.get("mode"), "policy_state": self.policy.state,
                "state_age_s": None if t is None else time.monotonic() - t, "last_error": self.last_error}

    def close(self) -> None:
        self._run = False
        self._th.join(1.0 + 2 * self.rpc_timeout_s)

    # ------------------------------------------------------------------------------------- background thread
    def _rpc(self, client, topic, data=None):
        from franka_deploy.rmq_rpc import RpcTimeout

        try:
            return True, client.request(topic, data)
        except RpcTimeout as e:
            self.counts["rpc_timeouts"] += 1
            self.last_error = str(e)
        except Exception as e:
            self.counts["rpc_errors"] += 1
            self.last_error = f"{type(e).__name__}: {e}"
        return False, None

    def _loop(self) -> None:
        from franka_deploy.bridge.link import BridgeClient

        client = BridgeClient(self.endpoint, timeout_s=self.rpc_timeout_s)
        nxt = time.monotonic()
        try:
            while self._run:
                self.counts["loops"] += 1
                ok, st = self._rpc(client, "hand_state")
                now = time.monotonic()
                if ok and isinstance(st, dict):
                    self._state, self._state_t = st, now
                    g = st.get("gap_m")
                    if g is not None and g == g:              # not NaN
                        self._width = self.wm.train(float(g))
                    self._first.set()
                fresh = self._state if (self._state_t is not None and now - self._state_t <= self.stale_s) else None
                w = self._w_cmd
                cmd = self.policy.update(w, now, fresh) if w is not None else None
                if cmd is not None:
                    self._seq += 1
                    ok, rep = self._rpc(client, "hand_cmd", cmd.to_msg(self._seq))
                    if ok and isinstance(rep, dict) and rep.get("accepted", False):
                        self.counts["cmds_sent"] += 1
                        self._target_gap = cmd.width_m
                    else:
                        if ok:
                            self.last_error = f"hand_cmd not accepted: {rep!r}"
                        self.policy.send_failed()
                nxt += self.period
                delay = nxt - time.monotonic()
                if delay < -self.period:                    # fell behind (dead server): restart the schedule
                    nxt = time.monotonic()
                    delay = 0.0
                time.sleep(max(0.0, delay))
        finally:
            client.close()
