"""Controller client that never hangs (reviewer fix 3) and can latch a controller-side HOLD.

Upstream BaseCartesianJointClient reads `eef_pose` with robotmq's default automatic_resend=True: a dead controller
blocks the task for up to 800 x 1 s. Here the RMQ client is replaced by rmq_rpc.NoResendClient, so every upstream
method (interpolated reads, schedule_*, reset, get_config) is sent once with a bounded timeout and raises
RpcTimeout/RpcError instead. The interpolation logic itself is upstream's, unchanged.

Extras:
- `state_age_s()`: age of the newest eef_pose sample (monotonic clock) -> the task's stale-state guard;
- `hold()` / `release()`: put on the controller's "hold" topic (franka_deploy controllers only). While held the
  controller freezes its target at the current pose and drops scheduled trajectories until released.
"""

from __future__ import annotations

import time

from robotmq import serialize

from real_env.controllers.base_controller_client import BaseCartesianJointClient, BaseControllerClient

from franka_deploy.rmq_rpc import NoResendClient, RpcTimeout


class SafeCartesianJointClient(BaseCartesianJointClient):
    def __init__(self, name: str, server_endpoint: str, data_remaining_time_s: float = 1.0,
                 timeout_s: float = 0.05, connect_timeout_s: float = 10.0):
        BaseControllerClient.__init__(self, name=name, server_endpoint=server_endpoint,
                                      data_remaining_time_s=data_remaining_time_s)
        self.rmq_client = NoResendClient(f"{name}_safe", server_endpoint, default_timeout_s=timeout_s,
                                         timeouts={"config": 1.0})
        self.timeout_s = timeout_s
        self.eef_poses, self.eef_timestamps = [], []
        self.eef_target_poses, self.eef_target_timestamps = [], []
        self.joint_pos, self.joint_timestamps = [], []
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                raw_e, _ = self.rmq_client.peek_data(topic="eef_pose", n=1)
                raw_j, _ = self.rmq_client.peek_data(topic="joint_state", n=1)
                if raw_e and raw_j:
                    from robotmq import deserialize

                    self.dof = len(deserialize(raw_j[0])["joint_pos"])
                    break
            except RpcTimeout:
                pass
            if time.monotonic() > deadline:
                raise RpcTimeout(f"{name}: controller at {server_endpoint} not publishing state within {connect_timeout_s} s")
            time.sleep(0.05)
        # prime the buffers so the first interpolated read has data (targets appear after the first control tick)
        while True:
            try:
                self.get_eef_xyz_wxyz(time.monotonic())
                self.get_joint_pos(time.monotonic())
                self.get_eef_target_xyz_wxyz(time.monotonic())
                break
            except IndexError:
                if time.monotonic() > deadline:
                    raise RpcTimeout(f"{name}: controller at {server_endpoint} publishes no targets")
                time.sleep(0.02)

    def state_age_s(self) -> float:
        return time.monotonic() - self.eef_timestamps[-1] if self.eef_timestamps else float("inf")

    def hold(self, reason: str = "") -> None:
        self.rmq_client.put_data(topic="hold", data=serialize({"hold": True, "reason": reason, "t": time.monotonic()}))

    def release(self) -> None:
        self.rmq_client.put_data(topic="hold", data=serialize({"hold": False, "t": time.monotonic()}))

    def reset(self, timeout_s: float = 30.0):
        from real_env.common.data_classes import ControllerCommand

        self.rmq_client.put_data(topic="command", data=serialize({"command": ControllerCommand.RESET}))
        self.rmq_client.pop_data(topic="eef_pose", n=0)
        self.rmq_client.pop_data(topic="joint_state", n=0)
        self.eef_poses, self.eef_timestamps, self.joint_pos, self.joint_timestamps = [], [], [], []
        deadline = time.monotonic() + timeout_s
        while self.rmq_client.get_topic_status(topic="eef_pose") <= 10:
            if time.monotonic() > deadline:
                raise RpcTimeout(f"{self.name}: no state after reset within {timeout_s} s")
            time.sleep(0.01)
        # wait for the controller's homing motion to finish (franka_deploy controllers publish "homing" on status),
        # like upstream controllers whose reset() blocks
        time.sleep(0.15)
        from robotmq import deserialize

        while time.monotonic() < deadline:
            raw, _ = self.rmq_client.peek_data(topic="status", n=-1)
            if not raw or not deserialize(raw[0]).get("homing"):
                break
            time.sleep(0.05)
        self.get_eef_xyz_wxyz(time.monotonic())
        self.get_joint_pos(time.monotonic())
