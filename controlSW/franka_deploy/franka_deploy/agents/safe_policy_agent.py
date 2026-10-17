"""SafePolicyAgent: upstream real-env PolicyAgent with every RPC sent once under an explicit timeout (F1).

Only `__init__` is re-implemented (upstream's constructs a plain RMQClient and asks for policy_config with
automatic_resend=True); `predict_actions`, `reset`, `send_prompt_request` are upstream's, unchanged. They call
`self.policy_rmq_client.request_with_data(...)`, which here is a rmq_rpc.NoResendClient: per-topic timeouts from
`rpc_timeouts_s`, never resent, RpcTimeout/RpcError on failure. An "ERROR..." reply still raises upstream's
ValueError/AssertionError. The task turns any of these into a HOLD.

A tap callback (`set_tap`) receives (topic, request_bytes, reply_bytes, seconds) after each successful request.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from robotmq import deserialize, serialize

from real_env.agents.base_agent import BaseAgent
from real_env.agents.policy_agent import PolicyAgent

from franka_deploy.rmq_rpc import NoResendClient

DEFAULT_TIMEOUTS = {"policy_config": 5.0, "policy_inference": 2.0, "policy_reset": 5.0, "policy_prompt": 60.0,
                    "export_recorded_data": 5.0}


class SafePolicyAgent(PolicyAgent):
    def __init__(self, policy_server_endpoint: str, action_as_proprio: bool, keep_one_step_dummy_proprio: bool,
                 use_relative_action: bool, smooth_action_window_size: int, rpc_timeouts_s: dict | None = None,
                 **kwargs):
        timeouts = dict(DEFAULT_TIMEOUTS)
        timeouts.update(dict(rpc_timeouts_s or {}))
        self.rpc_timeouts_s = timeouts
        self.policy_rmq_client = NoResendClient(f"{kwargs['name']}_policy", policy_server_endpoint,
                                                default_timeout_s=5.0, timeouts=timeouts)
        logger.info(f"Requesting policy config from {policy_server_endpoint} (timeout {timeouts['policy_config']} s)")
        self.config: dict[str, Any] = deserialize(
            self.policy_rmq_client.request_with_data(topic="policy_config", data=serialize(True)))
        if isinstance(self.config, str):
            raise RuntimeError(f"policy_config failed: {self.config}")
        kwargs.setdefault("proprio_history_len", self.config["workspace"]["proprio_length"] + 1)
        kwargs.setdefault("action_history_len", self.config["workspace"]["proprio_length"] + 1)
        if self.config["workspace"].get("img_obs_horizon") is not None:
            kwargs["image_history_len"] = self.config["workspace"]["img_obs_horizon"]
        BaseAgent.__init__(self, **kwargs)
        self.action_as_proprio = action_as_proprio
        self.keep_one_step_dummy_proprio = keep_one_step_dummy_proprio
        self.use_relative_action = use_relative_action
        if keep_one_step_dummy_proprio:
            assert use_relative_action, "keep_one_step_dummy_proprio requires use_relative_action"
        assert smooth_action_window_size >= 1 and smooth_action_window_size % 2 == 1
        self.smooth_action_window_size = smooth_action_window_size

    def set_tap(self, fn) -> None:
        self.policy_rmq_client.on_request = fn
