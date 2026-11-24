# copied from rnoro/deployment:deployment/message_comms/message_comms/policy_client.py @ 91b178b
# changes: expects c4.config.reply/v2 (PolicyConfig of policy_deploy.contracts); imports
"""PolicyClient: C4 to policy (STRUCTURE.md 6.2). The config reply is c4.config.reply/v2 (with kind)."""
from __future__ import annotations

from deploy_contracts import Ack, RpcClient

from policy_deploy.contracts import ActionChunk, Observation, PolicyConfig, PolicyConfigRequest, Prompt, Reset
from policy_deploy.manager.errors import Refused
from policy_deploy.manager.settings import Timeouts

PROCESS = "policy"


class PolicyClient:
    def __init__(self, endpoint: str, timeouts: Timeouts) -> None:
        self._rpc = RpcClient(PROCESS, endpoint)
        self._timeouts = timeouts

    def config(self) -> PolicyConfig:
        return self._rpc.request(PolicyConfigRequest(), PolicyConfig, self._timeouts.config_s)

    def reset(self) -> None:
        self._ack(self._rpc.request(Reset(), Ack, self._timeouts.reset_s), Reset.SCHEMA)

    def prompt(self, task_name: str, prompt_index: int) -> None:
        msg = Prompt(task_name=task_name, prompt_index=prompt_index)
        self._ack(self._rpc.request(msg, Ack, self._timeouts.prompt_s), Prompt.SCHEMA)

    def infer(self, obs: Observation) -> ActionChunk:
        return self._rpc.request(obs, ActionChunk, self._timeouts.policy_hang_s)

    def close(self) -> None:
        self._rpc.close()

    @staticmethod
    def _ack(ack: Ack, request: str) -> None:
        if not ack.ok:
            raise Refused(PROCESS, request, ack.reason)
