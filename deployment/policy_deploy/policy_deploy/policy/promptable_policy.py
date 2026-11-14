# copied from rnoro/deployment:deployment/policy/policy/promptable_policy.py @ 91b178b
# changes: imports only
"""PromptablePolicy: a Policy that takes one prompt after each reset."""

from __future__ import annotations

from abc import abstractmethod

from policy_deploy.policy.base_policy import Policy


class PromptablePolicy(Policy):
    USES_PROMPT = True

    @abstractmethod
    def prompt(self, task_name: str, prompt_index: int) -> None:
        """Apply the prompt. Raises BadRequest for an unknown task or index."""
