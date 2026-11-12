"""PolicyServer: answers C4 for one Policy and enforces the reset / prompt order (STRUCTURE.md 4.8)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from contracts import (Ack, ActionChunk, ContractError, Observation, PolicyConfig, PolicyConfigRequest, Prompt, Reset,
                       RpcServer)

from policy.base_policy import Policy
from policy.errors import BadRequest, PolicyFatal
from policy.promptable_policy import PromptablePolicy

log = logging.getLogger("policy.server")

NEEDS_RESET = "needs_reset"
NEEDS_PROMPT = "needs_prompt"
READY = "ready"


class EpisodeStats:
    """Inference count and times, and clipped widths, since the last reset."""

    def __init__(self) -> None:
        self.inferences = 0
        self.total_ns = 0
        self.max_ns = 0
        self.width_clipped = 0

    def add(self, dt_ns: int, width_clipped: int) -> None:
        self.inferences += 1
        self.total_ns += dt_ns
        self.max_ns = max(self.max_ns, dt_ns)
        self.width_clipped += width_clipped

    def summary(self) -> str:
        if self.inferences == 0:
            return "0 inferences"
        mean_ms = self.total_ns / self.inferences / 1e6
        return (f"{self.inferences} inferences, mean {mean_ms:.1f} ms, max {self.max_ns / 1e6:.1f} ms, "
                f"{self.width_clipped} clipped widths")


class PolicyServer:
    def __init__(self, policy: Policy, endpoint: str, reset_max_s: float,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._policy = policy
        self._reset_max_ns = int(reset_max_s * 1e9)
        self._clock = clock
        self._state = NEEDS_RESET
        self._stats = EpisodeStats()
        self._fatal: PolicyFatal | None = None
        self._server = RpcServer(endpoint, {
            PolicyConfigRequest: self._on_config,
            Reset: self._on_reset,
            Prompt: self._on_prompt,
            Observation: self._on_infer,
        })
        self.endpoint: str = self._server.endpoint

    def serve(self, stop: threading.Event, poll_s: float) -> None:
        """Answer requests until `stop` is set. Raises PolicyFatal (after its ErrorReply went out)."""
        while not stop.is_set():
            self._server.serve_once(timeout_s=poll_s)
            if self._fatal is not None:
                raise self._fatal

    def close(self) -> None:
        self._server.close()

    # -- handlers ---------------------------------------------------------------------------------------------------

    def _on_config(self, request: object) -> PolicyConfig:
        return self._guard("config", self._policy.config)

    def _on_reset(self, request: object) -> Ack:
        t0 = self._clock()
        self._guard("reset", self._policy.reset)
        dt = self._clock() - t0
        if dt > self._reset_max_ns:
            log.warning("reset took %.3f s (limit %.3f s)", dt / 1e9, self._reset_max_ns / 1e9)
        log.info("reset in %.1f ms; episode before it: %s", dt / 1e6, self._stats.summary())
        self._stats = EpisodeStats()
        self._state = NEEDS_PROMPT if isinstance(self._policy, PromptablePolicy) else READY
        return Ack(ok=True, reason="")

    def _on_prompt(self, request: object) -> Ack:
        assert isinstance(request, Prompt)
        policy = self._policy
        if not isinstance(policy, PromptablePolicy):
            raise BadRequest(f"prompt: this policy (kind {policy.KIND}) does not use prompts "
                             f"(config uses_prompt: false)")
        if self._state == NEEDS_RESET:
            raise BadRequest("prompt before reset")
        if self._state == READY:
            raise BadRequest("prompt already given since the last reset")
        self._guard("prompt", lambda: policy.prompt(request.task_name, request.prompt_index))
        self._state = READY
        log.info("prompt: task %r, index %d", request.task_name, request.prompt_index)
        return Ack(ok=True, reason="")

    def _on_infer(self, request: object) -> ActionChunk:
        assert isinstance(request, Observation)
        if self._state == NEEDS_RESET:
            raise BadRequest("infer before reset")
        if self._state == NEEDS_PROMPT:
            raise BadRequest("infer before prompt")
        t0 = self._clock()
        chunk = self._guard("infer", lambda: self._policy.infer(request))
        self._stats.add(self._clock() - t0, int(chunk.width_clipped))
        return chunk

    def _guard(self, request: str, fn: Callable[[], object]):
        """Run a policy call: BadRequest passes through; anything else becomes the stored PolicyFatal."""
        try:
            return fn()
        except BadRequest:
            raise
        except ContractError as e:
            fatal = PolicyFatal(f"own reply broke C4: {e}",
                                "the model output is invalid; check the checkpoint and its inputs")
            self._set_fatal(fatal, e)
            raise fatal from e
        except Exception as e:
            fatal = _fatal_for(request, e)
            self._set_fatal(fatal, e)
            raise fatal from e

    def _set_fatal(self, fatal: PolicyFatal, cause: BaseException) -> None:
        log.error("fatal during request: %s", fatal, exc_info=cause)
        self._fatal = fatal


def _fatal_for(request: str, e: BaseException) -> PolicyFatal:
    text = f"{type(e).__name__}: {e}"
    if "cuda" in text.lower():
        return PolicyFatal(f"CUDA error during {request}: {text}", "check GPU memory, or set policy.device: cpu")
    return PolicyFatal(f"{request} failed: {text}", "see the traceback in the log above")
