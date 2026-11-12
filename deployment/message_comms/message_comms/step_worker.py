"""StepWorker: the thread that runs policy jobs, one at a time, so the main loop never waits (STRUCTURE.md 4.6)."""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

from message_comms.policy_client import PolicyClient
from message_comms.policy_step import PolicyStep, StepJob

NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class WorkerResult:
    kind: str                     # "config" | "reset" | "step"
    value: object | None          # PolicyConfig | None | StepDone | StepSkipped | NoNewFrame
    error: BaseException | None   # set instead of value when the job raised
    process: str                  # whom the failing request went to: "policy" | "process_frames"
    request: str                  # the schema of the failing request, for the message
    started_ns: int
    finished_ns: int


class StepWorker:
    def __init__(self, step: PolicyStep, policy: PolicyClient, clock: Callable[[], int]) -> None:
        self._step = step
        self._policy = policy
        self._clock = clock
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._results: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="step-worker", daemon=True)
        self._busy = False
        self._started_ns: int | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Asks the thread to exit; does not wait for a hung infer (daemon thread)."""
        self._stop.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass

    def busy(self) -> bool:
        return self._busy

    def submit_config(self) -> None:
        self._submit(("config", None))

    def submit_reset(self, prompt: tuple[str, int] | None) -> None:
        """reset, then prompt if given."""
        self._submit(("reset", prompt))

    def submit_step(self, job: StepJob) -> None:
        self._submit(("step", job))

    def wait(self, timeout_s: float) -> WorkerResult | None:
        """Main thread: the result, or None after timeout_s (also when idle: then it only waits)."""
        if self._busy and not self._thread.is_alive() and self._results.empty():
            raise RuntimeError("the step worker thread died")
        try:
            result = self._results.get(timeout=timeout_s) if timeout_s > 0 else self._results.get_nowait()
        except queue.Empty:
            return None
        self._busy = False
        self._started_ns = None
        return result

    def running_for_ms(self) -> float | None:
        started = self._started_ns
        return None if started is None else (self._clock() - started) / NS_PER_MS

    def _submit(self, job: tuple[str, object]) -> None:
        if self._busy:
            raise RuntimeError(f"step worker busy: cannot submit {job[0]}")
        self._busy = True
        self._jobs.put_nowait(job)

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._jobs.get()
            if job is None or self._stop.is_set():
                return
            kind, payload = job
            self._started_ns = started = self._clock()
            where = ("policy", "c4.config/v1")
            value = error = None
            try:
                if kind == "config":
                    value = self._policy.config()
                elif kind == "reset":
                    where = ("policy", "c4.reset/v1")
                    self._policy.reset()
                    if payload is not None:
                        where = ("policy", "c4.prompt/v1")
                        self._policy.prompt(*payload)
                else:
                    value = self._step.run(payload)
            except BaseException as exc:          # passed back to the main thread, which concludes
                error = exc
                if kind == "step":
                    where = self._step.last_call
            self._results.put(WorkerResult(kind, value, error, where[0], where[1], started, self._clock()))
