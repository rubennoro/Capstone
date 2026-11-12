import threading
import time

import pytest

from fakes import FakePolicyClient, pose
from message_comms.policy_step import StepJob
from message_comms.step_worker import StepWorker


class GatedStep:
    """run() blocks until the test opens the gate; returns the job's chunk id."""

    def __init__(self):
        self.gate = threading.Event()
        self.last_call = ("policy", "c4.infer/v1")
        self.error = None

    def run(self, job):
        self.gate.wait(5.0)
        if self.error:
            raise self.error
        return job.chunk_id


def job(i):
    return StepJob(1, i, pose(), None)


@pytest.fixture
def parts():
    step, policy = GatedStep(), FakePolicyClient()
    worker = StepWorker(step, policy, time.monotonic_ns)
    worker.start()
    yield step, policy, worker
    step.gate.set()
    worker.stop()


def test_one_job_at_a_time(parts):
    step, _, worker = parts
    worker.submit_step(job(1))
    assert worker.busy()
    with pytest.raises(RuntimeError):
        worker.submit_step(job(2))
    with pytest.raises(RuntimeError):
        worker.submit_config()
    assert worker.wait(0.05) is None and worker.busy()
    assert worker.running_for_ms() is not None
    step.gate.set()
    result = worker.wait(5.0)
    assert result.kind == "step" and result.value == 1 and result.error is None
    assert result.finished_ns >= result.started_ns
    assert not worker.busy() and worker.running_for_ms() is None


def test_results_in_order(parts):
    step, _, worker = parts
    step.gate.set()
    for i in range(1, 6):
        worker.submit_step(job(i))
        assert worker.wait(5.0).value == i


def test_exception_comes_back_as_error(parts):
    step, _, worker = parts
    step.gate.set()
    step.error = ValueError("bug")
    worker.submit_step(job(1))
    result = worker.wait(5.0)
    assert isinstance(result.error, ValueError) and result.value is None
    assert (result.process, result.request) == ("policy", "c4.infer/v1")


def test_config_job(parts):
    _, policy, worker = parts
    worker.submit_config()
    result = worker.wait(5.0)
    assert result.kind == "config" and result.value.action_steps == 16


def test_reset_then_prompt_only_when_given(parts):
    _, policy, worker = parts
    worker.submit_reset(None)
    assert worker.wait(5.0).kind == "reset"
    assert policy.calls == ["reset"]
    worker.submit_reset(("raise", 2))
    worker.wait(5.0)
    assert policy.calls == ["reset", "reset", ("prompt", "raise", 2)]


def test_refused_reset_names_the_request(parts):
    _, policy, worker = parts
    policy.reset_ack = False
    worker.submit_reset(("raise", 0))
    result = worker.wait(5.0)
    assert result.error is not None and result.request == "c4.reset/v1" and "prompt" not in str(policy.calls)


def test_dead_thread_raises(parts):
    _, _, worker = parts
    worker.stop()
    worker._thread.join(1.0)
    worker._busy = True                   # a job "submitted" to a dead thread
    with pytest.raises(RuntimeError):
        worker.wait(0.01)
