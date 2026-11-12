from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import zmq
from contracts import (Ack, ActionChunk, ContractLimits, ErrorReply, GetFramesRequest, PolicyConfig,
                       PolicyConfigRequest, Prompt, RemoteError, Reset, RpcClient, decode)
from contracts.codec import pack_obj

from fakes import (BadChunkPolicy, FakePromptablePolicy, OddChunkPolicy, RaisingPolicy, ServeThread)
from policy.errors import BadRequest, PolicyFatal
from policy.hold_still_policy import HoldStillPolicy
from policy.neutral_observation import neutral_observation
from policy.policy_server import PolicyServer

ENDPOINT = "tcp://127.0.0.1:*"
TIMEOUT_S = 5.0


def _hold_still() -> HoldStillPolicy:
    return HoldStillPolicy(ContractLimits.load().gripper_width_max_m)


@pytest.fixture
def serve():
    """serve(policy) -> (ServeThread, RpcClient), both cleaned up after the test."""
    started: list[tuple[ServeThread, RpcClient]] = []

    def start(policy):
        thread = ServeThread(PolicyServer(policy, ENDPOINT, reset_max_s=0.5)).__enter__()
        client = RpcClient("policy", thread.server.endpoint)
        started.append((thread, client))
        return thread, client

    yield start
    for thread, client in started:
        client.close()
        thread.__exit__(None, None, None)


def _raw_request(endpoint: str, data: bytes) -> ErrorReply:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(endpoint)
    try:
        sock.send(data)
        assert sock.poll(int(TIMEOUT_S * 1000)), "no reply"
        reply = decode(sock.recv())
    finally:
        sock.close()
    assert isinstance(reply, ErrorReply)
    return reply


def _remote_error(client: RpcClient, msg, expect) -> str:
    with pytest.raises(RemoteError) as e:
        client.request(msg, expect, timeout_s=TIMEOUT_S)
    return e.value.reply.error


def test_config_reply(serve):
    _, client = serve(_hold_still())
    config = client.request(PolicyConfigRequest(), PolicyConfig, timeout_s=TIMEOUT_S)
    assert list(config.inputs) == ["gripper_width_m"] and config.uses_prompt is False


def test_reset_answers_fast_and_infer_works(serve):
    _, client = serve(_hold_still())
    t0 = time.monotonic()
    ack = client.request(Reset(), Ack, timeout_s=TIMEOUT_S)
    assert time.monotonic() - t0 < 0.5 and ack.ok
    chunk = client.request(neutral_observation(0.05), ActionChunk, timeout_s=TIMEOUT_S)
    assert chunk.eef_pos.shape == (16, 3)


def test_infer_before_first_reset_refused_and_server_keeps_serving(serve):
    _, client = serve(_hold_still())
    assert "infer before reset" in _remote_error(client, neutral_observation(0.05), ActionChunk)
    assert client.request(Reset(), Ack, timeout_s=TIMEOUT_S).ok
    client.request(neutral_observation(0.05), ActionChunk, timeout_s=TIMEOUT_S)


def test_invalid_requests_give_a_clear_error_and_serving_continues(serve):
    thread, client = serve(_hold_still())
    endpoint = thread.server.endpoint
    bad = neutral_observation(0.05).to_dict()
    bad["main"] = np.zeros((2, 224, 3), np.uint8)
    reply = _raw_request(endpoint, pack_obj(bad))
    assert reply.field == "main" and "main" in reply.error and reply.request == "c4.infer/v1"

    old = Reset().to_dict()
    old["schema"] = "c4.reset/v2"
    reply = _raw_request(endpoint, pack_obj(old))
    assert "v2" in reply.error

    assert "c1.get_frames/v1" in _remote_error(client, GetFramesRequest(), PolicyConfig)

    assert client.request(Reset(), Ack, timeout_s=TIMEOUT_S).ok         # still serving
    client.request(neutral_observation(0.05), ActionChunk, timeout_s=TIMEOUT_S)
    assert thread.error is None


def test_prompt_refused_for_a_non_promptable_policy(serve):
    _, client = serve(_hold_still())
    client.request(Reset(), Ack, timeout_s=TIMEOUT_S)
    error = _remote_error(client, Prompt(task_name="block_raise", prompt_index=0), Ack)
    assert "prompt: this policy (kind hold_still) does not use prompts (config uses_prompt: false)" in error
    client.request(neutral_observation(0.05), ActionChunk, timeout_s=TIMEOUT_S)


def test_promptable_order_exactly_one_prompt_after_each_reset(serve):
    policy = FakePromptablePolicy()
    thread, client = serve(policy)
    assert client.request(PolicyConfigRequest(), PolicyConfig, timeout_s=TIMEOUT_S).uses_prompt is True
    assert "prompt before reset" in _remote_error(client, Prompt(task_name="t", prompt_index=0), Ack)
    for cycle in range(3):
        client.request(Reset(), Ack, timeout_s=TIMEOUT_S)
        assert "infer before prompt" in _remote_error(client, neutral_observation(0.05), ActionChunk)
        assert client.request(Prompt(task_name="t", prompt_index=cycle), Ack, timeout_s=TIMEOUT_S).ok
        assert "prompt already given since the last reset" in _remote_error(
            client, Prompt(task_name="t", prompt_index=9), Ack)
        client.request(neutral_observation(0.05), ActionChunk, timeout_s=TIMEOUT_S)
    expected = []
    for cycle in range(3):
        expected += [("reset",), ("prompt", "t", cycle), ("infer",)]
    assert policy.calls == expected
    assert thread.error is None


@pytest.mark.parametrize("policy_cls", [RaisingPolicy, BadChunkPolicy])
def test_own_failure_replies_error_then_serve_raises_policy_fatal(serve, policy_cls):
    thread, client = serve(policy_cls())
    client.request(Reset(), Ack, timeout_s=TIMEOUT_S)
    error = _remote_error(client, neutral_observation(0.05), ActionChunk)
    assert error
    assert thread.join(TIMEOUT_S)
    assert isinstance(thread.error, PolicyFatal)
    if policy_cls is BadChunkPolicy:
        assert thread.error.what.startswith("own reply broke C4: ")
    else:
        assert "model exploded" in thread.error.what


def test_cuda_error_message():
    policy = RaisingPolicy(lambda: RuntimeError("CUDA error: out of memory"))
    server = PolicyServer(policy, ENDPOINT, reset_max_s=0.5)
    try:
        server._on_reset(Reset())
        with pytest.raises(PolicyFatal) as e:
            server._on_infer(neutral_observation(0.05))
        assert e.value.what.startswith("CUDA error during infer") and "policy.device: cpu" in e.value.check
    finally:
        server.close()


def test_odd_chunk_is_a_normal_reply(serve):
    thread, client = serve(OddChunkPolicy())
    client.request(Reset(), Ack, timeout_s=TIMEOUT_S)
    chunk = client.request(neutral_observation(0.05), ActionChunk, timeout_s=TIMEOUT_S)
    assert chunk.eef_pos[0, 0] == np.float32(0.05)
    client.request(neutral_observation(0.05), ActionChunk, timeout_s=TIMEOUT_S)
    assert thread.error is None


def test_bad_request_from_the_policy_keeps_serving(serve):
    policy = RaisingPolicy(lambda: BadRequest("unknown task_name 'x'"))
    thread, client = serve(policy)
    client.request(Reset(), Ack, timeout_s=TIMEOUT_S)
    assert "unknown task_name 'x'" in _remote_error(client, neutral_observation(0.05), ActionChunk)
    assert client.request(Reset(), Ack, timeout_s=TIMEOUT_S).ok
    assert thread.error is None and policy.infers == 1


def test_slow_reset_logs_a_warning(caplog):
    from fakes import FakeClock
    server = PolicyServer(_hold_still(), ENDPOINT, reset_max_s=0.5, clock=FakeClock(step_ns=600_000_000))
    try:
        with caplog.at_level("WARNING", logger="policy.server"):
            assert server._on_reset(Reset()).ok
        assert any("reset took 0.600 s" in r.getMessage() for r in caplog.records)
    finally:
        server.close()


def test_stop_event_returns_within_poll_s():
    server = PolicyServer(_hold_still(), ENDPOINT, reset_max_s=0.5)
    stop = threading.Event()
    thread = threading.Thread(target=server.serve, args=(stop, 0.05), daemon=True)
    try:
        thread.start()
        t0 = time.monotonic()
        stop.set()
        thread.join(1.0)
        assert not thread.is_alive() and time.monotonic() - t0 < 0.05 + 0.5     # poll_s plus scheduling slack
    finally:
        server.close()
