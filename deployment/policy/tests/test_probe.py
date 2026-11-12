from __future__ import annotations

import io

from contracts import ContractLimits, RpcClient

from fakes import RaisingPolicy, ServeThread
from policy import probe
from policy.hold_still_policy import HoldStillPolicy
from policy.policy_server import PolicyServer


def test_all_steps_pass_against_hold_still(station_file, capsys):
    server = PolicyServer(HoldStillPolicy(ContractLimits.load().gripper_width_max_m), "tcp://127.0.0.1:*", 0.5)
    with ServeThread(server) as thread:
        code = probe.main(["--station", str(station_file(None, endpoint=server.endpoint)), "--infers", "3"])
        out = capsys.readouterr().out
        assert code == 0, out
        lines = out.strip().splitlines()
        assert lines[0].startswith("PASS config: inputs ['gripper_width_m']")
        assert lines[1].startswith("PASS reset ") and lines[1].endswith("(max of 5)")
        assert lines[2].startswith("PASS infer ") and "width_clipped total 0" in lines[2]
        assert lines[3].startswith("PASS final reset")
        assert thread.error is None


def test_error_reply_is_printed():
    server = PolicyServer(RaisingPolicy(), "tcp://127.0.0.1:*", 0.5)
    with ServeThread(server):
        client = RpcClient("policy", server.endpoint)
        out = io.StringIO()
        try:
            code = probe.Probe(client, 0.05, 5.0, out).run(resets=2, infers=2, task_name=None, prompt_index=None)
        finally:
            client.close()
    assert code == 1
    last = out.getvalue().strip().splitlines()[-1]
    assert last.startswith("FAIL infer: ErrorReply: ") and "model exploded" in last
