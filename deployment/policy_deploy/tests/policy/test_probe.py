# adapted from rnoro/deployment:deployment/policy/tests/test_probe.py @ 91b178b
# changes: the probe's endpoint comes from station ports.policy; the config line names the kind; infer reports mean/p50/p95/max
from __future__ import annotations

import io

from deploy_contracts import ContractLimits, RpcClient

from policy_fakes import RaisingPolicy, ServeThread, endpoint_of
from policy_deploy.policy import probe
from policy_deploy.policy.hold_still_policy import HoldStillPolicy
from policy_deploy.policy.policy_server import PolicyServer


def test_all_steps_pass_against_hold_still(policy_station, capsys):
    station = policy_station({"kind": "hold_still"})
    server = PolicyServer(HoldStillPolicy(ContractLimits.load().gripper_width_max_m), endpoint_of(station), 0.5)
    with ServeThread(server) as thread:
        code = probe.main(["--station", str(station), "--infers", "3"])
        out = capsys.readouterr().out
        assert code == 0, out
        lines = out.strip().splitlines()
        assert lines[0].startswith("PASS config: kind hold_still, inputs ['gripper_width_m']")
        assert lines[1].startswith("PASS reset ") and lines[1].endswith("(max of 5)")
        assert lines[2].startswith("PASS infer mean ") and "width_clipped total 0" in lines[2]
        assert " p50 " in lines[2] and " p95 " in lines[2] and " max " in lines[2] and "(3 chunks" in lines[2]
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


def test_timing_summary_percentiles():
    text = probe.timing_summary([float(v) for v in range(1, 101)])
    assert text == "mean 50.5 ms, p50 50.5 ms, p95 95.0 ms, max 100.0 ms"


def test_probe_cli_settings_failure(tmp_path, capsys):
    assert probe.main(["--station", str(tmp_path / "missing.yaml")]) == 1
    assert capsys.readouterr().out.startswith("FAIL settings: missing.yaml: not found")
