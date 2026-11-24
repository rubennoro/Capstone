# adapted from rnoro/deployment:deployment/policy/tests/test_import_without_torch.py @ 91b178b
# changes: module paths (policy_deploy.policy, incl. bench's light import); the new Settings.load(station, mock)
from __future__ import annotations

import subprocess
import sys

from policy_fakes import BLOCKER_PRELUDE

SCRIPT = BLOCKER_PRELUDE + r"""
from pathlib import Path
station = Path(sys.argv[1])

import policy_deploy.policy
from policy_deploy.policy.settings import Settings
from policy_deploy.policy.policy_factory import build_policy
from policy_deploy.policy.policy_server import PolicyServer
from policy_deploy.policy.neutral_observation import neutral_observation
import policy_deploy.policy.self_check, policy_deploy.policy.probe, policy_deploy.policy.app
import policy_deploy.policy.key_map, policy_deploy.policy.errors, policy_deploy.policy.promptable_policy
import policy_deploy.policy.bench
import policy_deploy.policy.__main__

settings = Settings.load(station, mock=True)
p = build_policy(settings)
assert type(p).__name__ == "HoldStillPolicy"
p.reset()
p.infer(neutral_observation(0.05))
server = PolicyServer(p, "tcp://127.0.0.1:*", 0.5)
server.close()

loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
assert not loaded, loaded
print("OK")
"""


def test_light_paths_never_import_torch(policy_station):
    path = policy_station(None)
    proc = subprocess.run([sys.executable, "-c", SCRIPT, str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK")
