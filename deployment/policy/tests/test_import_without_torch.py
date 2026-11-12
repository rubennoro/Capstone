from __future__ import annotations

import subprocess
import sys

from fakes import BLOCKER_PRELUDE

SCRIPT = BLOCKER_PRELUDE + r"""
import threading
from pathlib import Path
station = Path(sys.argv[1])

import policy
from policy.settings import Settings
from policy.policy_factory import build_policy
from policy.policy_server import PolicyServer
from policy.neutral_observation import neutral_observation
import policy.self_check, policy.probe, policy.app, policy.key_map, policy.errors, policy.promptable_policy
import policy.__main__

settings = Settings.load(station, station.parent / "sess", mock=True)
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


def test_light_paths_never_import_torch(station_file):
    path = station_file(None)
    proc = subprocess.run([sys.executable, "-c", SCRIPT, str(path)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK")
