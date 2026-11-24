"""control_deploy imports only the standard library, yaml, numpy, zmq, msgpack and deploy_contracts; never
policy_deploy or the old packages (control_deploy/STRUCTURE.md 6.1, deployment/STRUCTURE.md 1.3)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import control_deploy

ALLOWED = {"yaml", "numpy", "zmq", "msgpack", "deploy_contracts", "control_deploy"}
FORBIDDEN = {"policy_deploy", "contracts", "deploy", "process_frames", "policy", "message_comms"}
PACKAGE = Path(control_deploy.__file__).resolve().parent


def imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_only_allowed_imports():
    files = sorted(PACKAGE.rglob("*.py"))
    assert len(files) >= 15
    for f in files:
        for name in imports(f):
            assert name not in FORBIDDEN, f"{f}: imports {name}"
            assert name in ALLOWED or name in sys.stdlib_module_names, f"{f}: imports {name}"


def test_never_reads_policy_deploy_files():
    for f in PACKAGE.rglob("*.py"):
        assert "policy_deploy/" not in f.read_text().replace("policy_deploy on the laptop", ""), f
