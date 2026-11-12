"""Every launch.py imports only the standard library, yaml, deploy and contracts (read with ast, never imported)."""
from __future__ import annotations

import ast
import sys

import pytest

from helpers import REAL_ROOT

FOLDERS = ["franka_control", "process_frames", "policy", "message_comms"]
ALLOWED = {"yaml", "deploy", "contracts"}


def imported_modules(source: str) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
    return names


def forbidden(source: str) -> list[str]:
    return [name for name in imported_modules(source)
            if name.startswith(".") or (name.split(".")[0] not in sys.stdlib_module_names
                                        and name.split(".")[0] not in ALLOWED)]


@pytest.mark.parametrize("folder", FOLDERS)
def test_launch_py_imports(folder):
    path = REAL_ROOT / folder / "launch.py"
    if not path.exists():
        pytest.skip(f"{folder}/launch.py does not exist yet")
    assert forbidden(path.read_text()) == [], f"{path} imports something other than stdlib, yaml, deploy, contracts"


def test_the_rule_itself():
    good = "from __future__ import annotations\nimport os, sys\nimport yaml\nfrom deploy import ManagedProcess\n" \
           "from contracts.codec import x\nfrom deploy.check_helpers import port_free\n"
    assert forbidden(good) == []
    assert forbidden("import policy\n") == ["policy"]
    assert forbidden("from process_frames.rig_file import RigFile\n") == ["process_frames.rig_file"]
    assert forbidden("from . import x\n") == ["."]
    assert forbidden("import numpy\n") == ["numpy"]
