"""deploy_contracts imports only the standard library, numpy, msgpack, zmq and yaml (STRUCTURE.md 1.3 rule 3)."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import deploy_contracts

ALLOWED_THIRD_PARTY = {"numpy", "msgpack", "zmq", "yaml"}
FORBIDDEN = {"policy_deploy", "control_deploy", "contracts"}   # the two folders, and the old package name
PACKAGE = Path(deploy_contracts.__file__).resolve().parent


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_every_module_imports_only_allowed_packages():
    allowed = set(sys.stdlib_module_names) | ALLOWED_THIRD_PARTY | {"deploy_contracts", "__future__"}
    files = sorted(PACKAGE.glob("*.py"))
    assert len(files) >= 20
    for path in files:
        names = _imports(path)
        assert not names & FORBIDDEN, (path.name, names & FORBIDDEN)
        assert names <= allowed, (path.name, names - allowed)


def test_no_relative_imports():
    # absolute deploy_contracts.* imports only, so the rule above sees every import
    for path in PACKAGE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, path.name


def test_importing_loads_neither_folder_nor_the_old_package():
    modules = ", ".join(f"'{p.stem}'" for p in PACKAGE.glob("*.py") if p.stem != "__init__")
    code = ("import importlib, sys; import deploy_contracts; "
            f"[importlib.import_module('deploy_contracts.' + m) for m in ({modules},)]; "
            "print(' '.join(sorted({m.split('.')[0] for m in sys.modules})))")
    loaded = set(subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                                text=True).stdout.split())
    assert not loaded & FORBIDDEN, loaded & FORBIDDEN
