# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/test_import_rules.py @ 2ffe641
# changes: teleop_collect's subpackages (contracts, common, frames, teleop, ui, fake_pc, launcher; PLAN.md rule 5,
# STRUCTURE.md 12 row E1): never policy_deploy or control_deploy; the process subpackages import only themselves,
# teleop_collect.contracts and teleop_collect.common; the torch rules dropped (no policy); third-party lists per
# subpackage from STRUCTURE.md 3.1.
"""The import rules, read with ast (nothing is imported). Every .py file under teleop_collect/ that exists is checked,
so the rules keep holding as files are added.

- nowhere: policy_deploy, control_deploy, or a module name of rnoro/deployment (contracts, deploy, process_frames,
  policy, message_comms).
- teleop_collect.contracts: stdlib, numpy, yaml, msgpack, zmq, deploy_contracts; no other teleop_collect subpackage.
- teleop_collect.common: stdlib, numpy, scipy; no other teleop_collect subpackage.
- a process subpackage (frames, teleop, ui, fake_pc): itself (teleop includes teleop.motion and teleop.episode),
  teleop_collect.contracts, teleop_collect.common, deploy_contracts, stdlib and its third-party libraries; never
  another process subpackage or the launcher.
- teleop_collect.launcher: itself, teleop_collect.contracts, teleop_collect.common, deploy_contracts (the contract
  files and their check rows), stdlib and yaml; never a process subpackage (it runs them as subprocesses).
- the package root (__init__, __main__): only teleop_collect.launcher (the cli).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "teleop_collect"
PROCESSES = ("frames", "teleop", "ui", "fake_pc")
SHARED = ("contracts", "common")
FORBIDDEN = ("policy_deploy", "control_deploy")
OLD_NAMES = ("contracts", "deploy", "process_frames", "policy", "message_comms")
STDLIB = set(sys.stdlib_module_names)
_BASE = {"numpy", "yaml", "msgpack", "zmq"}
THIRD_PARTY = {
    "contracts": _BASE,
    "common": {"numpy", "scipy"},
    "frames": _BASE | {"cv2", "scipy", "av"},
    "teleop": _BASE | {"scipy"},
    "ui": _BASE | {"aiohttp"},
    "fake_pc": _BASE | {"scipy"},
    "launcher": {"yaml"},
    None: set(),
}


def modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def subpackage(path: Path) -> str | None:
    rel = path.relative_to(PACKAGE).parts
    return rel[0] if len(rel) > 1 else None


def imports(path: Path) -> list[tuple[str, int]]:
    """(module name, line) for every import anywhere in the file (relative imports resolved)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg_parts = ["teleop_collect", *path.relative_to(PACKAGE).parts[:-1]]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[:len(pkg_parts) - node.level + 1]
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            out.append((name, node.lineno))
    return out


def problems_of(path: Path) -> list[str]:
    sub = subpackage(path)
    problems = []
    for name, line in imports(path):
        root = name.split(".")[0]
        where = f"{path.relative_to(PACKAGE)}:{line}: {name}"
        if root in FORBIDDEN:
            problems.append(f"{where}: never {root} (PLAN.md rule 5)")
            continue
        if root in OLD_NAMES:
            problems.append(f"{where}: a module name of rnoro/deployment")
            continue
        if root == "teleop_collect":
            parts = name.split(".")
            target = parts[1] if len(parts) > 1 else None
            if target is None:
                continue                                  # `import teleop_collect`: the root imports nothing
            if sub is None:
                if target != "launcher":
                    problems.append(f"{where}: the package root imports only the launcher")
            elif sub in SHARED:
                if target != sub:
                    problems.append(f"{where}: teleop_collect.{sub} imports no other subpackage")
            elif sub == "launcher":
                if target not in ("launcher", *SHARED):
                    problems.append(f"{where}: the launcher never imports a process subpackage (it runs them)")
            elif target not in (sub, *SHARED):
                problems.append(f"{where}: {sub} may import only itself, teleop_collect.contracts and "
                                "teleop_collect.common")
            continue
        if root == "deploy_contracts":
            if sub == "common":
                problems.append(f"{where}: teleop_collect.common is pure (no deploy_contracts)")
            continue
        if root in STDLIB or root == "__future__":
            continue
        if root not in THIRD_PARTY.get(sub, set()):
            problems.append(f"{where}: {root} is not allowed in {sub or 'the package root'}")
    return problems


def test_there_are_modules():
    assert len(modules()) > 10


@pytest.mark.parametrize("path", modules(), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_import_rules(path):
    problems = problems_of(path)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("source,sub,bad", [
    ("import policy_deploy.contracts\n", "frames", True),
    ("from control_deploy import x\n", "teleop", True),
    ("from teleop_collect.teleop import app\n", "frames", True),
    ("from teleop_collect.frames import app\n", "launcher", True),
    ("from teleop_collect.ui import server\n", "teleop", True),
    ("from teleop_collect.common import pose_math\n", "contracts", True),
    ("from teleop_collect.teleop.motion import planner\n", "teleop", False),
    ("from ..episode import store\n", "teleop/motion", False),
    ("from teleop_collect.contracts import Station\nfrom teleop_collect.common import pose_math\n", "ui", False),
    ("import av\n", "frames", False),
    ("import av\n", "teleop", True),
    ("def f():\n    import policy_deploy\n", "fake_pc", True),
])
def test_the_rules_catch_what_they_should(tmp_path, monkeypatch, source, sub, bad):
    """The checker itself, on small files in a fake package tree."""
    root = tmp_path / "teleop_collect"
    path = root / sub / "mod.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    monkeypatch.setitem(globals(), "PACKAGE", root)
    assert bool(problems_of(path)) is bad, problems_of(path)
