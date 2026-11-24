"""The import rules of STRUCTURE.md 2.2, read with ast (nothing is imported).

- policy_deploy.contracts: stdlib, numpy, yaml, msgpack, zmq, deploy_contracts; no other policy_deploy subpackage.
- a process subpackage (frames, policy, manager, fake_pc): its own subpackage, policy_deploy.contracts,
  deploy_contracts, stdlib and its third-party libraries; never another process subpackage, the launcher or
  control_deploy.
- policy_deploy.launcher: stdlib, yaml, policy_deploy.contracts, deploy_contracts; never a process subpackage.
- torch & co only in the seven torch files of policy, never at package import.
- no module name of rnoro/deployment (contracts, deploy, process_frames, policy, message_comms) is imported.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "policy_deploy"
PROCESSES = ("frames", "policy", "manager", "fake_pc")
OLD_NAMES = ("contracts", "deploy", "process_frames", "policy", "message_comms")
TORCH_LIBS = ("torch", "torchvision", "dill", "hydra", "omegaconf", "timm", "diffusers", "behavior_prompting")
TORCH_FILES = ("diffusion_policy.py", "model_runner.py", "checkpoint_loader.py", "light_workspace.py", "bench.py",
               "prompt_source.py", "promptable_diffusion_policy.py")
STDLIB = set(sys.stdlib_module_names)
THIRD_PARTY = {
    "contracts": {"numpy", "yaml", "msgpack", "zmq"},
    "launcher": {"yaml"},
    "frames": {"numpy", "yaml", "zmq", "cv2", "scipy", "av", "msgpack"},
    "policy": {"numpy", "yaml", "zmq"} | set(TORCH_LIBS),
    "manager": {"numpy", "yaml", "zmq", "scipy"},
    "fake_pc": {"numpy", "yaml", "zmq", "scipy"},
}


def modules():
    return sorted(p for p in PACKAGE.rglob("*.py"))


def subpackage(path: Path) -> str | None:
    rel = path.relative_to(PACKAGE).parts
    return rel[0] if len(rel) > 1 else None


def imports(path: Path) -> list[tuple[str, int, bool]]:
    """(module name, line, at top level) for every import in the file (relative imports resolved)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg_parts = ["policy_deploy", *path.relative_to(PACKAGE).parts[:-1]]
    top_level = set(id(n) for n in tree.body)
    for node in tree.body:                         # `if TYPE_CHECKING:` blocks and try/except at top count as top
        if isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                top_level.add(id(sub))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno, id(node) in top_level))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[:len(pkg_parts) - node.level + 1]
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            out.append((name, node.lineno, id(node) in top_level))
    return out


def test_there_are_modules():
    assert len(modules()) > 10


@pytest.mark.parametrize("path", modules(), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_import_rules(path):
    sub = subpackage(path)
    problems = []
    for name, line, _top in imports(path):
        root = name.split(".")[0]
        where = f"{path.relative_to(PACKAGE)}:{line}: {name}"
        if root in OLD_NAMES:
            problems.append(f"{where}: a module name of rnoro/deployment")
            continue
        if root == "control_deploy":
            problems.append(f"{where}: never control_deploy")
            continue
        if root == "policy_deploy":
            parts = name.split(".")
            target = parts[1] if len(parts) > 1 else None
            if sub is None:
                if target not in (None, "launcher"):       # __main__ runs the launcher's cli
                    problems.append(f"{where}: the package root imports only the launcher")
            elif sub == "contracts":
                if target not in (None, "contracts"):
                    problems.append(f"{where}: policy_deploy.contracts imports no other subpackage")
            elif sub == "launcher":
                if target not in ("contracts", "launcher"):
                    problems.append(f"{where}: the launcher never imports a process subpackage")
            elif target not in ("contracts", sub):
                problems.append(f"{where}: {sub} may import only itself and policy_deploy.contracts")
            continue
        if root == "deploy_contracts" or root in STDLIB or root == "__future__":
            continue
        allowed = THIRD_PARTY.get(sub or "launcher", set())
        if root not in allowed:
            problems.append(f"{where}: {root} is not allowed in {sub or 'the package root'}")
    assert not problems, "\n".join(problems)


def test_torch_only_in_the_torch_files():
    problems = []
    for path in modules():
        if subpackage(path) != "policy" or path.name in TORCH_FILES:
            continue
        bad = [f"{name}:{line}" for name, line, _ in imports(path) if name.split(".")[0] in TORCH_LIBS]
        if bad:
            problems.append(f"{path.name} imports {bad}; torch & co belong only in {TORCH_FILES}")
    assert not problems, "\n".join(problems)


def test_policy_package_init_imports_no_torch_file():
    init = PACKAGE / "policy" / "__init__.py"
    if not init.exists():
        pytest.skip("policy not built yet")
    names = [n for n, _, _ in imports(init)]
    assert not any(n.endswith(tuple(f[:-3] for f in TORCH_FILES)) for n in names)
