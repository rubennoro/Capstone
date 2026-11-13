"""Where the contract files are: deployment/contracts/, found from the editable install (STRUCTURE.md 1.4).

The three yaml files and examples/ sit next to the package, not inside it, so they are only found when the package
is installed editable from deployment/contracts (`uv pip install -e deployment/contracts`).
"""
from __future__ import annotations

from pathlib import Path

LIMITS_NAME = "limits.yaml"
FRAMES_NAME = "frames.yaml"
LINK_NAME = "link.yaml"
EXAMPLES_NAME = "examples"

INSTALL_HINT = ("deploy_contracts must be installed editable from deployment/contracts: "
                "uv pip install -e deployment/contracts")


def _package_file() -> Path:
    """deploy_contracts/paths.py; a function so a test can point it somewhere else."""
    return Path(__file__)


def contracts_dir() -> Path:
    """deployment/contracts/: the folder that holds deploy_contracts/, the three yaml files and examples/.

    Raises FileNotFoundError with the install hint if limits.yaml is not there (not an editable install).
    """
    root = _package_file().resolve().parents[1]
    if not (root / LIMITS_NAME).is_file():
        raise FileNotFoundError(f"{INSTALL_HINT} (no {LIMITS_NAME} in {root})")
    return root


def limits_file() -> Path:
    return contracts_dir() / LIMITS_NAME


def frames_file() -> Path:
    return contracts_dir() / FRAMES_NAME


def link_file() -> Path:
    return contracts_dir() / LINK_NAME


def examples_dir() -> Path:
    return contracts_dir() / EXAMPLES_NAME
