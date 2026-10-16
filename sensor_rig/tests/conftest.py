"""Shared pytest setup for sensor_rig (owned by WP-A; other builders add fixtures in their own modules).

The package is made importable by putting the project root (``sensor_rig/``) on ``sys.path``, so the tests run without
``pip install -e``. Run from the repo root:

    <scratchpad>/env_iphumi/bin/python -m pytest sensor_rig/tests
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ROOT = PROJECT_ROOT.parent
TESTS_DIR = Path(__file__).resolve().parent
TEST_DATA = TESTS_DIR / "data"
DEFAULT_IPHONE_DATA = REPO_ROOT / "data" / "raw" / "2026-10-01"
GRIPPER_CALIBRATION_YAML = (
    REPO_ROOT / "iPhUMI" / "python_package" / "iphumi" / "demonstration_processing" / "calibration"
    / "gripper_calibration.yaml"
)


def iphone_data_dir() -> Path:
    return Path(os.environ.get("SENSOR_RIG_IPHONE_DATA", str(DEFAULT_IPHONE_DATA)))


@pytest.fixture(scope="session")
def iphone_data() -> Path:
    """Directory with real iPhone exports (``<NAME>/`` folders). Tests using it are skipped if it is missing."""
    d = iphone_data_dir()
    if not d.is_dir() or not any(d.glob("*/*.json")):
        pytest.skip(f"real iPhone exports not found at {d} (set SENSOR_RIG_IPHONE_DATA)")
    return d


@pytest.fixture(scope="session")
def gripper_calibration_yaml() -> Path | None:
    return GRIPPER_CALIBRATION_YAML if GRIPPER_CALIBRATION_YAML.is_file() else None


@pytest.fixture(scope="session")
def test_rig_path() -> Path:
    """A small, fully valid rig yaml owned by the WP-A tests (P2 schema)."""
    return TEST_DATA / "rig_test.yaml"


@pytest.fixture(scope="session")
def test_rig(test_rig_path):
    from sensor_rig.rig_config import load_rig

    return load_rig(test_rig_path)
