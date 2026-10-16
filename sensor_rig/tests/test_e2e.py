"""D10c: the full iPhUMI pipeline on real iPhone episodes vs simulated-rig exports (slow, ~10-20 min).

Skipped unless ``SENSOR_RIG_E2E=1``. Runs ``tests/e2e/run_pipeline_compare.py`` in a fresh work dir under
pytest's tmp dir (kept for inspection; see ``report.json`` there).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.data,
              pytest.mark.skipif(os.environ.get("SENSOR_RIG_E2E") != "1", reason="set SENSOR_RIG_E2E=1 to run")]

SCRIPT = Path(__file__).resolve().parent / "e2e" / "run_pipeline_compare.py"


def test_d10c_pipeline_compare(iphone_data, tmp_path):
    W = tmp_path / "d10c"
    cmd = [sys.executable, str(SCRIPT), "--workdir", str(W), "--src", str(iphone_data),
           "--workers", os.environ.get("SENSOR_RIG_E2E_WORKERS", "4")]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3 * 3600)
    print(p.stdout[-20000:])
    report = W / "report.json"
    assert report.is_file(), p.stderr[-5000:]
    rep = json.loads(report.read_text())
    failed = [r for r in rep["rows"] if not r["ok"]]
    assert p.returncode == 0 and rep["ok"], f"failed checks: {failed}"
