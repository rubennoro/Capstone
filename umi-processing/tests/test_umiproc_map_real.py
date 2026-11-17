"""``umiproc map`` with the real umiproc_slam on a real Pi recording (slow: two real-time SLAM runs, about a minute or
two on the laptop). The default recording is a DEMO, not a mapping clip, so the map is expected to FAIL with clear
messages: nothing is created or recorded, and the report and logs are kept in maps_failed/. Skipped when the binary,
the vocabulary or the recording is absent.

Inputs (read only): ``$UMIPROC_TEST_MAP_SOURCE`` (default ~/Downloads/data/incoming) and ``$UMIPROC_TEST_MAP_NAME``
(default episode_000120).
"""

import json
import os
from pathlib import Path

import pytest

from umiproc.dp.paths import DataRoot
from umiproc.map import build as B
from umiproc.map.cli import format_outcome
from umiproc.slam.runner import default_binary, default_vocabulary
from umiproc_testlib import tree_digest

SOURCE = Path(os.environ.get("UMIPROC_TEST_MAP_SOURCE", Path.home() / "Downloads/data/incoming"))
NAME = os.environ.get("UMIPROC_TEST_MAP_NAME", "episode_000120")

_missing = [str(p) for p in (default_binary(), default_vocabulary(), SOURCE / NAME / "COMPLETE") if not p.is_file()]
pytestmark = [pytest.mark.slow, pytest.mark.skipif(bool(_missing), reason=f"needs {', '.join(_missing)}")]


def test_map_on_a_demo_fails_clearly(tmp_path):
    root = DataRoot(tmp_path / "root")
    before = tree_digest(SOURCE / NAME)
    lines = []
    o = B.build_map(root, "smoke", NAME, source=SOURCE, log=lines.append)
    print("\n" + "\n".join(lines) + "\n" + format_outcome(o, "smoke", NAME))
    assert tree_digest(SOURCE / NAME) == before                         # the source is only read
    assert (o.passed, o.exit_code) == (False, B.EXIT_FAILED)
    assert not root.tasks.exists() or not any(root.tasks.iterdir())    # nothing attached, nothing recorded
    assert not (root.path / "pi_sources.csv").exists()
    assert not any(root.incoming.iterdir()) and not (root.path / B.WORK_DIR).exists()
    rep = json.loads((o.where / B.REPORT_NAME).read_text())
    assert [c["id"] for c in rep["checks"]] == ["M1", "M2", "M3", "M4", "M5"]
    for c in o.checks:
        assert c.message
        if c.passed is False:
            assert c.hints, c.id
    assert (o.where / B.MAPPING_DIR / "umiproc_slam.log").is_file()
    assert rep["diagnostics"]["build"]["patches"] and rep["diagnostics"]["vio_inflation"]
    assert rep["runs"]["mapping"]["pacing"] is not None                  # reported (laptop: often below real time)
