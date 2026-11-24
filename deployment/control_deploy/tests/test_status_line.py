"""StatusLine from the C2 example states (control_deploy/STRUCTURE.md 6.4 step 6)."""
from __future__ import annotations

import pytest
from deploy_contracts import MODES, ArmState, decode, load_index

from control_deploy.status_line import StatusLine

STATES = [decode(e.data) for e in load_index() if e.valid and e.schema == "c2.state/v1"]


def line_for(s: ArmState, peers=None, now=0.0) -> str:
    sl = StatusLine()
    sl.feed(s, now)
    return sl.render(now, peers)


def test_every_mode_of_the_examples_is_covered():
    assert {s.mode for s in STATES} == set(MODES)


@pytest.mark.parametrize("s", STATES, ids=lambda s: s.mode)
def test_each_example_state(s):
    line = line_for(s, ["192.168.50.20"])
    assert len(line) <= 120
    assert line.startswith(s.mode) or line.startswith("fault: ")
    if s.mode == "fault":
        assert s.error.split()[0] in line
    else:
        g = s.gate
        assert f"gate a{g.accepted} s{g.shortened} p{g.pulled_back} r{g.rejected}" in line
        assert "C3 peer 192.168.50.20" in line and line.endswith("h p g s q")
        assert f"grip {s.gripper_width_m:.3f} m" in line


def test_the_documented_layout():
    s = [x for x in STATES if x.mode == "running"][0]
    line = line_for(s)
    assert "|tcp " in line.replace(" | ", "|") and "|late 3 (+0" in line.replace(" | ", "|")
    assert "C3 peer none" in line
    wide = StatusLine(width=200)
    wide.feed(s, 0.0)
    assert " | late 3 (+0 in 10 s) | C2 " in wide.render(0.0)   # the full layout when it fits


def test_late_cycles_in_the_last_ten_seconds_and_the_rate():
    s = [x for x in STATES if x.mode == "holding"][0]
    sl = StatusLine()
    for i in range(100):   # 100 Hz for 1 s, loop_late +2 over it
        sl.feed(s.__class__(**{**{f: getattr(s, f) for f in s.__dataclass_fields__}, "loop_late": 3 + (i // 50)}),
                i * 0.01)
    line = sl.render(0.99)
    assert "late 4 (+1" in line
    assert "C2 100 Hz" in line or "C2 99 Hz" in line


def test_no_c2_for_a_second():
    s = STATES[0]
    sl = StatusLine()
    assert sl.render(5.0) == "no C2: see the log"
    sl.feed(s, 1.0)
    assert sl.render(1.5) != "no C2: see the log"
    assert sl.render(2.1) == "no C2: see the log"
