from __future__ import annotations

import io

import pytest

from deploy import CheckResult, print_table
from deploy.check_result import exit_code


def table(results) -> list[str]:
    out = io.StringIO()
    print_table(results, out)
    return out.getvalue().splitlines()


def test_table_layout():
    lines = table([CheckResult("geometry", "tcp triangle", "pass", "closes"),
                   CheckResult("policy", "kind", "fail", "policy.kind is not set:\n choose one")])
    assert lines[0].split() == ["process", "check", "status", "detail"]
    assert set(lines[1].replace(" ", "")) == {"-"}
    assert lines[2] == "geometry  tcp triangle  PASS    closes"
    assert lines[3] == "policy    kind          FAIL    policy.kind is not set: choose one"


def test_columns_align():
    lines = table([CheckResult("a", "b", "info", "x"), CheckResult("franka_control", "rt kernel", "warn", "y")])
    starts = {line.rindex(" ") + 1 for line in lines[2:]}
    assert len(starts) == 1


@pytest.mark.parametrize("status", ["pass", "warn", "fail", "info"])
def test_statuses_accepted(status):
    assert CheckResult("p", "n", status, "d").status == status


def test_unknown_status_refused():
    with pytest.raises(ValueError):
        CheckResult("p", "n", "ok", "d")


def test_exit_code():
    assert exit_code([CheckResult("p", "a", "pass", ""), CheckResult("p", "b", "fail", "")]) == 2
    assert exit_code([CheckResult("p", "a", "warn", ""), CheckResult("p", "b", "info", "")]) == 0
    assert exit_code([]) == 0
