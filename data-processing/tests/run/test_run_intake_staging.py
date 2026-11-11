"""The intake pieces ``dataproc run`` uses: the staging target, :func:`move_slot`, and reconcile of ``.staging/`` /
``failed/`` slots (tiny contract episodes from ``tests/fixtures``)."""

from __future__ import annotations

import json

import pytest

from dataproc.intake import LEDGER_NAME, PENDING_NAME, IntakeError, move_slot, read_ledger, reconcile, run_intake
from dataproc.paths import data_root
from fixtures import build_episode

TASK = "block_raise"


def _quiet(*a, **k):
    pass


@pytest.fixture
def root(tmp_path):
    return data_root(tmp_path / "root")


def _folder(root):
    """A task folder made the normal way (one ingested episode), so it has a task.yaml and a ledger."""
    build_episode(root.incoming / "s01_e0", episode_number=0)
    assert run_intake(root, TASK, log=_quiet).by_status("ingested")
    return root.tasks / TASK


def _rows(root):
    return read_ledger(root.tasks / TASK / LEDGER_NAME)


def test_staging_target_needs_an_existing_folder(root):
    build_episode(root.incoming / "s01_e1", episode_number=1)
    with pytest.raises(IntakeError, match="never creates one"):
        run_intake(root, TASK, staging=True, log=_quiet)
    assert not root.tasks.exists() and (root.incoming / "s01_e1").is_dir()
    with pytest.raises(IntakeError, match="existing folder"):
        run_intake(root, TASK, staging=True, new=True, log=_quiet)


def test_staging_target_puts_the_episode_in_staging(root):
    td = _folder(root)
    build_episode(root.incoming / "s01_e1", episode_number=1)
    res = run_intake(root, TASK, staging=True, log=_quiet)
    (o,) = res.outcomes
    assert o.status == "ingested" and o.slot == ".staging/s01_e1"
    assert (td / ".staging" / "s01_e1" / "raw" / "meta.json").is_file()
    assert not (td / "episode_1").exists()
    assert [(r["slot"], r["pi_name"], r["kind"]) for r in _rows(root)][-1] == (
        ".staging/s01_e1", "s01_e1", "demonstration")


def test_staging_target_duplicate_is_quarantined(root):
    td = _folder(root)
    build_episode(root.incoming / "s01_e0", episode_number=0)        # same name as episode_0
    (o,) = run_intake(root, TASK, staging=True, log=_quiet).outcomes
    assert o.status == "quarantined" and "duplicate" in o.reasons
    assert not (td / ".staging").exists() or not any((td / ".staging").iterdir())


def test_move_slot_appends_the_new_slot(root):
    td = _folder(root)
    build_episode(root.incoming / "s01_e1", episode_number=1)
    run_intake(root, TASK, staging=True, log=_quiet)
    first = _rows(root)[-1]
    move_slot(td, ".staging/s01_e1", "failed/s01_e1", "s01_e1")
    move_slot(td, "failed/s01_e1", ".staging/s01_e1", "s01_e1")
    move_slot(td, ".staging/s01_e1", "episode_1", "s01_e1")
    assert (td / "episode_1" / "raw" / "meta.json").is_file() and not (td / ".staging" / "s01_e1").exists()
    rows = _rows(root)
    assert [r["slot"] for r in rows] == ["episode_0", ".staging/s01_e1", "failed/s01_e1", ".staging/s01_e1",
                                         "episode_1"]
    for r in rows[1:]:                                   # everything but the slot and the time is carried over
        assert {k: v for k, v in r.items() if k not in ("slot", "intake_unix_ns")} == \
               {k: v for k, v in first.items() if k not in ("slot", "intake_unix_ns")}
    assert json.loads((td / PENDING_NAME).read_text())["slot"] == "episode_1"
    with pytest.raises(IntakeError, match="already exists"):
        move_slot(td, "episode_1", "episode_0", "s01_e1")


def test_reconcile_rebuilds_staging_and_failed_rows(root):
    """Rows lost in a crash (here: removed by hand) are rebuilt with the Pi name from the dir name, or from the
    pending record for an episode_N."""
    td = _folder(root)
    for name in ("s01_e1", "s01_e2", "s01_e3"):
        build_episode(root.incoming / name, episode_number=int(name[-1]), start_offset_ns=int(name[-1]) * 10 ** 9)
    run_intake(root, TASK, staging=True, log=_quiet)
    move_slot(td, ".staging/s01_e2", "failed/s01_e2", "s01_e2")
    move_slot(td, ".staging/s01_e3", "episode_1", "s01_e3")
    lines = (td / LEDGER_NAME).read_text().splitlines()
    (td / LEDGER_NAME).write_text("\n".join(lines[:2]) + "\n")        # header + episode_0 only
    rebuilt = reconcile(root, log=_quiet)
    assert sorted(rebuilt) == [(TASK, ".staging/s01_e1"), (TASK, "episode_1"), (TASK, "failed/s01_e2")]
    by_slot = {r["slot"]: r["pi_name"] for r in _rows(root)}
    assert by_slot == {"episode_0": "s01_e0", ".staging/s01_e1": "s01_e1", "failed/s01_e2": "s01_e2",
                       "episode_1": "s01_e3"}
    assert reconcile(root, log=_quiet) == []                          # nothing left to do

