# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/launcher/test_session.py @ 2ffe641
# changes: imports only. The source's own header follows.
# adapted from rnoro/deployment:deployment/deploy/tests/test_session.py @ 91b178b
# changes: imports; + the three contract files copied into contracts/.
from __future__ import annotations

import re

import pytest
from deploy_contracts import paths as contract_paths

from teleop_collect.launcher.errors import LaunchError
from teleop_collect.launcher.session import Session


@pytest.mark.parametrize("mock,suffix", [(False, ""), (True, "_mock")])
def test_folder_name(tmp_path, make_station, mock, suffix):
    session = Session.create(tmp_path / "root", mock, make_station())
    assert session.path.parent == tmp_path / "root"
    assert re.fullmatch(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d" + suffix, session.path.name)


def test_folders_made_and_station_copied(tmp_path, make_station):
    station_file = make_station()
    session = Session.create(tmp_path / "root", False, station_file)
    assert (session.path / "logs").is_dir() and (session.path / "ready").is_dir()
    assert (session.path / "station.yaml").read_text() == station_file.read_text()
    assert session.log_path("policy") == session.path / "logs" / "policy.log"
    assert session.ready_file("policy") == session.path / "ready" / "policy.ready"
    assert session.process_dir("policy") == session.path / "policy"


def test_contract_files_copied_into_contracts(tmp_path, make_station):
    session = Session.create(tmp_path / "root", False, make_station())
    copied = sorted(p.name for p in (session.path / "contracts").iterdir())
    assert copied == ["frames.yaml", "limits.yaml", "link.yaml"]
    assert (session.path / "contracts" / "link.yaml").read_bytes() == contract_paths.link_file().read_bytes()
    assert not (session.path / "frames.yaml").exists()      # never next to the frames/ process folder


def test_existing_folder_refused(tmp_path, make_station, monkeypatch):
    import teleop_collect.launcher.session as module

    class FixedNow:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime(2026, 11, 12, 10, 0, 0)

    monkeypatch.setattr(module, "datetime", FixedNow)
    Session.create(tmp_path, False, make_station())
    with pytest.raises(LaunchError, match="already exists"):
        Session.create(tmp_path, False, make_station())
