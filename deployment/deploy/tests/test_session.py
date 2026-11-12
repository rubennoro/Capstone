from __future__ import annotations

import re

import pytest

from deploy import LaunchError, Session


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


def test_existing_folder_refused(tmp_path, make_station, monkeypatch):
    import deploy.session as module

    class FixedNow:
        @staticmethod
        def now():
            from datetime import datetime
            return datetime(2026, 11, 12, 10, 0, 0)

    monkeypatch.setattr(module, "datetime", FixedNow)
    Session.create(tmp_path, False, make_station())
    with pytest.raises(LaunchError, match="already exists"):
        Session.create(tmp_path, False, make_station())
