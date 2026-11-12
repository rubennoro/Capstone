from __future__ import annotations

from pathlib import Path

import pytest

from deploy import LaunchError, ManagedProcess, load_process_class

GOOD = '''
from deploy import ManagedProcess


class PolicyProcess(ManagedProcess):
    NAME = "{name}"
    READY_HINT = "warm-up"

    def program(self):
        return ["python", "-m", "policy"]

    def checks(self):
        return []


PROCESS_CLASS = PolicyProcess
'''


def launch_file(tmp_path: Path, folder: str, text: str) -> Path:
    (tmp_path / folder).mkdir()
    (tmp_path / folder / "launch.py").write_text(text)
    return tmp_path / folder


def test_loads_by_path(tmp_path):
    cls = load_process_class(launch_file(tmp_path, "policy", GOOD.format(name="policy")))
    assert issubclass(cls, ManagedProcess) and cls.NAME == "policy"
    assert cls.__module__ == "launch_policy"


def test_missing_file(tmp_path):
    (tmp_path / "policy").mkdir()
    with pytest.raises(LaunchError, match="no launch.py"):
        load_process_class(tmp_path / "policy")


def test_missing_process_class(tmp_path):
    text = GOOD.format(name="policy").replace("PROCESS_CLASS = PolicyProcess", "")
    with pytest.raises(LaunchError, match="does not define PROCESS_CLASS"):
        load_process_class(launch_file(tmp_path, "policy", text))


def test_wrong_base_class(tmp_path):
    text = "class PolicyProcess:\n    NAME = 'policy'\n\nPROCESS_CLASS = PolicyProcess\n"
    with pytest.raises(LaunchError, match="not a ManagedProcess subclass"):
        load_process_class(launch_file(tmp_path, "policy", text))


def test_name_must_match_folder(tmp_path):
    with pytest.raises(LaunchError, match="expected the folder name 'policy'"):
        load_process_class(launch_file(tmp_path, "policy", GOOD.format(name="message_comms")))


def test_abstract_class_refused(tmp_path):
    text = GOOD.format(name="policy").replace("    def checks(self):\n        return []\n", "")
    with pytest.raises(LaunchError, match="does not write: checks"):
        load_process_class(launch_file(tmp_path, "policy", text))


def test_import_error_reported(tmp_path):
    with pytest.raises(LaunchError, match="failed to import: ModuleNotFoundError"):
        load_process_class(launch_file(tmp_path, "policy", "import no_such_module_xyz\n"))
