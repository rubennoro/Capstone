"""paths: the folders from the editable install; the binary from config.yaml (control_deploy/STRUCTURE.md 2)."""
from __future__ import annotations

import deploy_contracts

from control_deploy import paths


def test_the_folders_resolve_from_the_editable_install():
    d = paths.deploy_dir()
    assert d.name == "control_deploy" and (d / "pyproject.toml").is_file()
    assert paths.franka_control_dir() == d / "franka_control"
    assert paths.config_file().is_file()
    assert paths.default_station() == d / "station.yaml"
    assert paths.contracts_dir() == d.parent / "contracts"
    assert paths.contracts_dir().resolve() == deploy_contracts.paths.contracts_dir().resolve()
    assert (paths.repo_root() / "deployment").is_dir()


def test_the_binary_comes_from_config_yaml_or_the_station():
    assert paths.binary() == paths.franka_control_dir() / "build" / "franka_control"
    assert paths.binary({"binary": "build-nolibfranka/franka_control"}) == \
        paths.franka_control_dir() / "build-nolibfranka" / "franka_control"
    assert str(paths.binary({"binary": "/opt/fc"})) == "/opt/fc"
