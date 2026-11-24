"""paths.py (STRUCTURE.md 1.4): the four paths resolve next to the package; a missing limits.yaml gives the hint."""
from __future__ import annotations

from pathlib import Path

import pytest

import deploy_contracts
from deploy_contracts import paths


def test_contracts_dir_is_the_parent_of_the_package():
    root = paths.contracts_dir()
    assert root == Path(deploy_contracts.__file__).resolve().parents[1]
    assert root.name == "contracts" and root.parent.name == "deployment"
    assert (root / "deploy_contracts" / "__init__.py").is_file()


def test_the_four_paths():
    root = paths.contracts_dir()
    assert paths.limits_file() == root / "limits.yaml" and paths.limits_file().is_file()
    assert paths.frames_file() == root / "frames.yaml" and paths.frames_file().is_file()
    assert paths.link_file() == root / "link.yaml" and paths.link_file().is_file()
    assert paths.examples_dir() == root / "examples" and (paths.examples_dir() / "index.yaml").is_file()
    assert deploy_contracts.examples_dir() == paths.examples_dir()


def test_files_are_not_inside_the_package():
    pkg = Path(deploy_contracts.__file__).resolve().parent
    assert not any((pkg / name).exists() for name in ("limits.yaml", "frames.yaml", "link.yaml", "examples"))


def test_missing_limits_gives_the_install_hint(tmp_path, monkeypatch):
    fake = tmp_path / "site-packages" / "deploy_contracts" / "paths.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    monkeypatch.setattr(paths, "_package_file", lambda: fake)
    for fn in (paths.contracts_dir, paths.limits_file, paths.frames_file, paths.link_file, paths.examples_dir):
        with pytest.raises(FileNotFoundError) as info:
            fn()
        assert "deploy_contracts must be installed editable from deployment/contracts: uv pip install -e" \
            in str(info.value)
