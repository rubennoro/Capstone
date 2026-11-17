"""umiproc.paths: the data root (default, $UMIPROC_ROOT, --root; /data, /data/incoming and dataproc's folders
refused) and the repo config paths.

(Named test_umiproc_paths.py, not test_paths.py: tests/dp/test_paths.py exists and, without __init__.py files,
pytest cannot collect two test modules with the same base name.)
"""

import os
from pathlib import Path

import pytest

from umiproc import paths


def test_default_root(monkeypatch):
    monkeypatch.delenv("UMIPROC_ROOT", raising=False)
    assert paths.data_root() == Path.home() / "data" / "umi"
    assert paths.DEFAULT_ROOT == Path.home() / "data" / "umi"


def test_env_root_and_explicit_path(monkeypatch, tmp_path):
    monkeypatch.setenv("UMIPROC_ROOT", str(tmp_path / "env"))
    assert paths.data_root() == tmp_path / "env"
    assert paths.data_root(tmp_path / "arg") == tmp_path / "arg"           # an explicit path wins over the env
    monkeypatch.setenv("UMIPROC_ROOT", "")
    assert paths.data_root() == paths.DEFAULT_ROOT                         # empty means unset


def test_root_is_absolute_and_expanded(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert paths.data_root("rel") == tmp_path / "rel"
    assert paths.data_root("~/x") == Path.home() / "x"


@pytest.mark.parametrize("spec", ["/data", "/data/", "/data/../data", "/data/./"])
def test_slash_data_is_refused(spec):
    with pytest.raises(ValueError, match="/data/incoming"):
        paths.data_root(spec)


def test_slash_data_refused_from_env(monkeypatch):
    monkeypatch.setenv("UMIPROC_ROOT", "/data")
    with pytest.raises(ValueError, match="refused"):
        paths.data_root()


def test_symlink_to_slash_data_is_refused(tmp_path):
    link = tmp_path / "root"
    os.symlink("/data", link)
    with pytest.raises(ValueError, match="refused"):
        paths.data_root(link)


def test_subfolder_of_slash_data_outside_incoming_is_allowed():
    assert paths.data_root("/data/umi") == Path("/data/umi")


@pytest.mark.parametrize("spec", ["/data/incoming", "/data/incoming/", "/data/incoming/x", "/data/incoming/x/y",
                                  "/data/umi/../incoming", "/data/incoming/./x"])
def test_pi_incoming_and_inside_it_are_refused(monkeypatch, spec):
    with pytest.raises(ValueError, match="Pi's delivery folder"):
        paths.data_root(spec)
    monkeypatch.setenv("UMIPROC_ROOT", spec)
    with pytest.raises(ValueError, match="refused"):
        paths.data_root()


@pytest.mark.parametrize("target", ["/data/incoming", "/data/incoming/x"])
def test_symlink_into_pi_incoming_is_refused(tmp_path, target):
    link = tmp_path / "root"
    os.symlink(target, link)
    with pytest.raises(ValueError, match="Pi's delivery folder"):
        paths.data_root(link)
    with pytest.raises(ValueError, match="Pi's delivery folder"):
        paths.data_root(link / "sub")                                      # inside the link, too


def test_pi_incoming_reached_through_a_symlinked_parent_is_refused(tmp_path):
    os.symlink("/data", tmp_path / "d")
    with pytest.raises(ValueError, match="Pi's delivery folder"):
        paths.data_root(tmp_path / "d" / "incoming" / "x")


@pytest.mark.parametrize("sub", paths.DATAPROC_MANAGED)
@pytest.mark.parametrize("inside", ["", "x", "x/y"])
def test_dataproc_default_managed_subfolders_are_refused(monkeypatch, sub, inside):
    monkeypatch.delenv("DATAPROC_ROOT", raising=False)
    bad = Path.home() / "data" / sub / inside
    with pytest.raises(ValueError, match="folder dataproc manages"):
        paths.data_root(bad)


@pytest.mark.parametrize("sub", paths.DATAPROC_MANAGED)
def test_dataproc_env_managed_subfolders_are_refused(monkeypatch, tmp_path, sub):
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path / "dp"))
    for bad in (tmp_path / "dp" / sub, tmp_path / "dp" / sub / "x"):
        with pytest.raises(ValueError, match=r"folder dataproc manages .*\$DATAPROC_ROOT"):
            paths.data_root(bad)
    with pytest.raises(ValueError, match="folder dataproc manages"):     # ~/data's are refused as well
        paths.data_root(Path.home() / "data" / sub / "x")


def test_symlink_into_dataproc_managed_subfolder_is_refused(monkeypatch, tmp_path):
    (tmp_path / "dp" / "tasks" / "t").mkdir(parents=True)
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path / "dp"))
    os.symlink(tmp_path / "dp" / "tasks" / "t", tmp_path / "link")
    with pytest.raises(ValueError, match="folder dataproc manages"):
        paths.data_root(tmp_path / "link")
    os.symlink(tmp_path / "dp", tmp_path / "dplink")                      # via a link to the root's parent
    with pytest.raises(ValueError, match="folder dataproc manages"):
        paths.data_root(tmp_path / "dplink" / "rigs")


def test_default_and_other_dataproc_subfolders_stay_allowed(monkeypatch, tmp_path):
    monkeypatch.delenv("UMIPROC_ROOT", raising=False)
    monkeypatch.delenv("DATAPROC_ROOT", raising=False)
    assert paths.data_root() == Path.home() / "data" / "umi"
    assert paths.data_root(Path.home() / "data" / "umi" / "x") == Path.home() / "data" / "umi" / "x"
    assert paths.data_root(Path.home() / "data" / "tasks_old") == Path.home() / "data" / "tasks_old"
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path / "dp"))
    assert paths.data_root() == Path.home() / "data" / "umi"
    assert paths.data_root(tmp_path / "dp" / "umi") == tmp_path / "dp" / "umi"


def test_repo_config():
    assert paths.REPO.name == "umi-processing"
    assert (paths.REPO / "umiproc" / "paths.py").is_file()
    assert paths.repo_config("rigs", "rig_pi_v3.yaml").is_file()
    assert paths.repo_config("slam", "rig_pi_v3.yaml") == paths.REPO / "configs" / "slam" / "rig_pi_v3.yaml"


def test_lens_paths(tmp_path):
    assert (paths.lens_dir(tmp_path, "pi-2026-11-16_episode_000123")
            == tmp_path / "lens" / "pi-2026-11-16_episode_000123")
    assert paths.rig_lens_file(tmp_path, "rig_pi_v3") == tmp_path / "rigs" / "rig_pi_v3.lens.yaml"
    for bad in ("..", ".hidden", "a/b", ""):
        with pytest.raises(ValueError):
            paths.lens_dir(tmp_path, bad)
    with pytest.raises(ValueError):
        paths.rig_lens_file(tmp_path, "../x")
