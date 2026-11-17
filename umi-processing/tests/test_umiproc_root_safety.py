"""No umiproc code path can reach dataproc's data root (``~/data``, ``$DATAPROC_ROOT``) or ``/data``: the copied
dataproc code takes its root from ``umiproc.paths.data_root`` (``$UMIPROC_ROOT``, default ``~/data/umi``), and those
roots are refused outright.

The end-to-end test runs in a subprocess with ``HOME`` and ``$DATAPROC_ROOT`` pointed at temporary folders (the
default roots are computed from ``HOME`` at import), runs every umiproc entry point that touches the root with
``$UMIPROC_ROOT`` unset, and checks what appeared where.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from umiproc import paths
from umiproc.dp import paths as dp_paths

PKG = Path(__file__).resolve().parents[1]


def test_one_source_of_the_root(monkeypatch, tmp_path):
    assert dp_paths.ENV_ROOT == paths.ENV_ROOT == "UMIPROC_ROOT"
    assert dp_paths.DEFAULT_ROOT == paths.DEFAULT_ROOT == Path.home() / "data" / "umi"
    monkeypatch.delenv("UMIPROC_ROOT", raising=False)
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path / "dataproc"))       # never read as a root
    assert dp_paths.data_root().path == paths.data_root() == paths.DEFAULT_ROOT
    monkeypatch.setenv("UMIPROC_ROOT", str(tmp_path / "umi"))
    assert dp_paths.data_root().path == paths.data_root() == tmp_path / "umi"
    assert dp_paths.data_root(tmp_path / "x").path == tmp_path / "x"


@pytest.mark.parametrize("make", [
    lambda tmp: Path("/data"),
    lambda tmp: Path.home() / "data",                     # dataproc's default root
    lambda tmp: Path.home() / "data" / ".",
    lambda tmp: tmp / "dataproc",                         # = $DATAPROC_ROOT below
])
def test_refused_roots(monkeypatch, tmp_path, make):
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path / "dataproc"))
    bad = make(tmp_path)
    for f in (paths.data_root, lambda p: dp_paths.data_root(p), lambda p: dp_paths.DataRoot(Path(p))):
        with pytest.raises(ValueError, match="refused"):
            f(bad)
    monkeypatch.setenv("UMIPROC_ROOT", str(bad))
    with pytest.raises(ValueError, match="refused"):
        dp_paths.data_root()


def test_symlink_to_dataproc_root_is_refused(monkeypatch, tmp_path):
    (tmp_path / "dataproc").mkdir()
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path / "dataproc"))
    os.symlink(tmp_path / "dataproc", tmp_path / "link")
    with pytest.raises(ValueError, match="dataproc's data root"):
        paths.data_root(tmp_path / "link")


def test_subfolders_are_allowed(monkeypatch, tmp_path):
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path / "dataproc"))
    assert paths.data_root(tmp_path / "dataproc" / "umi") == tmp_path / "dataproc" / "umi"
    assert paths.data_root(Path.home() / "data" / "umi") == Path.home() / "data" / "umi"


def test_no_module_reads_dataproc_root():
    """Only umiproc/paths.py names $DATAPROC_ROOT (to refuse it); nothing reads it as an environment variable."""
    hits = []
    for p in sorted((PKG / "umiproc").rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        if re.search(r"""environ(\.get)?\s*[\[(]\s*["']DATAPROC_ROOT""", text) or re.search(
                r"""getenv\(\s*["']DATAPROC_ROOT""", text):
            hits.append(str(p.relative_to(PKG)))
        if "DATAPROC_ROOT" in text and p.relative_to(PKG).as_posix() not in ("umiproc/paths.py",
                                                                             "umiproc/dp/paths.py"):
            hits.append(f"{p.relative_to(PKG)} mentions DATAPROC_ROOT")
    assert not hits, hits


_SCRIPT = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])                    # tests/: umiproc_testlib
import umiproc_testlib as tl
from umiproc import paths, recordings as R
from umiproc.dp import paths as dp_paths
from umiproc.cli import main as umiproc_main
from umiproc.dp.cli import main as dp_main
from umiproc.sensors import ensure_rig_files, load_sensors
from umiproc import slam_inputs as SI
from umiproc.slam_inputs import write_slam_inputs

home = Path(os.environ["HOME"])
out = {"default": str(paths.data_root()), "dp_default": str(dp_paths.data_root().path)}
src = Path(sys.argv[2])
tl.pi_recording(src, 1)
tl.pi_recording(src, 2)
root = paths.data_root()
ensure_rig_files(root, "rig_pi_v3", log=lambda *a: None)
build = Path(sys.argv[3]) / "fake_build"                    # a native/build.sh-like folder; ldd replaced
build.mkdir()
(build / "umiproc_slam").write_bytes(b"binary")
(build / "libORB_SLAM3.so").write_bytes(b"lib")
(build / "BUILD_INFO").write_text(f"library {build / 'libORB_SLAM3.so'}\npatch 0123456789abcdef 0001-a.patch\n")
SI.loaded_orbslam3_library = lambda b: build / "libORB_SLAM3.so"
write_slam_inputs(load_sensors(root, "rig_pi_v3", log=lambda *a: None), Path(sys.argv[3]) / "slam_inputs",
                  binary=build / "umiproc_slam")
t = R.take_recording(root, src, "episode_000001", R.MAP, rig_id="rig_pi_v3", task="block_raise",
                     log=lambda *a: None)
R.take_recording(root, src, "episode_000002", R.DEMONSTRATION, rig_id="rig_pi_v3", folder=t.folder,
                 log=lambda *a: None)
out["status"] = umiproc_main(["status"])
out["status_folder"] = umiproc_main(["status", t.folder])
out["dp_status"] = dp_main(["status", t.folder])
out["dp_sim"] = dp_main(["sim", "--out", str(Path(sys.argv[3]) / "sim_out"), "--episodes", "0"])
for bad in (str(home / "data"), os.environ["DATAPROC_ROOT"]):
    os.environ["UMIPROC_ROOT"] = bad
    out[f"refused {bad}"] = umiproc_main(["status"])
print("RESULT " + json.dumps(out))
'''


def test_nothing_reaches_dataproc_roots(tmp_path):
    home, dp_root, work = tmp_path / "home", tmp_path / "dataproc_root", tmp_path / "work"
    for d in (home, dp_root, work):
        d.mkdir()
    (home / "data").mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("UMIPROC_ROOT", "PYTHONPATH")}
    env.update(HOME=str(home), DATAPROC_ROOT=str(dp_root))
    p = subprocess.run([sys.executable, "-c", _SCRIPT, str(PKG / "tests"), str(tmp_path / "src"), str(work)],
                       cwd=work, env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    res = json.loads(p.stdout.split("RESULT ", 1)[1])
    umi = home / "data" / "umi"
    assert res["default"] == res["dp_default"] == str(umi)
    assert res["status"] == res["status_folder"] == res["dp_status"] == res["dp_sim"] == 0
    assert res[f"refused {home / 'data'}"] == res[f"refused {dp_root}"] == 1
    assert "refused" in p.stderr
    # everything went to ~/data/umi; nothing next to it in ~/data, nothing in $DATAPROC_ROOT
    assert sorted(x.name for x in (home / "data").iterdir()) == ["umi"]
    assert list(dp_root.iterdir()) == []
    assert (umi / "pi_sources.csv").is_file() and (umi / "tasks" / "block_raise" / "map").is_dir()
    assert (umi / "rigs" / "rig_pi_v3.yaml").is_file() and (umi / "rigs" / "rig_pi_v3.slam.yaml").is_file()
    # and nothing else in HOME but ~/data (and what Python or OpenCV may cache there)
    assert {x.name for x in home.iterdir()} - {".cache", ".config", ".local"} == {"data"}
