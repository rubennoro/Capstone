"""Deployment side of the iPhone regression guard: real-env's iPhone camera server loads exactly as at the baseline.

    python -m pytest scripts/regression/test_iphone_deploy_unchanged.py -v      [BASELINE_REF=0ddedef]

For two copies of ``behavior_prompting/deps/real-env`` (the baseline ref, extracted with ``git archive`` into a temp
dir, and the current working tree) a subprocess ("probe", this same file with ``--probe <real-env root>``):

1. composes every ``configs/peripherals/iphone_*.yaml`` with Hydra (``initialize_config_dir``, defaults resolved);
2. imports ``real_env.peripherals.iphone`` from that root (asserted via ``__file__``) and calls the REAL entry point
   ``run_iphone()`` for each config name (``sys.argv = ["iphone", <name>]``, i.e. ``hydra.initialize`` with the
   module-relative config path + ``hydra.utils.instantiate``) with ``iPhone.run`` replaced by a capture, so the
   constructed iPhone/BaseCamera object is recorded instead of entering the camera loop;
3. dumps the composed configs, the constructed object's state (name, endpoints, latency, camera configs, peertalk
   ports + selected UDID, RMQ topics, VideoLogger attrs, config_str), the constructor signatures and method lists,
   and the ``iphone`` console-script line of pyproject.toml.

The tests assert current == baseline. Missing third-party deps (robotmq, robologger, robot_utils, loguru, pypeertalk)
are ALWAYS replaced by small local fakes inside the probe (even where the real packages exist), so the probe never
opens sockets or USB connections and both sides see identical stand-ins. The fakes are defined here rather than
imported from sensor_rig/tests/realenv_stubs.py on purpose: the guard must not depend on the code it guards against.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tarfile
import io
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REAL_ENV_REL = "behavior_prompting/deps/real-env"
BASELINE_REF = os.environ.get("BASELINE_REF", "0ddedef")


# ================================================================================================= probe side

def _install_fakes() -> None:
    class RMQServer:
        def __init__(self, server_name="", server_endpoint=""):
            self.server_name, self.server_endpoint, self.topics = server_name, server_endpoint, {}

        def add_shared_memory_topic(self, topic, message_remaining_time_s=1.0, shared_memory_size_gb=0.5):
            self.topics[topic] = ["shm", message_remaining_time_s, shared_memory_size_gb]

        def add_topic(self, topic, message_remaining_time_s=1.0):
            self.topics[topic] = ["plain", message_remaining_time_s]

    class VideoLogger:
        def __init__(self, name, endpoint, attr, codec="h264_nvenc", depth_range=(0.0, 4.0)):
            self.name, self.endpoint, self.attr, self.codec, self.depth_range = name, endpoint, attr, codec, depth_range

    class EncoderOpts:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Device:
        def __init__(self, udid):
            self.udid = udid

    class PeerTalkClient:
        def __init__(self, device, port):
            self.device, self.port = device, port

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        m.__path__ = []
        sys.modules[name] = m
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, m)

    mod("robotmq", RMQServer=RMQServer, RMQClient=object, serialize=lambda x: x, deserialize=lambda x: x)
    mod("loguru", logger=types.SimpleNamespace(**{k: (lambda *a, **k: None) for k in
                                                   ("info", "debug", "warning", "error", "exception")}))
    mod("robot_utils")
    mod("robot_utils.logging_utils", echo_exception=lambda: "")
    mod("robot_utils.image_utils", resize_frame_without_distortion=lambda f, s, d: f)
    mod("robologger")
    mod("robologger.loggers")
    mod("robologger.loggers.video_logger", VideoLogger=VideoLogger)
    mod("robologger.utils")
    mod("robologger.utils.huecodec", depth2logrgb=lambda d, zrange=(0.0, 4.0): d, EncoderOpts=EncoderOpts)
    mod("pypeertalk", PeerTalkClient=PeerTalkClient,
        get_connected_devices=lambda: [Device(os.environ["FAKE_IPHONE_UDID"])])


def _state(obj) -> dict:
    out = {}
    for k, v in sorted(vars(obj).items()):
        if k == "rmq_server":
            v = {"name": v.server_name, "endpoint": v.server_endpoint, "topics": v.topics}
        elif k == "video_logger":
            v = {"name": v.name, "endpoint": v.endpoint, "attr": v.attr, "codec": v.codec,
                 "depth_range": list(v.depth_range)}
        elif k == "peertalk_clients":
            v = {cam: {"port": c.port, "udid": c.device.udid} for cam, c in v.items()}
        elif k == "opts":
            v = {"EncoderOpts": v.kwargs}
        elif k == "config_str":
            v = json.loads(v) if v else v
        out[k] = v
    return json.loads(json.dumps(out, default=repr))


def probe(real_env_root: str) -> dict:
    root = Path(real_env_root).resolve()
    sys.path.insert(0, str(root))
    _install_fakes()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import real_env.peripherals.base_camera as base_mod
    import real_env.peripherals.iphone as iphone_mod

    assert Path(iphone_mod.__file__).resolve().is_relative_to(root), iphone_mod.__file__
    cfg_dir = root / "real_env" / "configs" / "peripherals"
    names = sorted(p.stem for p in cfg_dir.glob("iphone_*.yaml"))
    res = {"iphone_module": str(Path(iphone_mod.__file__).relative_to(root)), "config_files": names,
           "composed": {}, "instances": {}}
    for n in names:
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cfg = compose(config_name=n)
        res["composed"][n] = {"raw": OmegaConf.to_container(cfg, resolve=False),
                              "resolved": OmegaConf.to_container(cfg, resolve=True)}

    captured = []
    iphone_mod.iPhone.run = lambda self: captured.append(self)
    for n in names:
        os.environ["FAKE_IPHONE_UDID"] = res["composed"][n]["raw"].get("iphone_udid") or "FAKE-ONLY-DEVICE"
        sys.argv = ["iphone", n[len("iphone_"):]]
        iphone_mod.run_iphone()
        obj = captured.pop()
        res["instances"][n] = {"class": type(obj).__qualname__, "state": _state(obj)}

    res["signatures"] = {"iPhone.__init__": str(inspect.signature(iphone_mod.iPhone.__init__)),
                         "BaseCamera.__init__": str(inspect.signature(base_mod.BaseCamera.__init__)),
                         "iPhone.get_images_bgr_hwc_and_timestamp":
                             str(inspect.signature(iphone_mod.iPhone.get_images_bgr_hwc_and_timestamp))}
    res["methods"] = {c.__name__: sorted(k for k, v in vars(c).items() if callable(v))
                      for c in (iphone_mod.iPhone, base_mod.BaseCamera, base_mod.BaseCameraClient)}
    pyproject = (root / "pyproject.toml").read_text().splitlines()
    res["entry_point"] = [ln.strip() for ln in pyproject if ln.strip().startswith("iphone ")]
    return res


# ================================================================================================= pytest side

def _run_probe(root: Path) -> dict:
    p = subprocess.run([sys.executable, __file__, "--probe", str(root)], capture_output=True, text=True,
                       env={**os.environ, "HYDRA_FULL_ERROR": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    assert p.returncode == 0, f"probe failed for {root}:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    return json.loads(p.stdout.strip().splitlines()[-1])


def _baseline_root(tmp: Path) -> Path:
    p = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", BASELINE_REF, REAL_ENV_REL],
                       capture_output=True)
    assert p.returncode == 0, p.stderr.decode()
    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as t:
        t.extractall(tmp)
    return tmp / REAL_ENV_REL


import pytest  # noqa: E402


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    base = _run_probe(_baseline_root(tmp_path_factory.mktemp("baseline")))
    cur = _run_probe(REPO / REAL_ENV_REL)
    return base, cur


def test_iphone_config_files_unchanged(probes):
    base, cur = probes
    assert cur["config_files"] == base["config_files"]
    assert {"iphone_wrist", "iphone_wrist_right", "iphone_wrist_left"} <= set(cur["config_files"])


@pytest.mark.parametrize("name", ["iphone_wrist", "iphone_wrist_right", "iphone_wrist_left", "iphone_third"])
def test_composed_config_matches_baseline(probes, name):
    base, cur = probes
    assert cur["composed"][name] == base["composed"][name]


@pytest.mark.parametrize("name", ["iphone_wrist", "iphone_wrist_right", "iphone_wrist_left", "iphone_third"])
def test_run_iphone_constructs_identical_camera(probes, name):
    base, cur = probes
    assert cur["instances"][name] == base["instances"][name]
    st = cur["instances"][name]["state"]
    assert set(st["peertalk_clients"]) == set(st["camera_configs"])  # one stream per configured camera


def test_iphone_class_interface_unchanged(probes):
    base, cur = probes
    assert cur["iphone_module"] == base["iphone_module"] == "real_env/peripherals/iphone.py"
    assert cur["signatures"] == base["signatures"]
    assert cur["methods"] == base["methods"]


def test_iphone_console_script_unchanged(probes):
    base, cur = probes
    assert cur["entry_point"] == base["entry_point"] == ['iphone = "real_env.peripherals.iphone:run_iphone"']


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        result = probe(sys.argv[2])
        print(json.dumps(result))
    else:
        sys.exit(pytest.main([__file__, "-v"]))
