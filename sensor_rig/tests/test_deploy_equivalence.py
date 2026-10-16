"""D10d: deployment images (RigFrameSource / RigCamera) == training images after the inference-time transform.

Part 1 (unit, always runs): an iPhone frame (real if ``data/raw/2026-10-01`` exists, else a synthetic texture) is
warped into the default rig's physical cameras (inverse remap), then
  (a) export path: forward remap to 1920x1440 / 640x480 -> RGB -> ``iphumi.common.cv_util`` pad+224 (what
      ``gen_replay_buffer_zarr.py`` does to the exported video), and
  (b) deploy path: RigFrameSource with fake sources -> 960x720 BGR -> BGR->RGB (as BaseCamera) ->
      ``behavior_prompting.common.cv_util`` pad+224 (what ``detached_inference.py::_process_obs`` does).
  Tolerance (P2 §7): MAE <= 0.5 grey levels, ECC translation <= 0.05 px at 224.

Part 2 (integration, skips without ``sensor_rig.sim`` or the iPhone data): WP-C's simulator turns a real iPhone demo
into a raw rig episode (sim_rig.yaml); ``MockCamera.replay`` (frozen R4 API, lockstep, recorded stamps) feeds RigCamera
(real-env adapter, stubs for its missing deps). Each packet is compared after the inference transform with
  (a) the export path on the same raw frames, (b) the original iPhone frame the sim used, and (c) when the exporter
  works, the frame of the exported MP4 (H.264), i.e. what the training zarr is built from.
  Tolerance: (a) MAE <= 0.5; (b)/(c) baseline-relative: MAE(deploy, X) <= MAE(export path, X) + 1.0 and <= 4.0.
  Timestamps: packet time == simulator truth capture time of the main frame (exact, ns).

Run with ``-s`` to see the numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from sensor_rig.deploy.frame_source import RigFrameSource
from sensor_rig.drivers.base import Frame
from sensor_rig.rig_config import load_rig
from test_frame_source import FakeSource

import realenv_stubs

realenv_stubs.install()          # puts behavior_prompting/ (and real-env) on sys.path

from iphumi.common.cv_util import get_image_transform_with_border as ip_transform  # noqa: E402

try:
    from behavior_prompting.common.cv_util import get_image_transform_with_border as bp_transform  # noqa: E402
except Exception:  # pragma: no cover - the two functions are identical (R3 §3)
    bp_transform = ip_transform

OUT = (224, 224)
DEMO_1406 = "2026-10-01T21-39-22.626Z_49242_firstreal_demonstration_left"
NUMBERS: dict = {}


# ------------------------------------------------------------------------------------------------- helpers
def to224_train(img_bgr):
    """Training: the exported video is decoded to rgb24 and padded+resized (gen_replay_buffer_zarr.py)."""
    h, w = img_bgr.shape[:2]
    return ip_transform(in_res=(w, h), out_res=OUT)(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def to224_deploy(img_bgr):
    """Deployment: BaseCamera publishes BGR->RGB; detached_inference pads+resizes."""
    h, w = img_bgr.shape[:2]
    return bp_transform(in_res=(w, h), out_res=OUT)(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def interior_mask(src_size, erode_px=3):
    w, h = src_size
    white = np.full((h, w, 3), 255, np.uint8)
    m = (ip_transform(in_res=(w, h), out_res=OUT)(white)[..., 0] == 255).astype(np.uint8)
    return cv2.erode(m, np.ones((2 * erode_px + 1,) * 2, np.uint8)).astype(bool)


def mae(a, b, mask):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))[mask]
    return float(d.mean()), int(d.max())


def ecc_shift(a, b, mask) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    _, warp = cv2.findTransformECC(ga, gb, warp, cv2.MOTION_TRANSLATION, crit, mask.astype(np.uint8), 1)
    return float(np.hypot(warp[0, 2], warp[1, 2]))


def synthetic_texture(w, h, seed):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), 3.0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    for _ in range(25):
        p1 = tuple(int(x) for x in rng.integers(0, [w, h]))
        p2 = tuple(int(x) for x in rng.integers(0, [w, h]))
        cv2.line(img, p1, p2, tuple(int(c) for c in rng.integers(0, 256, 3)), int(rng.integers(1, 6)), cv2.LINE_AA)
    return img


def read_video_frame(path: Path, index: int):
    import av

    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i == index:
                return fr.to_ndarray(format="bgr24")
    raise IndexError(f"{path} has no frame {index}")


def iphone_demo_dir() -> Path | None:
    from conftest import iphone_data_dir

    d = iphone_data_dir() / DEMO_1406
    return d if d.is_dir() and any(d.glob("*.json")) else None


def report(key, **vals):
    NUMBERS[key] = vals
    print(f"\nD10d {key}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in vals.items()))


# ------------------------------------------------------------------------------------------------- part 1
@pytest.fixture(scope="module")
def iphone_frames():
    """(main 1920x1440 BGR, ultrawide 640x480 BGR, K_main, K_uw, source label)."""
    d = iphone_demo_dir()
    if d is not None:
        import json

        name = next(d.glob("*.json")).stem
        js = json.loads((d / f"{name}.json").read_text())
        main = read_video_frame(d / f"{name}_rgb.mp4", 120)
        uw = read_video_frame(d / f"{name}_ultrawidergb.mp4", 12)
        return main, uw, js["mainCameraIntrinsics"], js["ultrawideCameraIntrinsics"], "iphone"
    rig = load_rig("rig_default.yaml")
    K = lambda v: [[v.fx, 0, v.cx], [0, v.fy, v.cy], [0, 0, 1]]  # noqa: E731
    return (synthetic_texture(1920, 1440, 1), synthetic_texture(640, 480, 2), K(rig.virtual["main"].reference),
            K(rig.virtual["ultrawide"].reference), "synthetic")


def test_unit_export_vs_deploy(iphone_frames):
    main_v, uw_v, K_main, K_uw, label = iphone_frames
    rig = load_rig("rig_default.yaml")
    rig = rig.with_virtual_reference("main", K_main, (main_v.shape[1], main_v.shape[0]))
    rig = rig.with_virtual_reference("ultrawide", K_uw, (uw_v.shape[1], uw_v.shape[0]))
    phys = {"main": rig.inverse_remap("main").apply(main_v), "ultrawide": rig.inverse_remap("ultrawide").apply(uw_v)}

    main_src, uw_src = FakeSource("main"), FakeSource("ultrawide")
    t0 = 7_000_000_000_000
    main_src.push(Frame(phys["main"], t0, 0, t0))
    uw_src.push(Frame(phys["ultrawide"], t0 + 5_000_000, 0, t0))
    main_src.finished = uw_src.finished = True
    src = RigFrameSource(rig, sources={"main": main_src, "ultrawide": uw_src}, sync="lockstep")
    src.start()
    packet = src.next_packet()
    assert packet.images["main"].shape == (720, 960, 3)

    for cam, orig in (("main", main_v), ("ultrawide", uw_v)):
        exp = to224_train(rig.remap(cam, "export").apply(phys[cam]))
        dep = to224_deploy(packet.images[cam])
        mask = interior_mask(rig.virtual[cam].size("export"))
        m, mx = mae(dep, exp, mask)
        s = ecc_shift(exp, dep, mask)
        m_orig_exp, _ = mae(exp, to224_train(orig), mask)
        m_orig_dep, _ = mae(dep, to224_train(orig), mask)
        report(f"unit_{cam}_{label}", mae_deploy_vs_export=m, max=mx, ecc_px=s,
               mae_export_vs_original=m_orig_exp, mae_deploy_vs_original=m_orig_dep)
        assert m <= 0.5, (cam, m)
        assert s <= 0.05, (cam, s)
        assert m_orig_dep <= m_orig_exp + 1.0

    # Sensitivity: a 1 px shift at 960x720 (0.233 px at 224) must be visible to both metrics.
    shifted = cv2.warpAffine(packet.images["main"], np.float32([[1, 0, 1], [0, 1, 0]]), (960, 720),
                             borderMode=cv2.BORDER_REPLICATE)
    exp = to224_train(rig.remap("main", "export").apply(phys["main"]))
    dep = to224_deploy(shifted)
    mask = interior_mask((1920, 1440))
    m_shift, _ = mae(dep, exp, mask)
    s_shift = ecc_shift(exp, dep, mask)
    report(f"unit_main_{label}_1px_shift", mae=m_shift, ecc_px=s_shift)
    assert m_shift > 2 * NUMBERS[f"unit_main_{label}"]["mae_deploy_vs_export"]
    assert abs(s_shift - 224 / 960) < 0.05


# ------------------------------------------------------------------------------------------------- part 2
@pytest.fixture(scope="module")
def sim_episode(tmp_path_factory):
    sim = pytest.importorskip("sensor_rig.sim.simulator", reason="sensor_rig.sim (WP-C) not available")
    d = iphone_demo_dir()
    if d is None:
        pytest.skip("iPhone demo for the D10d integration test not found (set SENSOR_RIG_IPHONE_DATA)")
    raw_root = tmp_path_factory.mktemp("d10d_raw")
    n = int(os.environ.get("SENSOR_RIG_D10D_FRAMES", "90"))
    opts = sim.SimOptions(max_frames=n, import_trajectory=True)
    ep = sim.simulate_raw_from_iphone(d, sim.SIM_RIG_PATH, raw_root, opts)
    return ep, sim.load_truth(ep), d


def _export_frames(ep: Path, tmp: Path):
    """Export the raw episode and decode its videos; None if the exporter is unavailable or fails."""
    try:
        from sensor_rig.export.exporter import export_episode
    except Exception as e:  # pragma: no cover
        return None, f"exporter not importable: {e!r}"
    try:
        res = export_episode(ep, tmp)
    except Exception as e:  # pragma: no cover
        return None, f"export failed: {e!r}"
    if res.status != "exported" or res.out_dir is None:
        return None, f"export {res.status}: {res.reason} {res.warnings}"
    out = Path(res.out_dir)
    mains = sorted(out.rglob("*_rgb.mp4"))
    uws = sorted(out.rglob("*_ultrawidergb.mp4"))
    if not mains or not uws:
        return None, f"export produced no videos ({res!r})"
    import av

    def decode(p):
        with av.open(str(p)) as c:
            return [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]

    return {"main": decode(mains[0]), "ultrawide": decode(uws[0])}, "ok"


def test_integration_sim_episode(sim_episode, tmp_path):
    ep, truth, iphone_dir = sim_episode
    from real_env.peripherals.rig_camera import RigCamera

    from sensor_rig.raw_episode import RawEpisode
    from sensor_rig.sim.iphone_reader import IphoneEpisode

    cam = RigCamera(
        name="right_wrist_camera_0", server_endpoint="tcp://localhost:16702", logger_endpoint="tcp://localhost:55556",
        camera_latency_s=0.0, display_resolution_WH=None,
        camera_configs={"main": {"type": "rgb", "width": 960, "height": 720, "fps": 60},
                        "ultrawide": {"type": "rgb", "width": 960, "height": 720, "fps": 60}},
        rig_config_path=None, backend="mock",
        backend_opts={"episode_dir": str(ep), "realtime": False, "preserve_timestamps": True},
        publish_mode="per_main_frame", sync="lockstep", stats_period_s=0, config_str="{}",
    )
    raw = RawEpisode(ep)
    rig = cam.rig
    n_main = len(raw.camera_times("main"))
    packets = []
    try:
        for _ in range(n_main):
            frames, ts = cam.get_images_bgr_hwc_and_timestamp()
            packets.append((dict(frames), ts, cam.source.last_packet))
        with pytest.raises(RuntimeError, match="exhausted"):
            cam.get_images_bgr_hwc_and_timestamp()
    finally:
        cam.source.stop()

    # Timestamps: exactly the simulator's true capture time of each main frame (latency applied once, here).
    seqs = [p.seq["main"] for _, _, p in packets]
    assert seqs == list(range(n_main))
    t_true = truth["t_ns"][truth["main_src_index"]]
    assert [p.t_capture_ns["main"] for _, _, p in packets] == [int(t) for t in t_true]
    assert all(ts == p.t_capture_ns["main"] / 1e9 for _, ts, p in packets)
    skews = np.array([abs(p.skew_s) for _, _, p in packets])
    # Pairing: the globally nearest ultrawide frame by capture time (the iPhone UW runs at ~10 Hz).
    t_uw_cap = rig.capture_time_ns("ultrawide", raw.camera_times("ultrawide"))
    for _, _, p in packets:
        d = np.abs(t_uw_cap - p.t_capture_ns["main"])
        assert abs(p.t_capture_ns["main"] - p.t_capture_ns["ultrawide"]) == d.min()

    E = IphoneEpisode(iphone_dir)
    orig_main = []
    for i, img in enumerate(E.iter_main_frames(bgr=True)):
        if i > int(truth["main_src_index"].max()):
            break
        orig_main.append(img)
    orig_uw = []
    for j, img in enumerate(E.iter_uw_frames(bgr=True)):
        if j > int(truth["uw_src_frame"].max()):
            break
        orig_uw.append(img)
    exported, export_status = _export_frames(ep, tmp_path / "export")

    stride = max(1, n_main // 30)
    res = {c: {"dep_exp": [], "dep_orig": [], "exp_orig": [], "dep_mp4": [], "exp_mp4": [], "ecc": []}
           for c in ("main", "ultrawide")}
    masks = {c: interior_mask(rig.virtual[c].size("export")) for c in ("main", "ultrawide")}
    uw_done = set()
    for frames, _, p in packets[::stride]:
        for c in ("main", "ultrawide"):
            seq = p.seq[c]
            if c == "ultrawide":
                if seq in uw_done:
                    continue
                uw_done.add(seq)
            phys = raw.read_image(c, seq)
            exp = to224_train(rig.remap(c, "export").apply(np.ascontiguousarray(phys)))
            dep = to224_deploy(frames[c])
            src_img = orig_main[int(truth["main_src_index"][seq])] if c == "main" else \
                orig_uw[int(truth["uw_src_frame"][seq])]
            org = to224_train(src_img)
            m = masks[c]
            res[c]["dep_exp"].append(mae(dep, exp, m)[0])
            res[c]["dep_orig"].append(mae(dep, org, m)[0])
            res[c]["exp_orig"].append(mae(exp, org, m)[0])
            res[c]["ecc"].append(ecc_shift(exp, dep, m))
            if exported is not None:
                vid = exported[c]
                k = seq                      # native 60 Hz grid, no trim: export frame k == raw frame k
                if k < len(vid) and len(vid) == (n_main if c == "main" else len(raw.camera_times(c))):
                    mp4 = to224_train(vid[k])
                    res[c]["dep_mp4"].append(mae(dep, mp4, m)[0])
                    res[c]["exp_mp4"].append(mae(exp, mp4, m)[0])

    for c, r in res.items():
        vals = {k: float(np.mean(v)) for k, v in r.items() if v}
        vals["n"] = len(r["dep_exp"])
        vals["ecc_max"] = float(np.max(r["ecc"]))
        report(f"integration_{c}", **vals)
        assert vals["dep_exp"] <= 0.5, (c, vals)
        assert vals["ecc_max"] <= 0.05, (c, vals)
        assert vals["dep_orig"] <= vals["exp_orig"] + 1.0 and vals["dep_orig"] <= 4.0, (c, vals)
        if r["dep_mp4"]:
            assert vals["dep_mp4"] <= vals["exp_mp4"] + 1.0 and vals["dep_mp4"] <= 4.0, (c, vals)
    report("integration_meta", frames=n_main, skew_ms_max=float(skews.max() * 1e3), export=export_status,
           mp4_compared={c: len(res[c]["dep_mp4"]) for c in res})
