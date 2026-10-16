import json

import cv2
import numpy as np
import pytest

from sensor_rig.calib.align import (
    align_main_from_chain, align_ultrawide, align_ultrawide_frames, detect_tags, kabsch, rotation_from_image_pair,
    tcp_pose_in_arkit_frame, with_virtual_rotation,
)
from sensor_rig.calib.check import pad_resize_square, render_check, run_calib_check
from sensor_rig.calib.cli import main as calib_main
from sensor_rig.camera_model import VirtualPinhole
from sensor_rig.rig_config import DEFAULT_RIG_PATH, load_rig
from sensor_rig.transforms import D_CV_ARKIT, inv_T, make_T, rot_x, rot_y, rot_z, rotation_angle_deg

# iPhone17,1 entry of gripper_calibration.yaml: I_T_TCP (TCP in the ARKit main camera frame)
IPHONE_TCP = np.array([[1, 0, 0, 0], [0, -0.9659258, 0.25881904, -0.06441],
                       [0, -0.25881904, -0.9659258, -0.211397], [0, 0, 0, 1]])
# closed-gripper finger tags in the (virtual) ultrawide frame, facing the camera
TAG_POSES = {0: (rot_y(15) @ rot_x(150), np.array([-0.028, 0.045, 0.098])),
             1: (rot_y(-15) @ rot_x(150), np.array([0.028, 0.045, 0.098]))}


@pytest.fixture(scope="module")
def rig():
    return load_rig(DEFAULT_RIG_PATH)


def render_tags(K, size=(640, 480), poses=TAG_POSES, seed=0, marker_size=0.016):
    """Synthetic camera image (pinhole K) of ArUco DICT_4X4_50 tags with a white quiet zone on a textured background."""
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.uniform(60, 190, (size[1], size[0])).astype(np.float32), (0, 0), 3)
    img = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for tid, (R, t) in poses.items():
        n = 600
        pad = n // 4
        m = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(d, tid, n), pad, pad, pad, pad,
                               cv2.BORDER_CONSTANT, value=255)
        s, h = marker_size / n, marker_size / 2
        A = np.array([[s, 0, -pad * s - h], [0, -s, h + pad * s], [0, 0, 1]])   # marker px -> plane metres
        H = np.asarray(K) @ np.c_[R[:, 0], R[:, 1], t] @ A
        warped = cv2.warpPerspective(m, H, size, flags=cv2.INTER_AREA).astype(np.float32)
        a = cv2.warpPerspective(np.full_like(m, 255), H, size, flags=cv2.INTER_AREA).astype(np.float32)[..., None]
        img = img * (1 - a / 255) + warped[..., None] * (a / 255)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_iphone_export(folder, uw_frames, K_uw, main_frames=None, K_main=None):
    """Minimal iPhUMI-like export folder (JSON + H.264 mp4s) for the loaders under test."""
    import av
    folder.mkdir(parents=True, exist_ok=True)
    name = folder.name
    js = {"ultrawideCameraIntrinsics": np.asarray(K_uw).tolist(),
          "mainCameraIntrinsics": np.asarray(K_main if K_main is not None else K_uw).tolist(),
          "ultrawideRGBTimes": ["t"] * len(uw_frames)}
    (folder / f"{name}.json").write_text(json.dumps(js))
    for suffix, frames in (("_ultrawidergb.mp4", uw_frames), ("_rgb.mp4", main_frames)):
        if frames is None:
            continue
        with av.open(str(folder / f"{name}{suffix}"), "w") as c:
            st = c.add_stream("libx264", rate=10)
            st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
            st.options = {"crf": "4"}
            for f in frames:
                for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="bgr24")):
                    c.mux(pkt)
            for pkt in st.encode():
                c.mux(pkt)
    return folder


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(0)
    h = 0.008
    sq = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    R_true, t_true = rot_z(1.0) @ rot_y(-2.5) @ rot_x(1.2), np.array([0.0012, -0.0009, 0.0013])
    assert rotation_angle_deg(R_true) == pytest.approx(3.0, abs=0.2) and np.linalg.norm(t_true) < 0.0021
    # noise-free on the finger-tag geometry: exact
    P = np.concatenate([sq @ R.T + t for R, t in TAG_POSES.values()])            # 8 tag corners
    R0, t0, rms0 = kabsch(P, P @ R_true.T + t_true)
    assert np.allclose(R0, R_true, atol=1e-12) and np.allclose(t0, t_true, atol=1e-12) and rms0 < 1e-12
    # 0.2 mm corner noise, 20 frames averaged, on tags spanning all three axes. (On the two nearly coplanar
    # finger tags the rotation about x is ill-conditioned: ~0.2 deg error. That is why align_ultrawide fits
    # the rotation on bearings and uses Kabsch only for the translation diagnostic.)
    poses = [(rot_x(150), [-0.03, 0.0, 0.09]), (rot_y(40) @ rot_x(180), [0.03, 0.0, 0.09]),
             (rot_x(120), [0.0, 0.035, 0.13])]
    P = np.concatenate([sq @ R.T + np.asarray(t) for R, t in poses])
    src = np.mean([P + rng.normal(0, 2e-4, P.shape) for _ in range(20)], axis=0)
    dst = np.mean([P @ R_true.T + t_true + rng.normal(0, 2e-4, P.shape) for _ in range(20)], axis=0)
    R, t, rms = kabsch(src, dst)
    assert rotation_angle_deg(R.T @ R_true) <= 0.1
    assert np.linalg.norm(t - t_true) <= 5e-4 and rms < 2e-4


def test_align_ultrawide_images(rig, tmp_path):
    K = rig.virtual["ultrawide"].reference.K
    iphone = render_tags(K)
    assert set(detect_tags(iphone, K)) == {0, 1}
    R_true = rot_y(2.0) @ rot_x(-0.7)
    phys = with_virtual_rotation(rig, "ultrawide", R_true).inverse_remap("ultrawide").apply(iphone)
    res = align_ultrawide_frames(rig, [iphone], K, [phys])
    assert rotation_angle_deg(res.R_phys_virt.T @ R_true) <= 0.2
    assert res.rms_px < 0.5 < res.rms_px_before
    # applying the result aligns the tags; the same through the export-folder loader (H.264 round trip)
    rig2 = with_virtual_rotation(rig, "ultrawide", res.R_phys_virt)
    again = align_ultrawide_frames(rig2, [iphone], K, [phys])
    assert again.angle_deg < 0.2
    ep = write_iphone_export(tmp_path / "x_grippercalibration_left", [iphone] * 3,
                             VirtualPinhole.from_K(K, (640, 480)).K_float32_rows())
    res2 = align_ultrawide(rig, ep, [phys])
    assert rotation_angle_deg(res2.R_phys_virt.T @ R_true) <= 0.2 and res2.n_frames == (3, 1)


def test_align_main_from_chain(rig):
    rig_r = with_virtual_rotation(rig, "ultrawide", rot_y(1.0) @ rot_x(0.5))
    R_iu_im = rot_z(0.3)
    R_main = align_main_from_chain(rig_r, R_iu_im)
    rig_r = with_virtual_rotation(rig_r, "main", R_main)
    # the virtual pair then has exactly the iPhone's relative rotation: R_vmain_vuw = R_iu_im^T
    assert np.allclose(rig_r.T("virtual_main", "virtual_ultrawide")[:3, :3], R_iu_im.T, atol=1e-12)


def test_rotation_from_image_pair():
    K_m = VirtualPinhole(960, 720, 703.1714, 703.1714, 476.37515, 361.360015).K
    K_u = VirtualPinhole(640, 480, 259.01538, 259.01538, 320.2935, 239.67455).K
    rng = np.random.default_rng(1)
    noise = cv2.GaussianBlur(rng.uniform(0, 255, (720, 960)).astype(np.float32), (0, 0), 1.5)
    main = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    R_true = rot_y(0.6) @ rot_x(-0.4)                     # d_uw = R d_main
    uw = cv2.warpPerspective(main, K_u @ R_true @ np.linalg.inv(K_m), (640, 480), flags=cv2.INTER_AREA)
    R, ratio, med = rotation_from_image_pair(main, uw, K_m, K_u)
    assert rotation_angle_deg(R.T @ R_true) < 0.05 and ratio > 0.5 and med < 1.0


def test_tcp_identity_case(rig):
    # R_phys_virt = I and T_physmain_TCP = D @ I_T_TCP(iPhone) -> the iPhone17,1 matrix
    T = tcp_pose_in_arkit_frame(rig, D_CV_ARKIT @ IPHONE_TCP)
    assert np.allclose(T, IPHONE_TCP, atol=1e-9, rtol=0)
    # general case equals the frame chain inv(T_main_virtualmain_arkit) @ T_main_TCP
    rig_r = with_virtual_rotation(rig, "main", rot_y(1.5) @ rot_z(-0.4))
    T_pm_tcp = make_T(rot_x(20), [0.01, 0.15, 0.2])
    expect = inv_T(rig_r.T("main", "virtual_main") @ D_CV_ARKIT) @ T_pm_tcp
    assert np.allclose(tcp_pose_in_arkit_frame(rig_r, T_pm_tcp), expect, atol=1e-12)


def test_calib_check_outputs(rig, tmp_path, capsys):
    K_uw = rig.virtual["ultrawide"].reference.K
    iphone_uw = render_tags(K_uw)
    rng = np.random.default_rng(2)
    iphone_main = cv2.GaussianBlur(rng.uniform(0, 255, (1440, 1920, 3)).astype(np.float32), (0, 0), 3)
    iphone_main = cv2.normalize(iphone_main, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    phys_main = rig.inverse_remap("main").apply(iphone_main)
    phys_uw = rig.inverse_remap("ultrawide").apply(iphone_uw)
    pm, pu = tmp_path / "main.png", tmp_path / "ultrawide.png"
    cv2.imwrite(str(pm), phys_main)
    cv2.imwrite(str(pu), phys_uw)
    ref = write_iphone_export(tmp_path / "ref_demonstration_left", [iphone_uw], K_uw, [iphone_main],
                              rig.virtual["main"].reference.K)
    out = tmp_path / "check"
    rc = calib_main(["calib-check", "--rig", "rig_default.yaml", "--images", str(pm), str(pu), "--out", str(out),
                     "--reference-iphone", str(ref)])
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    expected = {f"{c}_{k}.png" for c in ("main", "ultrawide")
                for k in ("physical", "virtual", "virtual_224", "blend", "edges")}
    assert set(report["files"]) == expected and all((out / f).is_file() for f in expected)
    assert report["coverage"] == rig.coverage_report()
    assert report["placeholder"] is True and report["rig_fingerprint"] == rig.fingerprint()
    assert report["tag_z_m"] == pytest.approx(0.098, abs=0.004)
    assert report["tag_sep_m"] == pytest.approx(0.056, abs=0.004)
    assert report["rms_vs_reference_px"] < 1.0
    assert cv2.imread(str(out / "ultrawide_virtual_224.png")).shape == (224, 224, 3)
    assert cv2.imread(str(out / "main_virtual.png")).shape == (1440, 1920, 3)
    # deploy purpose through the Python API, without reference
    panels = render_check(rig, {"main": phys_main}, purpose="deploy")
    assert set(panels) == {"main_physical", "main_virtual", "main_virtual_224"}
    assert panels["main_virtual"].shape == (720, 960, 3)
    rep = run_calib_check(rig, "images", tmp_path / "c2", images=[pm, pu], detect_tags=False)
    assert rep["tag_z_m"] is None


def test_calib_check_uncovered_is_tinted(rig, tmp_path):
    rig_r = with_virtual_rotation(rig, "ultrawide", rot_y(40.0))
    phys = np.full((1232, 1640, 3), 255, np.uint8)
    panels = render_check(rig_r, {"ultrawide": phys}, detect_tags=False)
    t = rig_r.remap("ultrawide", "export")
    assert t.coverage < 1.0
    v = panels["ultrawide_virtual"].astype(int)
    unc = ~t.valid
    assert (v[unc][:, 2] > v[unc][:, 0] + 50).mean() > 0.95           # red tint on uncovered pixels


def test_pad_resize_square():
    img = np.zeros((720, 960, 3), np.uint8)
    img[:] = 200
    out = pad_resize_square(img)
    assert out.shape == (224, 224, 3)
    assert out[0, 112, 0] == 0 and out[112, 112, 0] == 200        # 120 px black bands top and bottom
