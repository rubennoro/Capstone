"""``python -m franka_sim finger-kit <sub>``: inspect, pin, render and report the finger kit / finger model (A9d).

Why: the finger geometry decides grasp physics, the camera-to-TCP chain and the tag images. These commands make every
number checkable: ``check`` validates a kit directory, ``pin`` writes SOURCE.md and prints the sha to paste into the
tooling YAML, ``show`` renders the hand region with frame axes for a visual check of the frames, ``report`` writes the
machine-readable numbers that ``docs/finger_model.md`` embeds (generated, never typed).

Usage:
    python -m franka_sim finger-kit check DIR
    python -m franka_sim finger-kit pin DIR                      # prints ``kit_sha256: <hex>``
    python -m franka_sim finger-kit show   --profile sim-iphone [--set k=v ...] --out DIR    (RE; headless EGL)
    python -m franka_sim finger-kit report --profile sim-iphone [--set k=v ...] --out DIR    (RE)
  (station selection as ``python -m franka_sim model``: --profile | --ids | --deploy-config, --robot, --root, --set)

``show`` writes ``<view>_<gap>.png`` (views ``sidep|sidem|endp|endm`` = external free camera looking along +-TCP y /
+-TCP x, pitched 20 deg up from below the tips; ``main`` /
``uw`` = cam_main / cam_uw; gaps ``open`` = width_map.gap_max_m, ``mid`` = half, ``closed`` = 0) and ``show.txt``. Axes
(x red, y green, z blue; length by frame): flange 40 mm, TCP 30 mm, J 20 mm, M 15 mm, tag centres 10 mm.

``report`` writes ``finger_report.json`` and ``finger_report.md``. JSON keys (stable):
    schema ("finger_report/v1"), station {ids, fingerprint, robot, overrides}, mode, finger_kit (info["finger_kit"]
    dict or null), kit_variants (names or null), numbers (dict key -> {value, unit, status, source}), F_T_TCP (4x4),
    C_T_TCP (4x4), F_T_cam_main / F_T_cam_uw (4x4), markdown (the table text).
    numbers keys: mode, kit_id, kit_sha256, kit_variant, kit_path, adapter_length_m (kit L), extension_m (box only),
    dy_l_m, dy_r_m, yaw_deg, side_l, side_r, flange_to_tcp_m, flange_to_tip_m, pad_length_m, pad_offset_m,
    tag_separation_closed_m, tags_skipped, tcp_at, tcp_residual_x_mm, tcp_residual_y_mm, multiccd, mass_<geom>_kg,
    tooling_mass_kg, friction, solref, solimp, F_T_TCP_xyz_mm, F_T_TCP_rpy_deg, F_T_TCP_source, C_T_TCP_xyz_mm,
    C_T_TCP_rpy_deg, F_T_cam_main_xyz_mm, F_T_cam_main_rpy_deg, F_T_cam_uw_xyz_mm, F_T_cam_uw_rpy_deg.
    (keys that do not apply to the mode are absent). status is the station provenance status of the config key the
    value comes from (placeholder / decision / measured / ...), or "derived (model)" when the model computes it,
    "derived (kit)" for numbers read from the kit; source is the provenance source text.
    flange_to_tip_m: distal pad edge (kit: model value; box: tcp + pad_length/2; stock: tcp + pad_length/2).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from franka_sim import finger_kit as FK

SCHEMA = "finger_report/v1"
AXIS_LEN_M = {"flange": 0.04, "tcp": 0.03, "J": 0.02, "M": 0.015, "tag": 0.01}
GAPS = ("open", "mid", "closed")
VIEW_PITCH_DEG = -20.0       # camera below the tips looking slightly up (the arm and phone housing are above/behind)
VIEWS = {"sidep": (1, +1.0), "sidem": (1, -1.0), "endp": (0, +1.0), "endm": (0, -1.0)}   # axis of the TCP frame, sign


# --------------------------------------------------------------------------------------------------- check / pin


def _fmt(a, nd=3) -> str:
    return "[" + ", ".join(f"{float(x):.{nd}f}" for x in np.ravel(a)) + "]"


def cmd_check(args) -> int:
    try:
        kit = FK.load_kit(args.dir)
        digest = FK.kit_digest(args.dir)
    except (FK.KitError, OSError) as e:
        print(f"finger-kit check FAILED: {e}", file=sys.stderr)
        return 1
    print(f"kit {kit.id}  units={kit.units}  path={kit.path}")
    print("parts:")
    for n, p in kit.parts.items():
        lo, hi = p.vertices.min(0) * 1e3, p.vertices.max(0) * 1e3
        print(f"  {n:16s} role={p.role:9s} material={p.material} vertices={len(p.vertices)} faces={len(p.faces)} "
              f"bbox_mm={_fmt(lo, 2)}..{_fmt(hi, 2)} volume_cm3={p.volume_m3 * 1e6:.3f} "
              f"mass_g={p.mass_kg * 1e3:.2f} ({p.mass_source}) collision={'hull' if p.collision == 'hull' else len(p.collision)}")
    print("sides:")
    for n, s in kit.sides.items():
        print(f"  {n}: parts={list(s.parts)} contact_part={s.contact_part} camera_face={s.camera_face:+.0f}")
        print(f"     kit_T_M rows: " + "; ".join(_fmt(r, 4) for r in s.kit_T_M))
        print(f"     tip_mm={_fmt(s.tip * 1e3, 2)}")
        for t in s.tags:
            on = getattr(t, "on", None)
            print(f"     tag id={t.id} size_mm={t.size_m * 1e3:.1f} centre_mm={_fmt(t.centre * 1e3, 2)} "
                  f"normal={_fmt(t.normal)} up={_fmt(t.up)}" + (f" on={on}" if on else ""))
    variants = getattr(kit, "variants", None)
    if variants:
        print("variants: " + "; ".join(f"{k}: " + ", ".join(f"{sd}={list(v)}" for sd, v in d.items())
                                       for k, d in variants.items()))
    print(f"contact: {kit.contact}")
    print(f"digest: {digest}")
    return 0


def cmd_pin(args) -> int:
    try:
        p = FK.write_source_md(args.dir)
        digest = FK.kit_digest(args.dir)
    except (FK.KitError, OSError) as e:
        print(f"finger-kit pin FAILED: {e}", file=sys.stderr)
        return 1
    print(f"wrote {p}")
    print(f"kit_sha256: {digest}")
    return 0


# --------------------------------------------------------------------------------------------------- station / model


def _build(args):
    from franka_sim import model as M

    st = M.station_from_args(args)
    return st, M.build_model(st, with_scene=False), M


def _writepng(path: Path, rgb_or_bgr: np.ndarray, bgr: bool) -> None:
    img = np.ascontiguousarray(rgb_or_bgr[:, :, ::-1] if not bgr else rgb_or_bgr)     # -> BGR
    try:
        import cv2

        ok = cv2.imwrite(str(path), img)
        if not ok:
            raise OSError(f"cv2.imwrite failed for {path}")
    except ImportError:
        from PIL import Image

        Image.fromarray(img[:, :, ::-1]).save(path)


# --------------------------------------------------------------------------------------------------- show


def _frames(sm, M, d) -> list:
    """[(label, kind, 4x4 world pose)] of the frames to draw (axes)."""
    m = sm.model
    out = [("flange", "flange", M.xform_of(d, "site", M._id(m, "site", "flange"))),
           ("tcp", "tcp", M.xform_of(d, "site", M._id(m, "site", "tcp")))]
    fk = sm.info.get("finger_kit")
    for s in ("l", "r"):
        fb = sm.names["bodies"][f"finger_{s}"]
        W_b = M.xform_of(d, "body", M._id(m, "body", fb))
        y_c, z_b = M.jaw_mount(m, fb)
        J = FK.jaw_T(y_c, z_b)
        out.append((f"J_{s}", "J", W_b @ J))
        if fk:
            out.append((f"M_{s}", "M", W_b @ J @ FK.J_T_M(fk["L"], fk["dy"][s], fk["yaw"])))
        for gname in (f"tag_{s}",):
            try:
                gi = M._id(m, "geom", gname)
            except KeyError:
                continue
            T = np.eye(4)
            T[:3, :3] = d.geom_xmat[gi].reshape(3, 3)
            T[:3, 3] = d.geom_xpos[gi]
            out.append((gname, "tag", T))
    return out


def _draw_axes(scene, frames) -> None:
    import mujoco

    cols = ([1, 0, 0, 1], [0, 0.8, 0, 1], [0.1, 0.3, 1, 1])
    for _, kind, T in frames:
        L = AXIS_LEN_M[kind]
        for k in range(3):
            if scene.ngeom >= scene.maxgeom:
                return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                np.array(cols[k], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0012, T[:3, 3], T[:3, 3] + L * T[:3, k])
            scene.ngeom += 1


def _free_camera(T_tcp: np.ndarray, view: str):
    import mujoco

    x, y, z = T_tcp[:3, 0], T_tcp[:3, 1], T_tcp[:3, 2]
    a = np.deg2rad(VIEW_PITCH_DEG)
    ax, sg = VIEWS[view]
    f = sg * (x, y)[ax] * np.cos(a) + z * np.sin(a)             # forward: along +-x / +-y of the TCP frame, pitched
    f = f / np.linalg.norm(f)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = T_tcp[:3, 3] - 0.03 * z
    cam.distance = 0.22
    cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
    cam.elevation = float(np.degrees(np.arcsin(np.clip(f[2], -1, 1))))
    return cam


def cmd_show(args) -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from franka_sim.render import Renderer

    st, sm, M = _build(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gmax = float(st.get("tooling.width_map.gap_max_m"))
    gaps = {"open": gmax, "mid": gmax / 2, "closed": 0.0}
    W, H = 480, 360
    sizes = {}
    for s in ("main", "ultrawide"):
        w0 = int(st.get(f"sensor.streams.{s}.width_px"))
        h0 = int(st.get(f"sensor.streams.{s}.height_px"))
        sizes[s] = (W, int(round(h0 * W / w0)))
    lines = [f"finger-kit show: robot={sm.robot} mode={sm.info.get('finger_mode')} tcp_at={sm.info.get('tcp_at')} "
             f"fingerprint={sm.station_fingerprint}",
             "axes: x red, y green, z blue; lengths flange 40 mm, TCP 30 mm, J 20 mm, M 15 mm, tag centre 10 mm",
             "views: sidep/sidem/endp/endm = external free camera at the hand (forward along +-TCP y / +-TCP x, "
             "pitched 20 deg up), main/uw = cam_main/cam_uw (no axes); gaps open/mid/closed = "
             + ", ".join(f"{k} {v * 1e3:.1f} mm" for k, v in gaps.items())]
    ext = mujoco.Renderer(sm.model, H, W)
    rnd = Renderer(sm, st, sizes=sizes)
    written = []
    try:
        for gn, g in gaps.items():
            d = M.forward_at(sm, None, g)
            frames = _frames(sm, M, d)
            T_tcp = dict((k, T) for k, _, T in frames)["tcp"]
            for view in VIEWS:
                ext.update_scene(d, camera=_free_camera(T_tcp, view))
                _draw_axes(ext.scene, frames)
                _writepng(out / f"{view}_{gn}.png", ext.render(), bgr=False)
                written.append(f"{view}_{gn}.png")
            imgs = rnd.render(d.qpos.copy())
            for s, nm in (("main", "main"), ("ultrawide", "uw")):
                _writepng(out / f"{nm}_{gn}.png", imgs[s], bgr=True)
                written.append(f"{nm}_{gn}.png")
            if gn == "closed":
                for label, kind, T in frames:
                    lines.append(f"frame {label:7s} world xyz_mm={_fmt(T[:3, 3] * 1e3, 2)} (gap closed)")
    finally:
        ext.close()
        rnd.close()
    lines.append("files: " + " ".join(written))
    (out / "show.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {len(written)} PNGs + show.txt to {out}")
    return 0


# --------------------------------------------------------------------------------------------------- report

DERIVED = "derived (model)"


def _prov(st, path: str):
    try:
        p = st.prov(path)
        return p.status, p.source
    except Exception:  # noqa: BLE001  (StationError: no provenance for this key)
        return "unspecified", ""


def _rpy(M, T):
    return [round(v, 4) for v in M._rpy_deg(np.asarray(T)[:3, :3])]


def build_report(st, sm, M) -> dict:
    info = sm.info
    fa = st.get("tooling.finger_adapter")
    mode = info["finger_mode"]
    fk = info.get("finger_kit")
    nums: dict = {}

    def add(key, value, unit, status, source=""):
        nums[key] = {"value": value, "unit": unit, "status": status, "source": source}

    def cfg(key, value, unit, path):
        add(key, value, unit, *_prov(st, path))

    FAP = "tooling.finger_adapter."
    cfg("mode", mode, "", FAP + "mode")
    kit = None
    if fk:
        kit = FK.load_kit(fk["path"])
        cfg("kit_id", fk["id"], "", FAP + "kit")
        cfg("kit_sha256", fk["sha"], "", FAP + "kit_sha256")
        cfg("kit_variant", fk.get("variant", "full"), "", FAP + "kit_variant")
        add("kit_path", fk["path"], "", "decision", "tooling.finger_adapter.kit")
        cfg("adapter_length_m", fk["L"], "m", FAP + "adapter_length_m")
        for s in ("l", "r"):
            if fk.get("dy_source") == "auto":
                add(f"dy_{s}_m", fk["dy"][s], "m", DERIVED, "auto_dy: contact part inner face at finger-body y = 0 at gap 0")
            else:
                cfg(f"dy_{s}_m", fk["dy"][s], "m", FAP + "lateral_offset_m")
            add(f"side_{s}", fk["side_for_finger"][s], "", DERIVED, "kit side whose camera_face matches the camera direction")
        cfg("yaw_deg", fk["yaw"], "deg", FAP + "adapter_yaw_deg")
        gs = kit.provenance.get("geometry", {})
        ksrc = f"kit {kit.id} geometry: {gs.get('status', 'unspecified')}: {gs.get('source', '')}"
        add("tag_separation_closed_m", fk["tag_separation_closed_m"], "m", "derived (kit)", ksrc)
        add("pad_offset_m", fk["pad_offset_m"], "m", "derived (kit)", ksrc + " (contact parts' inner faces at gap 0)")
        add("tags_skipped", list(fk.get("tags_skipped", [])), "", DERIVED, "variant: tags whose carrier part is absent")
        msrc = "kit " + kit.id + " part masses: " + ", ".join(f"{n}={p.mass_source}" for n, p in kit.parts.items())
        for g, v in fk["masses_kg"].items():
            add(f"mass_{g}_kg", v, "kg", "placeholder" if g.startswith("finger_adapter") else "derived (kit)",
                "model.ADAPTER_DENSITY_KG_M3 x block volume" if g.startswith("finger_adapter") else msrc)
        add("tcp_residual_x_mm", fk["tcp_residual_mm"][0], "mm", DERIVED, "tip midpoint off the hand centre line")
        add("tcp_residual_y_mm", fk["tcp_residual_mm"][1], "mm", DERIVED, "tip midpoint off the hand centre line")
        add("pad_length_m", fk["pad_length_m"], "m", "derived (kit)", ksrc + " (contact part z extent)")
        fric = list(kit.contact.get("friction", []))
        add("friction", fric, "", "derived (kit)", f"kit {kit.id} contact.friction")
    elif mode == "stock":
        add("pad_length_m", info["pad_length_m"], "m", "datasheet", "menagerie fingertip_pad_collision_1 z extent")
        add("pad_offset_m", info["pad_offset_m"], "m", "datasheet", info.get("pad_offset_source", ""))
        sp = info["stock_pads"]["tcp_residual_mm"]
        add("tcp_residual_x_mm", sp[0], "mm", DERIVED, "stock pad midpoint off the hand centre line")
        add("tcp_residual_y_mm", sp[1], "mm", DERIVED, "stock pad midpoint off the hand centre line")
        add("friction", None, "", "datasheet", "menagerie stock pad friction (unchanged)")
    else:
        cfg("extension_m", float(fa["extension_m"]), "m", FAP + "extension_m")
        add("pad_length_m", info["pad_length_m"], "m", *_prov(st, FAP + "pad_size_m"))
        add("pad_offset_m", info["pad_offset_m"], "m", _prov(st, FAP + "pad_offset_m")[0],
            info.get("pad_offset_source", ""))
        add("tcp_residual_x_mm", 0.0, "mm", DERIVED, "box TCP on the centre line by construction")
        add("tcp_residual_y_mm", 0.0, "mm", DERIVED, "box TCP on the centre line by construction")
        add("friction", [float(fa.get("friction", 1.0))], "", *_prov(st, FAP + "friction"))
    add("tcp_at", info["tcp_at"], "", DERIVED, "model")
    if "multiccd" in info:
        add("multiccd", info["multiccd"], "", DERIVED, "model: kit mode only")
    gc = M.grip_contact(M.sim_config(st, None))
    add("solref", gc["solref"], "", "decision", "run/sim grip_solref (model.grip_contact)")
    add("solimp", gc["solimp"], "", "decision", "run/sim grip_solimp (model.grip_contact)")
    pay = info["payload"]
    add("tooling_mass_kg", pay["tooling_mass_kg"], "kg", DERIVED, "sum of the tooling geom masses")

    fr = M.export_frames(sm)
    T_tcp, T_m, T_uw = (np.array(fr[k]) for k in ("F_T_TCP", "F_T_cam_main", "F_T_cam_uw"))
    pl = float(info["pad_length_m"])
    tip = fk["flange_to_tip_m"] if fk else float(T_tcp[2, 3]) + pl / 2
    add("flange_to_tcp_m", float(T_tcp[2, 3]), "m", DERIVED, "F_T_TCP z (flange frame, along the approach axis)")
    add("flange_to_tip_m", tip, "m", DERIVED, "distal pad edge on the centre line" if not fk else "kit tip midpoint")
    if mode != "stock":
        if not fk:
            d = M.forward_at(sm, None, 0.0)
            tl, tr = (d.geom_xpos[M._id(sm.model, "geom", f"tag_{s}")] for s in ("l", "r"))
            add("tag_separation_closed_m", float(np.linalg.norm(tl - tr)), "m", DERIVED, "tag box centres at gap 0")
    F_src = info.get("F_T_TCP_source", "derived")
    fs = (DERIVED, "TCP from the finger geometry") if F_src == "derived" else _prov(st, "tooling.F_T_TCP")
    add("F_T_TCP_xyz_mm", (T_tcp[:3, 3] * 1e3).round(3).tolist(), "mm", *fs)
    add("F_T_TCP_rpy_deg", _rpy(M, T_tcp), "deg", *fs)
    add("F_T_TCP_source", F_src, "", DERIVED, "model")
    cs = _prov(st, "tooling.C_T_TCP")
    C = np.array(st.get("tooling.C_T_TCP"), float)
    add("C_T_TCP_xyz_mm", (C[:3, 3] * 1e3).round(3).tolist(), "mm", *cs)
    add("C_T_TCP_rpy_deg", _rpy(M, C), "deg", *cs)
    cam_src = info.get("F_T_C_main_source", "")
    cam_status = (DERIVED, "F_T_TCP @ inv(C_T_TCP)") if cam_src.startswith("derived") else _prov(st, "tooling.F_T_C_main")
    for nm, T in (("main", T_m), ("uw", T_uw)):
        add(f"F_T_cam_{nm}_xyz_mm", (T[:3, 3] * 1e3).round(3).tolist(), "mm", *cam_status)
        add(f"F_T_cam_{nm}_rpy_deg", _rpy(M, T), "deg", *cam_status)

    rep = {"schema": SCHEMA,
           "station": {"ids": dict(st.ids), "fingerprint": sm.station_fingerprint, "robot": sm.robot,
                       "overrides": list(st.overrides)},
           "mode": mode, "finger_kit": fk, "kit_variants": sorted(getattr(kit, "variants", {}) or {}) if kit else None,
           "numbers": nums, "F_T_TCP": T_tcp.tolist(), "C_T_TCP": C.tolist(),
           "F_T_cam_main": T_m.tolist(), "F_T_cam_uw": T_uw.tolist()}
    rep["markdown"] = to_markdown(rep)
    return rep


def _val(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    return "" if v is None else str(v)


def to_markdown(rep: dict) -> str:
    st = rep["station"]
    lines = [f"Finger report: robot `{st['robot']}`, tooling `{st['ids'].get('tooling')}`, mode `{rep['mode']}`, "
             f"station fingerprint `{st['fingerprint']}`" + (f", overrides {st['overrides']}" if st["overrides"] else ""),
             "", "| number | value | unit | status | source |", "|---|---|---|---|---|"]
    for k, e in rep["numbers"].items():
        src = str(e["source"]).replace("|", "\\|")
        lines.append(f"| {k} | {_val(e['value'])} | {e['unit']} | {e['status']} | {src} |")
    return "\n".join(lines) + "\n"


def cmd_report(args) -> int:
    st, sm, M = _build(args)
    rep = build_report(st, sm, M)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "finger_report.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / "finger_report.md").write_text(rep["markdown"])
    print(rep["markdown"])
    print(f"wrote {out / 'finger_report.json'} and {out / 'finger_report.md'}")
    return 0


# --------------------------------------------------------------------------------------------------- CLI


def _station_args(p: argparse.ArgumentParser) -> None:
    from franka_deploy.station import PROFILES, STATION_ROOT

    p.add_argument("--deploy-config")
    p.add_argument("--profile", choices=sorted(PROFILES))
    p.add_argument("--ids", help="robot=panda,tooling=...,sensor=...,scene=...")
    p.add_argument("--robot", choices=["panda", "fr3"])
    p.add_argument("--root", default=str(STATION_ROOT))
    p.add_argument("--set", action="append", default=[], metavar="group.path=value")
    p.add_argument("--out", required=True, help="output directory")


def add_arguments(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="sub", metavar="<sub>")
    c = sub.add_parser("check", help="load + validate a kit directory; print parts, sides, digest")
    c.add_argument("dir")
    c.set_defaults(_sub=cmd_check)
    c = sub.add_parser("pin", help="(re)write SOURCE.md and print the kit_sha256 line for the tooling YAML")
    c.add_argument("dir")
    c.set_defaults(_sub=cmd_pin)
    c = sub.add_parser("show", help="render the hand at gaps open/mid/closed with frame axes (RE)")
    _station_args(c)
    c.set_defaults(_sub=cmd_show)
    c = sub.add_parser("report", help="write finger_report.json + finger_report.md for the built station (RE)")
    _station_args(c)
    c.set_defaults(_sub=cmd_report)


def run(args: argparse.Namespace) -> int:
    fn = getattr(args, "_sub", None)
    if fn is None:
        print("usage: python -m franka_sim finger-kit {check,pin,show,report} ...", file=sys.stderr)
        return 2
    return int(fn(args) or 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_sim finger-kit", description=__doc__.split("\n\n")[0])
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
