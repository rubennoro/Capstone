"""Dimensioned drawing of the finger adapter (E3), from the same solids adapter.py exports: part sections in the carrier
frame K (right side) and the assembly on the Franka Hand (hand frame, gap 0). Called by adapter.py (make_drawing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MENAGERIE = Path(__file__).resolve().parents[2] / "controlSW/franka_sim/franka_sim/assets/franka_emika_panda/assets"


def _tm(solid, tol=0.02):
    import trimesh

    v, f = solid.tessellate(tol, 0.2)
    return trimesh.Trimesh(np.array([[p.x, p.y, p.z] for p in v]), np.array(f, int), process=True)


def _sec(ax, mesh, normal, origin, dims, **kw):
    import trimesh

    s = trimesh.intersections.mesh_plane(mesh, normal, origin)
    for e in s:
        ax.plot(e[:, dims[0]], e[:, dims[1]], solid_capstyle="round", **kw)


def _dim(ax, p0, p1, text, off=(0, 0), color="#b00020", fs=7, side=0.5):
    """Dimension line p0 -> p1 shifted by off, with extension lines and the label at the middle."""
    p0, p1, off = np.asarray(p0, float), np.asarray(p1, float), np.asarray(off, float)
    a, b = p0 + off, p1 + off
    if np.linalg.norm(off) > 0:
        for p, q in ((p0, a), (p1, b)):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.4)
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<|-|>", color=color, lw=0.6, mutation_scale=6,
                                                    shrinkA=0, shrinkB=0))
    m = a + side * (b - a)
    vert = abs(b[1] - a[1]) > abs(b[0] - a[0])
    ax.text(m[0], m[1], text, color=color, fontsize=fs, ha="center", va="center", rotation=90 if vert else 0,
            bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.9))


def make_drawing(parts_K: dict, D: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    from adapter import DS, FR, HO, K_to_hand, holder_parts_K, to_hand

    A = _tm(parts_K["right"])
    holder, finger = holder_parts_K("right")
    k = dict(color="k", lw=0.9)
    kh = dict(color="#2f63c4", lw=0.6)
    kt = dict(color="#888888", lw=0.6, ls="--")
    xb, xk, xp = D["back_x_K"], D["keep_x_K"], D["plate_under_x_K"]
    zt, zb, zl, zp0, ztop = D["tongue_top_K"], D["clamp_bottom_K"], D["lip_bottom_K"], D["pad_z0_K"], D["part_top_K"]
    yo = FR.tongue_thickness + DS.outer_wall
    hw = FR.tongue_width / 2

    fig = plt.figure(figsize=(17, 11.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    fig.suptitle("Franka Hand finger adapter (E3), right side (tag 1); left = mirror image.  mm.  Frame K: origin = "
                 "midpoint of the M4 holes on the tongue inner face, x = camera side, y = outward, z = hand axis",
                 fontsize=10)

    # (a) section x = 8 (through the pad, the M3 pilots and the slot edge): y-z
    ax = fig.add_subplot(gs[0, 0])
    _sec(ax, A, (1, 0, 0), (8.0, 0, 0), (1, 2), **k)
    _sec(ax, holder, (1, 0, 0), (8.0, 0, 0), (1, 2), **kh)
    ax.add_patch(plt.Rectangle((0, zl - 3), FR.tongue_thickness, zt - zl + 3, fc="#dddddd", ec="#888888", lw=0.5))
    for (y, z) in D["holes_K"]["right"]:
        ax.add_patch(plt.Circle((y, z), DS.m3_pilot_d / 2, fill=False, ec="#b00020", lw=0.6))
    h = D["holes_K"]["right"]
    _dim(ax, (h[0][0], h[0][1]), (h[1][0], h[1][1]), "10.0", off=(0, -4.5))
    _dim(ax, (h[1][0], h[1][1]), (h[3][0], h[3][1]), "15.0", off=(6.5, 0))
    _dim(ax, (DS.inner_y, zl), (DS.inner_y, ztop), f"{ztop - zl:.1f}", off=(-4, 0))
    _dim(ax, (DS.inner_y, ztop), (DS.pad_y_max, ztop), f"{DS.pad_y_max - DS.inner_y:.1f}", off=(0, 3))
    _dim(ax, (0, zt), (0, 0), f"{zt:.1f}", off=(-2.5, 0))
    ax.text(3, zt - 6, "tongue", fontsize=6, ha="center", color="#555555", rotation=90)
    ax.text(10, zp0 + 1, "pad", fontsize=6, color="k")
    ax.text(-5, D["z_rack_K"] + 2.6, "holder rack", fontsize=6, color="#2f63c4")
    ax.set_title("(a) section x = 8: pad, 4x M3 pilots 2.5 (MGN9C 15 x 10), holder (blue)", fontsize=8)

    # (b) section x = 0 (centre of the tongue): y-z, with the holder and the pocket datum
    ax = fig.add_subplot(gs[0, 1])
    _sec(ax, A, (1, 0, 0), (0.0, 0, 0), (1, 2), **k)
    _sec(ax, holder, (1, 0, 0), (0.0, 0, 0), (1, 2), **kh)
    ax.add_patch(plt.Rectangle((0, zl - 3), FR.tongue_thickness, zt - zl + 3, fc="#dddddd", ec="#888888", lw=0.5))
    ax.axhline(FR.housing_max_z - FR.hole_z, color="#888888", lw=0.5, ls=":")
    ax.text(-12, FR.housing_max_z - FR.hole_z - 1.8, "housing top (66.0)", fontsize=6, color="#555555")
    _dim(ax, (yo, zl), (yo, zb), f"lip {zb - zl:.1f}", off=(3, 0), fs=6)
    _dim(ax, (FR.tongue_thickness, 0), (yo, 0), f"{DS.outer_wall:.1f}", off=(0, -9), fs=6)
    _dim(ax, (-DS.tongue_gap_inner, zt), (FR.tongue_thickness, zt), f"{FR.tongue_thickness + DS.tongue_gap_inner:.1f}",
         off=(0, 3), fs=6)
    _dim(ax, (2, zt), (2, D["z_rack_K"]), f"{DS.rack_clearance:.1f}", off=(14, 0), fs=6)
    ax.text(1.0, zt + 0.4, "ceiling = datum on tongue top", fontsize=5.5)
    ax.set_title("(b) section x = 0: pocket on the tongue, screw wall + lip", fontsize=8)

    # (c) section y = 9 (through the screw wall): x-z, slots
    ax = fig.add_subplot(gs[0, 2])
    _sec(ax, A, (0, 1, 0), (0, 9.0, 0), (0, 2), **k)
    _sec(ax, holder, (0, 1, 0), (0, 9.0, 0), (0, 2), **kh)
    _dim(ax, (-FR.hole_pitch / 2, 0), (FR.hole_pitch / 2, 0), f"{FR.hole_pitch:.0f}", off=(0, -9.5))
    _dim(ax, (FR.hole_pitch / 2 + 2.25, -DS.slot_travel), (FR.hole_pitch / 2 + 2.25, DS.slot_travel),
         f"slot +-{DS.slot_travel:.0f}", off=(4, 0), fs=6)
    _dim(ax, (xb, ztop), (-xb, ztop), f"{-2 * xb:.1f}", off=(0, 3))
    _dim(ax, (xb, ztop), (xp, ztop), f"{xp - xb:.1f}", off=(0, 7))
    _dim(ax, (xk, zt), (xk, zp0), f"{zp0 - zt:.1f}", off=(-3, 0), fs=6)
    ax.text(-hw, -12, f"M4 slots {DS.m4_clear_d} wide, centres 0 +-{DS.slot_travel:.0f}", fontsize=6)
    ax.text(xp - 0.5, zp0 + 8, "pad face\n(holder plate)", fontsize=6, ha="right")
    ax.set_title("(c) section y = 9 (screw wall): M4 slots, rack notch, pad", fontsize=8)

    # (d) section z = 3 (through the clamp): x-y
    ax = fig.add_subplot(gs[0, 3])
    _sec(ax, A, (0, 0, 1), (0, 0, 3.0), (0, 1), **k)
    ax.add_patch(plt.Rectangle((-hw, 0), 2 * hw, FR.tongue_thickness, fc="#dddddd", ec="#888888", lw=0.5))
    _dim(ax, (-hw - DS.tongue_gap_x, -DS.tongue_gap_inner), (hw + DS.tongue_gap_x, -DS.tongue_gap_inner),
         f"{FR.tongue_width + 2 * DS.tongue_gap_x:.1f}", off=(0, -3), fs=6)
    _dim(ax, (xb, DS.inner_y), (xb, yo), f"{yo - DS.inner_y:.1f}", off=(-3, 0), fs=6)
    _dim(ax, (xb, DS.inner_y), (-xb, DS.inner_y), f"{-2 * xb:.1f}", off=(0, -3), fs=6)
    ax.text(0, 3, "tongue 21 x 6", fontsize=6, ha="center", color="#555555")
    ax.set_title("(d) section z = 3: pocket 21.3 x 6.3 (outer face = datum)", fontsize=8)

    for a in fig.axes:
        a.set_aspect("equal")
        a.grid(True, lw=0.2)
        a.tick_params(labelsize=6)

    # (e) assembly in the hand frame at gap 0, section x = 0 (y-z), both sides
    ax = fig.add_subplot(gs[1, 0:2])
    for i in range(5):
        m = trimesh.load(MENAGERIE / f"hand_{i}.obj", force="mesh")
        m.apply_scale(1000.0)
        _sec(ax, m, (1, 0, 0), (0.0, 0, 0), (1, 2), color="#999999", lw=0.5)
    for side in ("left", "right"):
        Ah = _tm(to_hand(parts_K[side], side))
        hold, fing = holder_parts_K(side)
        T = K_to_hand(side)
        for mm, kw in ((Ah, k), (hold.copy().apply_transform(T), kh),
                       (fing.copy().apply_transform(T), dict(color="#1d7a46", lw=0.6))):
            _sec(ax, mm, (1, 0, 0), (0.0, 0, 0), (1, 2), **kw)
        sy = 1 if side == "right" else -1
        ax.add_patch(plt.Rectangle((min(sy * FR.y_inner, sy * (FR.y_inner + 6)), FR.slot_floor_z),
                                   6, FR.hole_z + FR.hole_below_top - FR.slot_floor_z, fc="#dddddd", ec="#888888",
                                   lw=0.5))
    tipz = D["tip_hand_z"]
    _dim(ax, (-70, 0), (-70, tipz), f"fingertip point {tipz:.2f} from the flange (D7)", off=(0, 0), side=0.55)
    _dim(ax, (60, 0), (60, FR.hole_z + FR.hole_below_top), f"tongue top {FR.hole_z + FR.hole_below_top:.1f}",
         fs=6, side=0.6)
    _dim(ax, (75, 0), (75, D["holder_back_hand_z"]), f"holder back end {D['holder_back_hand_z']:.1f}", fs=6, side=0.45)
    _dim(ax, (-FR.y_inner, 50), (FR.y_inner, 50), "25.0 (closed)", fs=6)
    ax.plot([0], [tipz], "r+", ms=10)
    tz = D["holder_back_hand_z"] + HO.tag_centre[1] - HO.back_y
    ax.text(-60, tz, f"tags: hand z {tz:.1f}, on the camera face (x = {HO.tag_centre[2] - HO.finger_mid_z:.1f})",
            fontsize=6, color="#2f63c4")
    ax.axhline(0, color="k", lw=0.8)
    ax.text(-95, 2, "flange (hand base)", fontsize=6)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-35, 250)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.2)
    ax.tick_params(labelsize=6)
    ax.set_title("(e) on the Franka Hand, gap 0, section x = 0 (hand frame y-z): adapters (black), holders (blue), "
                 "soft fingers (green), tongues (grey)", fontsize=8)

    # (f) camera-side view of the assembly (hand x-z at y = 12 through the right side)
    ax = fig.add_subplot(gs[1, 2])
    Ah = _tm(to_hand(parts_K["right"], "right"))
    hold, fing = holder_parts_K("right")
    T = K_to_hand("right")
    yc = FR.y_inner + 3.0
    for mm, kw in ((Ah, k), (hold.copy().apply_transform(T), kh),
                   (fing.copy().apply_transform(T), dict(color="#1d7a46", lw=0.6))):
        _sec(ax, mm, (0, 1, 0), (0, yc, 0), (0, 2), **kw)
    for i in range(5):
        m = trimesh.load(MENAGERIE / f"hand_{i}.obj", force="mesh")
        m.apply_scale(1000.0)
        _sec(ax, m, (0, 1, 0), (0, yc, 0), (0, 2), color="#999999", lw=0.5)
    ax.add_patch(plt.Rectangle((-hw, FR.slot_floor_z), 2 * hw, FR.hole_z + FR.hole_below_top - FR.slot_floor_z,
                               fc="#dddddd", ec="#888888", lw=0.5))
    ax.set_xlim(-30, 30)
    ax.set_ylim(50, 130)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.2)
    ax.tick_params(labelsize=6)
    xm = HO.plate_top_z - HO.finger_mid_z
    _dim(ax, (0, 110), (xm, 110), f"{xm:.1f}", off=(0, 0), fs=6)
    ax.set_title(f"(f) section y = {yc:.1f} (hand x-z): camera on +x", fontsize=8)

    # (g) notes
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    tag_d0 = sum(HO.tag_centre[0] - HO.tip_x[s] for s in ("left", "right"))
    notes = [
        "Interface (Franka R50010 v1.2 Fig. 5.4/5.5):",
        "  tongue 6.00 x 21, 2x M4, pitch 12, 7.0 below its top,",
        "  top 11.40 above the housing; inner faces 25 / 105 apart",
        "Fasteners per side:",
        "  2x M4x12 ISO 4762 + DIN 125 washers, 1.2 Nm (printed)",
        "  4x M3x16 ISO 4762 into 2.5 pilots, 13 deep",
        f"Part: {ztop - zl:.1f} tall, {-2 * xb:.1f} x {DS.pad_y_max - DS.inner_y:.1f} footprint",
        f"Holder back end (rack face) at hand z {D['holder_back_hand_z']:.1f}",
        f"   = tongue top + {DS.rack_clearance:.1f}",
        f"Fingertip point (TCP, D7): hand z {tipz:.2f}",
        f"   = {tipz - FR.hole_z:.2f} beyond the M4 holes,",
        f"   12.5 inward of the tongue inner face",
        f"Tag centre distance (CAD) = gap + {tag_d0:.2f} (D8)",
        f"   real stickers: gap + {tag_d0 + 2 * HO.tag_outward_real:.2f} (D8: width = gap)",
        "Print: PETG (or PLA), lying on the back face (x min),",
        "   no supports; 0.2 layers, 5 walls, 60 % infill",
        "Clearances: housing 1.0 (lip in the slot: 1.0 to the",
        "   sides, 5.4 to the floor, 2.5 to the slot end at open),",
        "   rack 0.1 / 1.0, finger pocket 0.4, adapters 8.0 apart",
    ]
    ax.text(0, 1, "\n".join(notes), va="top", fontsize=7.5, family="monospace")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.mkdir(exist_ok=True)
    fig.savefig(out / "drawing.png", dpi=170)
    fig.savefig(out / "drawing.svg")
    plt.close(fig)
