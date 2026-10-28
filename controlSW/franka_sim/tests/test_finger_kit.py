"""A9a finger kit: loader validation, digest, STL parsing (ASCII + binary), place/auto_dy math on the box kit, kit_T_M
handling (rigid re-posing is invisible after placement, reflections are rejected), and the box-mode mapping."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import yaml

from franka_sim import finger_kit as fk
from franka_sim.model import PAD_OFFSET_M, TAG_SIZE_M, TAG_Y_EXTRA_M, TAG_Z_FROM_PAD_M, Rz, make_T

PAD = (0.02, 0.008, 0.03)
J = fk.jaw_T(0.0144, 0.0)            # finger_0: y_c 14.4 mm, base z 0 (A9.md)
TIP_Z, EXT = 0.0539, 0.1


def _edit(d, fn):
    p = d / "kit.yaml"
    doc = yaml.safe_load(p.read_text())
    fn(doc)
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    (d / "SOURCE.md").unlink(missing_ok=True)


@pytest.fixture
def box(tmp_path):
    return fk.make_box_kit(tmp_path / "box")


def _box_mode(side_sign):
    """model.py step 3 in the finger body frame (box mode): pad centre, white tag box centre."""
    px, pt, pl = PAD
    b = PAD_OFFSET_M
    pad_c = np.array([0.0, b / 2 + pt / 2, TIP_Z + EXT - pl / 2])
    tag_c = np.array([side_sign * px / 2 + side_sign * 0.0005, b / 2 + TAG_SIZE_M / 2 + TAG_Y_EXTRA_M,
                      pad_c[2] + TAG_Z_FROM_PAD_M])
    return pad_c, tag_c


# ----------------------------------------------------------------------------------------- loading


def test_no_mujoco_import():
    code = "import sys; sys.modules['mujoco'] = None; from franka_sim import finger_kit; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(fk.Path(fk.__file__).parents[1]))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_box_kit_loads(box):
    k = fk.load_kit(box)
    assert k.id == "box_equivalent" and k.units == "m" and set(k.sides) == {"left", "right"}
    p = k.parts["pad"]
    assert p.role == "contact" and p.collision == "hull" and p.mesh_scale == 1.0
    assert p.vertices.shape == (8, 3) and p.faces.shape == (12, 3)
    assert p.volume_m3 == pytest.approx(np.prod(PAD), rel=1e-12)
    assert p.mass_kg == pytest.approx(1000 * np.prod(PAD), rel=1e-12) and p.mass_source == "density"
    assert k.sides["left"].contact_part == "pad" and k.sides["left"].camera_face == 1.0
    assert k.sides["right"].camera_face == -1.0
    assert k.contact == {"friction": [1.0, 0.005, 0.0001], "solref": None, "solimp": None}
    assert k.sha256 == fk.kit_digest(box)
    assert (box / "SOURCE.md").is_file()


def test_mm_units_convert(tmp_path):
    km = fk.load_kit(fk.make_box_kit(tmp_path / "mm", units="mm"))
    k = fk.load_kit(fk.make_box_kit(tmp_path / "m"))
    assert km.parts["pad"].mesh_scale == 1e-3
    np.testing.assert_allclose(km.parts["pad"].vertices, k.parts["pad"].vertices, atol=1e-12)
    for s in ("left", "right"):
        a, b = km.sides[s], k.sides[s]
        np.testing.assert_allclose(a.tip, b.tip, atol=1e-12)
        np.testing.assert_allclose(a.kit_T_M, b.kit_T_M, atol=1e-12)
        assert a.tags[0].size_m == pytest.approx(b.tags[0].size_m)
        np.testing.assert_allclose(a.tags[0].centre, b.tags[0].centre, atol=1e-12)


def test_expect_sha(box):
    sha = fk.kit_digest(box)
    assert fk.load_kit(box, expect_sha=sha).sha256 == sha
    with pytest.raises(fk.KitError, match="sha256 mismatch"):
        fk.load_kit(box, expect_sha="0" * 64)


def test_digest_stable_and_changes_on_edit(tmp_path):
    a, b = fk.make_box_kit(tmp_path / "a"), fk.make_box_kit(tmp_path / "b")
    assert fk.kit_digest(a) == fk.kit_digest(b) == fk.kit_digest(a)
    assert len(fk.kit_digest(a)) == 64
    d0 = fk.kit_digest(a)
    with open(a / "pad.stl", "a") as f:                   # mesh edit
        f.write("\n")
    assert fk.kit_digest(a) != d0
    d1 = fk.kit_digest(b)
    (b / "kit.yaml").write_text((b / "kit.yaml").read_text() + "\n")   # yaml edit (bytes, not just content)
    assert fk.kit_digest(b) != d1
    (b / "unlisted.txt").write_text("x")                  # unlisted files do not count
    d2 = fk.kit_digest(b)
    (b / "unlisted.txt").write_text("y")
    assert fk.kit_digest(b) == d2


def test_source_md_mismatch_detected(box):
    with open(box / "pad.stl", "a") as f:
        f.write("\n")
    with pytest.raises(fk.KitError, match="differs from SOURCE.md"):
        fk.load_kit(box)
    fk.write_source_md(box)
    fk.load_kit(box)


@pytest.mark.parametrize("edit, msg", [
    (lambda d: d.pop("contact"), "missing top-level keys \\['contact'\\]"),
    (lambda d: d.update(extra=1), "unknown top-level keys"),
    (lambda d: d.update(schema="finger_kit/v0"), "schema must be"),
    (lambda d: d.update(units="inch"), "units must be one of"),
    (lambda d: d["parts"]["pad"].pop("role"), "parts.pad is missing 'role'"),
    (lambda d: d["parts"]["pad"].update(role="soft"), "role must be one of"),
    (lambda d: d["parts"]["pad"].update(material="abs"), "not in materials"),
    (lambda d: d["parts"]["pad"].update(mass_kg=-1), "mass_kg must be a positive"),
    (lambda d: d["parts"]["pad"].update(mesh="../x.stl"), "inside the kit directory"),
    (lambda d: d["parts"]["pad"].update(mesh="nope.stl"), "listed file nope.stl does not exist"),
    (lambda d: d["parts"]["pad"].update(collision="box"), "collision must be 'hull'"),
    (lambda d: d["sides"].pop("right"), "exactly 'left' and 'right'"),
    (lambda d: d["sides"]["left"].pop("tip"), "sides.left is missing 'tip'"),
    (lambda d: d["sides"]["left"].update(parts=["pad", "ghost"]), "unknown part 'ghost'"),
    (lambda d: d["sides"]["left"].update(camera_face="+y"), "camera_face must be"),
    (lambda d: d["sides"]["right"].update(camera_face="+x"), "opposite camera_face"),
    (lambda d: d["sides"]["left"]["tags"][0].update(id=1), "tag ids must be unique"),
    (lambda d: d["sides"]["left"]["tags"][0].update(up=[1, 0, 0]), "orthogonal to normal"),
    (lambda d: d["sides"]["left"]["tags"][0].pop("size"), "must have exactly id, centre"),
    (lambda d: d["sides"]["left"].update(tip=[0, 0]), "tip must be 3"),
    (lambda d: d["sides"]["left"].update(kit_T_M=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]), "4x4"),
    (lambda d: d["sides"]["left"].update(kit_T_M=np.diag([2.0, 1, 1, 1]).tolist()), "not orthonormal"),
    (lambda d: d["sides"]["left"].update(kit_T_M=np.diag([1.0, -1, 1, 1]).tolist()), "reflection"),
    (lambda d: d["materials"]["pad"].update(density_kg_m3=0), "density_kg_m3 must be a positive"),
    (lambda d: d["contact"].update(friction=[1, 2, 3, 4]), "friction must be 1-3"),
    (lambda d: d["contact"].update(solref=[0.1]), "solref must be null or 2"),
    (lambda d: d["provenance"].update(x={"status": "guessed", "source": "?"}), "provenance.x must be"),
])
def test_bad_kits(box, edit, msg):
    _edit(box, edit)
    with pytest.raises(fk.KitError, match=msg):
        fk.load_kit(box)


def test_two_contact_parts_on_a_side(box):
    import shutil

    shutil.copy(box / "pad.stl", box / "pad2.stl")

    def ed(d):
        d["parts"]["pad2"] = dict(d["parts"]["pad"], mesh="pad2.stl")
        d["sides"]["left"]["parts"] = ["pad", "pad2"]
    _edit(box, ed)
    with pytest.raises(fk.KitError, match="exactly one role: contact part, got \\['pad', 'pad2'\\]"):
        fk.load_kit(box)


def test_unused_part_and_structure_part(box):
    import shutil

    shutil.copy(box / "pad.stl", box / "holder.stl")

    def ed(d):
        d["parts"]["holder"] = {"mesh": "holder.stl", "role": "structure", "material": "pad", "mass_kg": 0.012}
    _edit(box, ed)
    with pytest.raises(fk.KitError, match="not used by any side"):
        fk.load_kit(box)
    _edit(box, lambda d: d["sides"]["left"].update(parts=["pad", "holder"]))
    k = fk.load_kit(box)
    assert k.parts["holder"].mass_kg == 0.012 and k.parts["holder"].mass_source == "mass_kg"
    assert set(fk.place(k, "left", J, 0.1, 0.0)["parts"]) == {"pad", "holder"}


def test_not_a_kit(tmp_path):
    with pytest.raises(fk.KitError, match="not a directory"):
        fk.load_kit(tmp_path / "missing")
    (tmp_path / "k").mkdir()
    with pytest.raises(fk.KitError, match="no kit.yaml"):
        fk.load_kit(tmp_path / "k")
    (tmp_path / "k" / "kit.yaml").write_text("a: [")
    with pytest.raises(fk.KitError, match="not valid YAML"):
        fk.load_kit(tmp_path / "k")


# ----------------------------------------------------------------------------------------- STL


def test_stl_ascii_and_binary_agree(tmp_path):
    v, f = fk._box_mesh([-1, -2, -3], [1, 2, 3])
    fk.write_stl(tmp_path / "a.stl", v, f, ascii=True)
    fk.write_stl(tmp_path / "b.stl", v, f, ascii=False)
    assert (tmp_path / "a.stl").read_bytes().startswith(b"solid")
    va, fa = fk.read_stl(tmp_path / "a.stl")
    vb, fb = fk.read_stl(tmp_path / "b.stl")
    np.testing.assert_array_equal(va, vb)
    np.testing.assert_array_equal(fa, fb)
    assert len(va) == 8 and len(fa) == 12
    assert fk._volume(va, fa) == pytest.approx(48.0)
    # outward normals: signed volume positive
    t = va[fa]
    assert np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() > 0


def test_binary_stl_with_solid_header(tmp_path):
    v, f = fk._box_mesh([0, 0, 0], [1, 1, 1])
    fk.write_stl(tmp_path / "b.stl", v, f)
    data = bytearray((tmp_path / "b.stl").read_bytes())
    data[:80] = b"solid but actually binary".ljust(80, b" ")
    (tmp_path / "b.stl").write_bytes(bytes(data))
    assert len(fk.read_stl(tmp_path / "b.stl")[0]) == 8


@pytest.mark.parametrize("content, msg", [
    (b"\x00\x01garbage", "neither binary"),
    (b"solid x\n facet normal 0 0 1\n outer loop\n vertex 0 0\n", "bad vertex line"),
    (b"solid x\n facet normal 0 0 1\n outer loop\n vertex 0 0 0\n vertex 1 0 0\n", "multiple of 3"),
    (b"solid x\n vertex 0 0 nan\n vertex 1 0 0\n vertex 0 1 0\n", "non-finite"),
    (b"solid x\nendsolid x\n", "0 vertices"),
])
def test_unreadable_stl(tmp_path, content, msg):
    (tmp_path / "x.stl").write_bytes(content)
    with pytest.raises(fk.KitError, match=msg):
        fk.read_stl(tmp_path / "x.stl")


def test_unreadable_stl_in_kit(box):
    (box / "pad.stl").write_bytes(b"junk")
    (box / "SOURCE.md").unlink()
    with pytest.raises(fk.KitError, match="unreadable STL .*pad.stl"):
        fk.load_kit(box)


def test_binary_box_kit_float32(tmp_path):
    k = fk.load_kit(fk.make_box_kit(tmp_path / "bin", ascii=False))
    kb = fk.load_kit(fk.make_box_kit(tmp_path / "asc"))
    np.testing.assert_allclose(k.parts["pad"].vertices, kb.parts["pad"].vertices, atol=5e-9)


# ----------------------------------------------------------------------------------------- placement


@pytest.mark.parametrize("side, sgn", [("left", 1.0), ("right", -1.0)])
def test_box_kit_reproduces_box_mode(box, side, sgn):
    k = fk.load_kit(box)
    pl = fk.box_equivalent_placement(J, TIP_Z, EXT, PAD)
    assert pl["dy"] == pytest.approx(-0.0144, abs=1e-15)
    assert pl["L"] == pytest.approx(TIP_Z + EXT - PAD[2], abs=1e-15) and pl["yaw_deg"] == 0.0
    o = fk.place(k, side, J, pl["L"], pl["dy"], pl["yaw_deg"])
    pad_c, tag_c = _box_mode(sgn)
    v = fk._apply(o["parts"]["pad"], k.parts["pad"].vertices)
    np.testing.assert_allclose(v.mean(0), pad_c, atol=1e-12)
    np.testing.assert_allclose(v.max(0) - v.min(0), PAD, atol=1e-12)
    np.testing.assert_allclose(o["body_T_kit"][:3, :3], np.eye(3), atol=1e-15)
    t = o["tags"][0]
    np.testing.assert_allclose(t["centre"] + fk.TAG_LIFT_M * t["normal"], tag_c, atol=1e-12)
    np.testing.assert_allclose(t["normal"], [sgn, 0, 0], atol=1e-15)
    assert t["size_m"] == TAG_SIZE_M and t["id"] == (0 if side == "left" else 1)
    R = t["body_T_tag"][:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R[:, 0], t["normal"])
    np.testing.assert_allclose(R[:, 2], [0, 0, 1])
    # TCP: tip = box pad centre + pl/2 along z (on the centre line after the two-finger midpoint)
    assert o["tip"][2] - pad_c[2] == pytest.approx(PAD[2] / 2, abs=1e-12)
    assert o["camera_face"] == sgn and o["contact_part"] == "pad"


def test_auto_dy(box):
    k = fk.load_kit(box)
    for side in ("left", "right"):
        for yaw in (0.0, 30.0):
            dy = fk.auto_dy(k, side, J, 0.1, yaw)
            o = fk.place(k, side, J, 0.1, dy, yaw)
            ymin = fk._apply(o["parts"]["pad"], k.parts["pad"].vertices)[:, 1].min()
            assert ymin == pytest.approx(0.0, abs=1e-12)
        # box kit: auto closes the placeholder pad offset (documented mapping)
        assert fk.auto_dy(k, side, J, 0.1) == pytest.approx(-0.0144 - PAD_OFFSET_M / 2, abs=1e-12)


def test_place_frame_chain(box):
    k = fk.load_kit(box)
    o = fk.place(k, "left", J, 0.05, -0.01, 90.0)
    np.testing.assert_allclose(o["body_T_M"], J @ make_T(Rz(90.0), [0, -0.01, 0.05]), atol=1e-15)
    np.testing.assert_allclose(o["tags"][0]["normal"], [0, 1, 0], atol=1e-12)    # +x rotated 90 deg -> +y
    with pytest.raises(fk.KitError, match="unknown side"):
        fk.place(k, "middle", J, 0, 0, 0)


def _rigid(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return make_T(R, rng.normal(scale=0.05, size=3))


def test_kit_T_M_mirrored_layout(tmp_path):
    """CAD-style layout: the right part sits mirrored in the assembly (Rz 180 + offset), the left one arbitrarily
    posed. Placement must be identical to the identity-kit_T_M kit: kit_T_M is undone by inv(kit_T_M)."""
    TL = _rigid(1)
    TR = make_T(Rz(180.0), [0.0, -0.09, 0.012])
    km = fk.load_kit(fk.make_box_kit(tmp_path / "mir", kit_T_M_left=TL, kit_T_M_right=TR))
    k0 = fk.load_kit(fk.make_box_kit(tmp_path / "id"))
    assert set(km.parts) == {"pad_left", "pad_right"}
    assert km.parts["pad_right"].vertices[:, 1].max() < 0          # the right pad really is on the -y side in kit coords
    for side in ("left", "right"):
        pn = km.sides[side].contact_part
        a = fk.place(km, side, J, 0.12, -0.0144, 10.0)
        b = fk.place(k0, side, J, 0.12, -0.0144, 10.0)
        va = np.sort(fk._apply(a["parts"][pn], km.parts[pn].vertices), axis=0)
        vb = np.sort(fk._apply(b["parts"]["pad"], k0.parts["pad"].vertices), axis=0)
        np.testing.assert_allclose(va, vb, atol=1e-12)
        np.testing.assert_allclose(a["tip"], b["tip"], atol=1e-15)
        np.testing.assert_allclose(a["tags"][0]["body_T_tag"], b["tags"][0]["body_T_tag"], atol=1e-15)
        assert fk.auto_dy(km, side, J, 0.12) == pytest.approx(fk.auto_dy(k0, side, J, 0.12), abs=1e-12)


def test_box_equivalent_placement_rejects_x_offset():
    with pytest.raises(fk.KitError, match="x offset"):
        fk.box_equivalent_placement(make_T(None, [0.001, 0.0144, 0]), TIP_Z, EXT, PAD)


def test_tip_and_tags_are_in_kit_coords(tmp_path):
    """tip / tag centre / normal / up are kit (CAD) coords: with a non-identity kit_T_M they are stored moved with the
    meshes, and place() maps them through inv(kit_T_M) to the same body-frame values as the identity kit."""
    TL, TR = _rigid(2), make_T(Rz(180.0), [0.0, -0.09, 0.012])
    km = fk.load_kit(fk.make_box_kit(tmp_path / "rot", kit_T_M_left=TL, kit_T_M_right=TR))
    k0 = fk.load_kit(fk.make_box_kit(tmp_path / "id"))
    for side, T in (("left", TL), ("right", TR)):
        a, b = km.sides[side], k0.sides[side]
        assert np.linalg.norm(a.tip - b.tip) > 1e-3                     # stored values really differ
        np.testing.assert_allclose(a.tip, fk._apply(T, b.tip), atol=1e-12)
        np.testing.assert_allclose(a.tags[0].centre, fk._apply(T, b.tags[0].centre), atol=1e-12)
        np.testing.assert_allclose(a.tags[0].normal, T[:3, :3] @ b.tags[0].normal, atol=1e-12)
        np.testing.assert_allclose(a.tags[0].up, T[:3, :3] @ b.tags[0].up, atol=1e-12)
        # the stored tag centre lies on the stored (kit-coords) pad's camera face
        v_M = fk._apply(fk._inv(T), km.parts[a.contact_part].vertices)
        c_M = fk._apply(fk._inv(T), a.tags[0].centre)
        assert c_M[0] == pytest.approx(v_M[:, 0].max() if a.camera_face > 0 else v_M[:, 0].min(), abs=1e-12)
        pa, pb = fk.place(km, side, J, 0.1, -0.0144, 5.0), fk.place(k0, side, J, 0.1, -0.0144, 5.0)
        np.testing.assert_allclose(pa["tip"], pb["tip"], atol=1e-12)
        np.testing.assert_allclose(pa["tags"][0]["centre"], pb["tags"][0]["centre"], atol=1e-12)
        np.testing.assert_allclose(pa["tags"][0]["body_T_tag"], pb["tags"][0]["body_T_tag"], atol=1e-12)


# ----------------------------------------------------------------------------------------- A9 addendum: variants


@pytest.fixture
def vbox(tmp_path):
    return fk.make_box_kit(tmp_path / "vbox", variants=True)


def test_variants_kit_loads(vbox):
    k = fk.load_kit(vbox)
    assert set(k.parts) == {"pad", "holder_left", "holder_right"}
    assert k.parts["holder_left"].role == "structure" and k.parts["holder_left"].mass_kg > 0
    assert k.variants == {"full": {"left": ("pad", "holder_left"), "right": ("pad", "holder_right")},
                          "finger_only": {"left": ("pad",), "right": ("pad",)}}
    assert [t.on for t in k.sides["left"].tags] == ["holder_left"] and k.sides["right"].tags[0].on == "holder_right"
    # the tag centre lies on the holder plate's camera-facing x face (M frame = kit coords here)
    for side, sgn in (("left", 1), ("right", -1)):
        v, c = k.parts[f"holder_{side}"].vertices, k.sides[side].tags[0].centre
        assert c[0] == pytest.approx(v[:, 0].max() if sgn > 0 else v[:, 0].min(), abs=1e-15)
        assert v[:, 1].min() < c[1] < v[:, 1].max() and v[:, 2].min() < c[2] < v[:, 2].max()


def test_no_variants_means_only_full_and_default_carrier(box, tmp_path):
    k = fk.load_kit(box)
    assert k.variants == {"full": {"left": ("pad",), "right": ("pad",)}}
    assert k.sides["left"].tags[0].on == "pad"                   # no structure part: the contact part carries it
    # with structure parts (no 'on'): the side's first structure part in its parts order
    d = fk.make_box_kit(tmp_path / "v", variants=True)

    def ed(doc):
        for sd in doc["sides"].values():
            for t in sd["tags"]:
                t.pop("on")
        doc["sides"]["left"]["parts"] = ["holder_left", "pad"]
    _edit(d, ed)
    k = fk.load_kit(d)
    assert k.sides["left"].tags[0].on == "holder_left" and k.sides["right"].tags[0].on == "holder_right"


def test_place_variants(vbox, box):
    kv, k0 = fk.load_kit(vbox), fk.load_kit(box)
    for side in ("left", "right"):
        full = fk.place(kv, side, J, 0.1, -0.0144, 5.0)
        same = fk.place(kv, side, J, 0.1, -0.0144, 5.0, variant="full")
        only = fk.place(kv, side, J, 0.1, -0.0144, 5.0, variant="finger_only")
        ref = fk.place(k0, side, J, 0.1, -0.0144, 5.0)
        assert list(full["parts"]) == ["pad", f"holder_{side}"] and full["variant"] == "full"
        assert list(only["parts"]) == ["pad"] and only["variant"] == "finger_only"
        assert only["tags"] == [] and only["tags_skipped"] == [{"id": 0 if side == "left" else 1, "on": f"holder_{side}"}]
        assert full["tags_skipped"] == [] and ref["tags_skipped"] == [] and ref["variant"] == "full"
        for o in (full, same, only):          # pad pose and tip do not depend on the variant / holder presence
            np.testing.assert_array_equal(o["parts"]["pad"], ref["parts"]["pad"])
            np.testing.assert_array_equal(o["tip"], ref["tip"])
        for o in (full, same):                # the tag moved onto the holder sits exactly where the box kit put it
            np.testing.assert_array_equal(o["tags"][0]["body_T_tag"], ref["tags"][0]["body_T_tag"])
            assert o["tags"][0]["on"] == f"holder_{side}"
        assert fk.auto_dy(kv, side, J, 0.1) == fk.auto_dy(k0, side, J, 0.1)
    with pytest.raises(fk.KitError, match=r"unknown variant 'tips' \(variants: \['finger_only', 'full'\]\)"):
        fk.place(kv, "left", J, 0.1, 0.0, variant="tips")
    with pytest.raises(fk.KitError, match=r"unknown variant 'finger_only' \(variants: \['full'\]\)"):
        fk.place(k0, "left", J, 0.1, 0.0, variant="finger_only")


def test_variant_parts_in_side_order(vbox):
    _edit(vbox, lambda d: d["variants"].update(both={"left": ["holder_left", "pad"], "right": ["pad", "holder_right"]}))
    k = fk.load_kit(vbox)
    assert fk.variant_parts(k, "both", "left") == ("pad", "holder_left") == k.variants["full"]["left"]


def test_full_may_be_listed_only_unchanged(vbox):
    _edit(vbox, lambda d: d["variants"].update(full={"left": ["holder_left", "pad"], "right": ["pad", "holder_right"]}))
    assert fk.load_kit(vbox).variants["full"]["left"] == ("pad", "holder_left")
    _edit(vbox, lambda d: d["variants"].update(full={"left": ["pad"], "right": ["pad", "holder_right"]}))
    with pytest.raises(fk.KitError, match="variants.full must equal the sides' own parts"):
        fk.load_kit(vbox)


@pytest.mark.parametrize("edit, msg", [
    (lambda d: d.update(variants=[]), "variants must be a non-empty mapping"),
    (lambda d: d.update(variants={}), "variants must be a non-empty mapping"),
    (lambda d: d["variants"].update({"Finger-Only": {"left": ["pad"], "right": ["pad"]}}), "must match \\[a-z0-9_\\]\\+"),
    (lambda d: d["variants"].update(x={"left": ["pad"]}), "variants.x must have exactly 'left' and 'right'"),
    (lambda d: d["variants"].update(x={"left": ["pad"], "right": ["pad"], "mid": []}), "exactly 'left' and 'right'"),
    (lambda d: d["variants"].update(x={"left": "pad", "right": ["pad"]}), "variants.x.left must be a list"),
    (lambda d: d["variants"].update(x={"left": ["pad", "pad"], "right": ["pad"]}), "without duplicates"),
    (lambda d: d["variants"].update(x={"left": ["pad", "ghost"], "right": ["pad"]}), "unknown part 'ghost'"),
    (lambda d: d["variants"].update(x={"left": ["pad", "holder_right"], "right": ["pad"]}),
     "variants.x.left part 'holder_right' is not in sides.left.parts"),
    (lambda d: d["variants"].update(x={"left": ["holder_left"], "right": ["pad"]}),
     "variants.x.left must have exactly one role: contact part, got \\[\\]"),
    (lambda d: d["variants"].update(x={"left": [], "right": ["pad"]}), "exactly one role: contact part"),
    (lambda d: d["sides"]["left"]["tags"][0].update(on="ghost"), "tags\\[0\\].on 'ghost' must be one of sides.left.parts"),
    (lambda d: d["sides"]["left"]["tags"][0].update(on="holder_right"), "on 'holder_right' must be one of"),
    (lambda d: d["sides"]["left"]["tags"][0].update(where="x"), "must have exactly id, centre, normal, up, size"),
])
def test_bad_variants(vbox, edit, msg):
    _edit(vbox, edit)
    with pytest.raises(fk.KitError, match=msg):
        fk.load_kit(vbox)


def test_variants_change_the_digest(tmp_path):
    a = fk.kit_digest(fk.make_box_kit(tmp_path / "a"))
    b = fk.kit_digest(fk.make_box_kit(tmp_path / "b", variants=True))
    assert a != b and a == fk.kit_digest(fk.make_box_kit(tmp_path / "c"))
