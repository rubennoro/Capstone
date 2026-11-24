"""W2 step 1 "look": write the look stations from ``stations/sim_rig.yaml`` (text copy, only the look lines changed).

    unset PYTHONPATH; cd ~/Capstone-main-sim
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.make_stations [--check]

Writes ``sim_deploy/stations/sim_rig_look_real.yaml`` (id ``sim_rig_look_real``),
``sim_rig_look_real_bg.yaml``, ``sim_rig_look_real_kit.yaml`` (``real`` + the handheld kit's colours: training's
light-blue pads and blue holders), ``sim_rig_look_real_bg_kit.yaml`` (``real_bg`` + those colours), ``sim_rig_look_train.yaml`` (``real_bg_kit`` +
the eraser at the length the training frames show, 111 mm, width 30 mm as W0: the full training-scene look),
``sim_rig_look_robot.yaml`` (the real robot cell, from ``look/assets/robot_look.json``)
and the attribution-only ``sim_rig_look_real_nocam.yaml`` / ``sim_rig_look_robot_nocam.yaml`` (``sim_pi.model`` = the
look's build, no camera model; never built) from the CURRENT ``sim_rig.yaml``
(read-only), so a later change of sim_rig.yaml (physics, kit, layout) is carried over by running this again. Each
replaced line is asserted to exist exactly once (a changed sim_rig.yaml fails loudly instead of silently diverging).
``--check``: only report whether the files on disk equal what would be written. Both modes (without ``--only``) then
fail (exit 2) if any ``stations/*.yaml`` eraser width is not ``sim_deploy/eraser.py`` WIDTH_M (30 mm).

What differs from sim_rig.yaml (visual only; physics identical, see ``look/check_look.py physics``):

* ``appearance.table_texture: file``: the real table (``look/assets/table_real.png`` from ``make_assets table``) at
  real scale and its real place in O (``table_real.json``: size, origin), on a visual-only slab over the table box;
  matte (specular 0); the box's sides get the tile's mean colour.
* ``scene.eraser.rgba``: the real eraser's top-face colour as the IMX519 records it (``eraser_real.json``) divided
  by the light on a horizontal face (ambient + diffuse x cos, ~1.0), so the render reproduces it.
* lights: soft and diffuse like the real room: headlight ambient ``AMBIENT`` (view-independent) + one directional
  light from almost straight above with diffuse ``DIFFUSE``, no shadow casting, no specular; ``sim_pi.shadows:
  false`` (no shadow pass); ``sim_pi.camera_look: ../look/camera_look_real.yaml`` (step 2's camera model) once that
  file exists.
* skybox / floor neutral grey.
* ``_bg`` only: ``appearance.background_mesh`` = the PICK_TEST1 workspace reconstruction (``make_assets mesh``,
  vertices already in O: pose identity) + proxy walls painted from the real IMX219 frames (``make_assets backdrop``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sim_deploy import eraser as eraser_mod

ROOT = Path(__file__).resolve().parents[1]                 # sim_deploy/
STATIONS = ROOT / "stations"
ASSETS = ROOT / "look" / "assets"
SRC = STATIONS / "sim_rig.yaml"

AMBIENT = 0.75                                             # headlight ambient (all faces)
DIFFUSE = 0.25                                             # one directional light, nearly straight down
LIGHT_DIR = (0.05, 0.03, -1.0)


def _top_light() -> float:
    d = np.asarray(LIGHT_DIR, float)
    return AMBIENT + DIFFUSE * float(-d[2] / np.linalg.norm(d))


def _f(x) -> str:
    return "[" + ", ".join(repr(round(float(v), 4)) for v in x) + "]"


def _replace_once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise ValueError(f"sim_rig.yaml: expected exactly one {old!r}, found {n} (sim_rig.yaml changed: update "
                         "make_stations.py)")
    return text.replace(old, new)


def _replace_line_starting(text: str, prefix: str, new: str) -> str:
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    if len(hits) != 1:
        raise ValueError(f"sim_rig.yaml: expected one line starting {prefix!r}, found {len(hits)}")
    lines[hits[0]] = new
    return "\n".join(lines)


#: The eraser's width is the same in EVERY station (2026-11-21, the user): sim_deploy/eraser.py WIDTH_M (W0's 30 mm, as
#: sim_rig.yaml; the 0.040 m gripper must straddle it), image looks included; ``--check`` fails if any station's
#: eraser width differs. Each look keeps its own length and the 12 mm thickness:
#: * image looks (real*, the robot what-ifs): the user's eraser's 127 mm length (it is 127 x 51 x 12 mm; until
#:   2026-11-21 these looks used 51 mm, W2's numbers are from then) -> 127 x 30 x 12 = sim_rig's size (physics as sim_rig);
#: * ``train``: the length the PICK_TEST1 training frames show, 111 mm (look/assets/table_real.json
#:   eraser_check_summary: 100 IMX519 frames rectified at its top face, length median 110.9 mm (p10-p90 107.7-113.1);
#:   width 52.1 mm, not used), not the 127 mm of the user's eraser: a 15 % longer eraser looks nearer, and the run_006
#:   first chunk on the approach references came out shorter in the sim than on the real frames;
#: * ``robot``: 127 mm (= sim_rig).
#: ``robot`` and ``train`` are also the sim's closed-loop looks (run.sh --look). The attribution-only ``*_nocam``
#: stations are rendered from their look and follow it.
RUN_LOOKS = ("robot", "train")
ERASER_THICKNESS_M = 0.012
W_MM = f"{eraser_mod.WIDTH_M * 1e3:.0f}"                  # for the station headers
ERASER_LENGTH_M = {"real": 0.127, "robot": 0.127, "train": 0.111}   # image looks and the what-ifs: "real"


def eraser_size_line(look: str) -> str:
    """scene.eraser.size_m for the look: its length (ERASER_LENGTH_M), sim_deploy/eraser.py's width, 12 mm."""
    L = ERASER_LENGTH_M.get(look, ERASER_LENGTH_M["real"])
    sz = eraser_mod.size_m(L, ERASER_THICKNESS_M)
    why = {"robot": "run look", "train": "run look, the training frames' length"}.get(look, "image look, the real length")
    return (f"    size_m: [{sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f}]           # {why}; width = sim_deploy/eraser.py "
            f"WIDTH_M ({sz[1] * 1e3:.0f} mm, every station)")


def _real_eraser_size(t: str, line: str) -> str:
    return _replace_line_starting(t, "    size_m: [0.127,", line)


CAMERA_LOOK = ROOT / "look" / "camera_look_real.yaml"      # step 2's camera model (emitted once it exists)


def _sim_pi_block(model: str | None = None) -> str:
    """The generated sim_pi block: no shadow pass; the camera model when ``look/camera_look_real.yaml`` exists
    (omitted for the nocam stations, which instead point ``model`` at the look's build)."""
    lines = ["# W2: no light casts a shadow (soft room light): no shadow pass", "sim_pi:", "  shadows: false"]
    if model is not None:
        lines.append(f"  model: {model}   # nocam: the look's own build, rendered without the camera model")
    elif CAMERA_LOOK.is_file():
        lines.append("  camera_look: ../look/camera_look_real.yaml   # W2 step 2: the real cameras' response")
    return "\n".join(lines) + "\n"


def render_nocam(look: str) -> str:
    """``sim_rig_look_<look>_nocam.yaml``: the look's station text with id ``..._nocam``, ``sim_pi.model`` = the look's
    build and no camera model (attribution only: the scene without the camera model; never built)."""
    base = "sim_rig_look_" + look
    t = render(look)
    t = _replace_once(t, f"\nid: {base}\n", f"\nid: {base}_nocam\n")
    i = t.index("# W2: no light casts a shadow")
    t = t[:i] + _sim_pi_block(model=f"../build/{base}/scene.xml")
    head = [f"# sim_deploy station '{base}_nocam': ATTRIBUTION ONLY, DO NOT BUILD. The scene of {base} (its build,",
            f"# sim_pi.model ../build/{base}/scene.xml) rendered WITHOUT the camera model (no sim_pi.camera_look).",
            "# Generated by `python -m sim_deploy.look.make_stations` with the look station below.", ""]
    return "\n".join(head) + t


def render(look: str) -> str:
    if look == "robot" or look in ROBOT_WHATIF:
        return render_robot(ROBOT_WHATIF.get(look, ()))
    src = SRC.read_text()
    tile = json.loads((ASSETS / "table_real.json").read_text())["tile"]
    er = json.loads((ASSETS / "eraser_real.json").read_text())
    top = _top_light()
    er_rgba = [min(1.0, c / 255.0 / top) for c in er["median_rgb"]] + [1.0]
    table_rgba = [c / 255.0 for c in tile["mean_rgb"]] + [1.0]
    sid = {"real": "sim_rig_look_real", "real_bg": "sim_rig_look_real_bg", "real_kit": "sim_rig_look_real_kit",
           "real_bg_kit": "sim_rig_look_real_bg_kit",
           "train": "sim_rig_look_train"}[look]
    head = [
        f"# sim_deploy station {sid!r} (W2 look {look!r}): GENERATED by `python -m sim_deploy.look.make_stations` from",
        "# stations/sim_rig.yaml (do not edit by hand; rerun it after sim_rig.yaml changes). Physics as sim_rig",
        *([f"# except the eraser box's length (111 mm, the training frames'; width {W_MM} mm as every station). A RUN look",
           "# (run.sh --look train): the sim PC and the sim Pi both load its build. Only the look"] if look == "train" else
          [f"# (eraser 127 x {W_MM} x 12 mm: every station's width, sim_deploy/eraser.py); the look stations are for",
           "# images only (render_sets / robot_trials), not for the closed loop. Only the look"]),
        "# differs, to make the sim camera images close to the real Pi frames (W2, PLAN.md section 11):",
        f"#  - table: the real light wood photographed by the IMX519 (look/assets/table_real.png, {tile['size_m'][0]:.2f} x "
        f"{tile['size_m'][1]:.2f} m per tile,",
        "#    at its real place in O via layout.json T_O_tag), matte, on a visual-only slab; box sides = its mean colour",
        f"#  - eraser rgba = the real lime top face as the IMX519 records it {er['median_rgb']} / 255 / {top:.3f} (the light",
        "#    on a horizontal face), so the render reproduces it",
        f"#  - soft diffuse light (the real frames show only faint soft shadows): headlight ambient {AMBIENT} + one",
        f"#    directional light from nearly straight above, diffuse {DIFFUSE}, no shadows, no specular; sim_pi shadows off",
        "#  - skybox and floor neutral grey",
    ]
    if look in ("real_bg", "real_bg_kit", "train"):
        head += ["#  - background: the PICK_TEST1 workspace reconstruction (umiproc mesh_3d, look/assets/workspace_mesh.obj,",
                 "#    already in O; table faces and anything near the robot removed; colours matched to table_real.png)",
                 "#    + proxy walls behind the desk painted from the real handheld IMX219 frames (look/assets/backdrop_real.obj)"]
    if look in ("real_kit", "real_bg_kit", "train"):
        head += [f"#  - kit colours of the HANDHELD in training: fin-ray pads {KIT_REAL['pads_rgba'][:3]} (light blue), holders",
                 f"#    {KIT_REAL['holders_rgba'][:3]} (blue); sampled in B1's IMX519 640 frames under the sim's pad / holder",
                 "#    labels (median), scaled by real / sim-render per channel (the robot's printed pads are white)"]
    head += ["# The rest of this file is sim_rig.yaml's text (its comments describe sim_rig's look where they talk about it).", ""]
    t = _replace_once(src, "\nid: sim_rig\n", f"\nid: {sid}\n")
    if look in ("real_kit", "real_bg_kit", "train"):
        t = _replace_line_starting(t, "  pads_rgba:", f"  pads_rgba: {_f(KIT_REAL['pads_rgba'])}   # W2 real_kit: training's light-blue pads")
        t = _replace_line_starting(t, "  holders_rgba:", f"  holders_rgba: {_f(KIT_REAL['holders_rgba'])}   # W2 real_kit: training's blue holders")
    app = [
        "  table_texture: file                       # W2: the real table (make_assets table)",
        "  table_texture_file: ../look/assets/table_real.png",
        f"  table_texture_size_m: {_f(tile['size_m'])}",
        f"  table_texture_origin_O_m: {_f(tile['origin_O_m'])}   # O (x, y) of the image's bottom-left corner",
        "  table_specular: 0.0",
        "  table_shininess: 0.0",
        f"  table_rgba: {_f(table_rgba)}   # the box's sides: the tile's mean colour",
    ]
    if look in ("real_bg", "real_bg_kit", "train"):
        app[-1] = "  table_rgba: [0.5, 0.5, 0.5, 0.0]          # the colliding box: hidden (alpha 0), physics unchanged"
        app += [f"  table_visual_x_m: {_f(DESK_X_M)}           # W2 real_bg: the real desk's extent (DESK_X_M / DESK_Y_M)",
                f"  table_visual_y_m: {_f(DESK_Y_M)}",
                "  table_texture_slab_m: 0.03"]
        app += ["  background_mesh:                          # W2: the lab around the table",
                "    - obj: ../look/assets/workspace_mesh.obj   # umiproc's reconstruction (make_assets mesh)",
                "      texture: ../look/assets/workspace_mesh.png",
                "      pos: [0.0, 0.0, 0.0]                  # the OBJ is already in O (workspace_mesh.json T_O_mesh)",
                "      quat: [1.0, 0.0, 0.0, 0.0]",
                "      emission: 3.0                         # nearly unlit: the photographed colours as they are",
                "    - obj: ../look/assets/backdrop_real.obj    # walls behind the desk painted from the real IMX219",
                "      texture: ../look/assets/backdrop_real.png  # frames (make_assets backdrop; where the mesh has holes)",
                "      pos: [0.0, 0.0, 0.0]",
                "      quat: [1.0, 0.0, 0.0, 0.0]",
                "      emission: 3.0"]
    t = _replace_once(t, "  table_texture: wood\n", "\n".join(app) + "\n")
    t = _replace_line_starting(t, "  table_texrepeat:", "")
    t = t.replace("\n\n\n", "\n\n")
    t = _replace_line_starting(t, "    rgba: [0.96, 0.80, 0.08, 1.0]",
                               f"    rgba: {_f(er_rgba)}   # W2: the real lime top face (eraser_real.json) / light")
    t = _replace_line_starting(t, "  floor_rgba:", "  floor_rgba: [0.45, 0.45, 0.45, 1.0]")
    t = _real_eraser_size(t, eraser_size_line(look))
    t = _replace_line_starting(t, "  skybox:", "  skybox: {rgb_top: [0.72, 0.72, 0.72], rgb_bottom: [0.50, 0.50, 0.50]}   # W2: neutral grey")
    t = _replace_once(t, "    - {dir: [-0.25, 0.15, -1.0], diffuse: [0.50, 0.50, 0.50], specular: [0.05, 0.05, 0.05], castshadow: true}\n"
                         "    - {dir: [0.40, -0.50, -0.70], diffuse: [0.20, 0.20, 0.20], specular: [0.0, 0.0, 0.0], castshadow: false}\n",
                      f"    - {{dir: {_f(LIGHT_DIR)}, diffuse: {_f([DIFFUSE] * 3)}, specular: [0.0, 0.0, 0.0], castshadow: false}}   # W2\n")
    t = _replace_line_starting(t, "  headlight:", f"  headlight: {{ambient: {_f([AMBIENT] * 3)}, diffuse: [0.0, 0.0, 0.0], "
                                                  "specular: [0.0, 0.0, 0.0]}   # W2: soft room light")
    if "\nsim_pi:" in t:
        raise ValueError("sim_rig.yaml has a sim_pi block now: update make_stations.py")
    t = t.rstrip("\n") + "\n\n" + _sim_pi_block()
    return "\n".join(head) + t


#: The real PICK_TEST1 desk's extent in O (W2 real_bg looks): the umiproc workspace mesh's own table faces (within
#: 12 mm of the fitted table plane, make_assets mesh; 94 412 faces) on a 5 cm grid, cells with > 20 faces: x from
#: -0.05 to 1.10 m, y from -0.30 (a straight edge) to 0.55 m (the far edge, under the monitors; the keyboard on the
#: desk at y 0.3-0.4 is a hole in the table faces). The handheld IMX219 looks along +y and sees the far edge at
#: about a quarter of the frame height; the sim table box is larger, so beyond this the sim drew wood where the real
#: frames show the keyboard, monitors and the lab. The colliding box is hidden (alpha 0) and only this visual slab
#: carries the wood; beyond it the workspace mesh, the backdrop walls and the floor.
DESK_X_M = (-0.05, 1.10)
DESK_Y_M = (-0.30, 0.55)

#: The handheld kit's colours in the training frames (W2 real_kit). B1's 15 references, IMX519 640 frames (raw640_main_*),
#: median under the sim's pad / holder labels (eroded 7 px) of the sim_rig_look_real handheld renders (fit_camera_look
#: render): pads real (121, 180.5, 224) vs sim (213, 213, 211) with rgba (0.93, 0.93, 0.92); holders real (33, 94, 165) vs
#: sim (51, 70, 145) with (0.22, 0.30, 0.62); new rgba = old x real / sim per channel (IMX219: pads (134, 170, 202)).
KIT_REAL = {"pads_rgba": [0.528, 0.788, 0.977, 1.0], "holders_rgba": [0.142, 0.403, 0.705, 1.0]}

HOLDER_LIGHT = 0.85       # the light on the holders' faces (mostly side-on to the overhead light: ambient + a bit)


#: What-if variants of the robot cell (W2): the real robot cell with ONE training-scene element put back, to see which
#: difference between the robot cell and the training scene changes a policy's response (sim only; each is a
#: recommendation candidate for the real cell, never the default): lime = training's lime eraser colour (eraser_real.json),
#: kit = the handheld's light-blue pads and blue holders (KIT_REAL), wood = training's wood table texture (table_real.png)
#: on the robot table's extent; train = all three.
ROBOT_WHATIF = {"robot_lime": ("lime",), "robot_kit": ("kit",), "robot_wood": ("wood",),
                "robot_train": ("lime", "kit", "wood")}


def render_robot(whatif=()) -> str:
    """The ``robot`` look (the REAL ROBOT CELL, W2 addendum): the colliding table box hidden (alpha 0) and a visual
    table of the real extent (white, from the mount IMX519 frames), the sim floor hidden, a projective backdrop of
    the cell (floor beyond the table edge, walls, the person beside the table) from the mount frames, dark grey
    holders / adapters, the yellow object as the IMX519 records it on the robot; the light as in ``real``."""
    src = SRC.read_text()
    rb = json.loads((ASSETS / "robot_look.json").read_text())
    top = _top_light()
    tt, tb = rb["table_texture"], rb["table"]
    er_rgba = [min(1.0, c / 255.0 / top) for c in rb["object_rgb_imx519_224"]] + [1.0]
    hold = [min(1.0, c / 255.0 / HOLDER_LIGHT) for c in rb["holder_rgb_imx519_224"]] + [1.0]
    sid = "sim_rig_look_robot" + ("" if not whatif else "_" + ("train" if len(whatif) == 3 else whatif[0]))
    head = [
        f"# sim_deploy station {sid!r} (W2 look {sid[len('sim_rig_look_'):]!r}): GENERATED by `python -m sim_deploy.look.make_stations` from",
        "# stations/sim_rig.yaml + look/assets/robot_look.json (do not edit by hand; rerun after sim_rig.yaml changes).",
        (f"# Physics as sim_rig (eraser 127 x {W_MM} x 12; images only). The look = the REAL ROBOT CELL as the" if whatif else
         f"# Physics as sim_rig (eraser 127 x {W_MM} x 12; a RUN look, run.sh --look robot: the sim PC and the sim Pi both load its build). The look = the REAL ROBOT CELL as the"),
        "# real-arm mount frames of 2026-11-20 show it (W2 addendum; look/make_robot.py):",
        f"#  - table: the colliding box is kept but hidden (alpha 0); a visual-only white table of the real extent",
        f"#    x {tb['x_m']}, y {tb['y_m']} (its right and far edges measured in both mount cameras), 30 mm thick,",
        "#    colour from the mount IMX519 frames (look/assets/table_robot.png)",
        "#  - beyond it: a projective backdrop (look/assets/backdrop_robot.obj: floor at the sim floor z, walls, the",
        "#    person beside the table as a flat plane) painted from the real mount frames; the sim floor hidden",
        f"#  - holders and adapters dark grey as on the robot ({rb['holder_rgb_imx519_224']} / 255 / {HOLDER_LIGHT})",
        f"#  - eraser rgba = the yellow object as the IMX519 records it on the robot {rb['object_rgb_imx519_224']} / 255",
        f"#  - soft diffuse light as in sim_rig_look_real (ambient {AMBIENT} + one top light {DIFFUSE}, no shadows)",
        "# The rest of this file is sim_rig.yaml's text (its comments describe sim_rig's look where they talk about it).", ""]
    if whatif:
        head[-2:-2] = [f"# WHAT-IF variant {sid!r}: the robot cell with the training scene's {', '.join(whatif)} (ROBOT_WHATIF in",
                       "# make_stations.py); everything else as sim_rig_look_robot. Sim-only, for the attribution."]
    t = _replace_once(src, "\nid: sim_rig\n", f"\nid: {sid}\n")
    if "lime" in whatif:
        er = json.loads((ASSETS / "eraser_real.json").read_text())
        er_rgba = [min(1.0, c / 255.0 / top) for c in er["median_rgb"]] + [1.0]
    if "kit" in whatif:
        hold = KIT_REAL["holders_rgba"]
    if "wood" in whatif:
        tile = json.loads((ASSETS / "table_real.json").read_text())["tile"]
        tt = {"size_m": tile["size_m"], "origin_O_m": tile["origin_O_m"], "file": "table_real.png"}
    app = [
        "  table_texture: file                       # W2 robot look: the real robot table (make_robot)",
        f"  table_texture_file: ../look/assets/{tt.get('file', 'table_robot.png')}",
        f"  table_texture_size_m: {_f(tt['size_m'])}",
        f"  table_texture_origin_O_m: {_f(tt['origin_O_m'])}",
        f"  table_visual_x_m: {_f(tb['x_m'])}            # the real table's extent (visual only)",
        f"  table_visual_y_m: {_f(tb['y_m'])}",
        "  table_texture_slab_m: 0.03",
        "  table_specular: 0.0",
        "  table_shininess: 0.0",
        "  table_rgba: [0.5, 0.5, 0.5, 0.0]          # the colliding box: hidden (alpha 0), physics unchanged",
        "  background_mesh:                          # the robot cell around the table (make_robot)",
        "    obj: ../look/assets/backdrop_robot.obj",
        "    texture: ../look/assets/backdrop_robot.png",
        "    pos: [0.0, 0.0, 0.0]",
        "    quat: [1.0, 0.0, 0.0, 0.0]",
        "    emission: 3.0",
    ]
    t = _replace_once(t, "  table_texture: wood\n", "\n".join(app) + "\n")
    t = _replace_line_starting(t, "  table_texrepeat:", "")
    t = t.replace("\n\n\n", "\n\n")
    t = _replace_line_starting(t, "  holders_rgba:", f"  holders_rgba: {_f(hold)}   # W2 robot: "
                               + ("training's blue holders (what-if)" if "kit" in whatif else "dark grey as on the robot"))
    t = _replace_line_starting(t, "  adapters_rgba:", f"  adapters_rgba: {_f(hold)}")
    if "kit" in whatif:
        t = _replace_line_starting(t, "  pads_rgba:", f"  pads_rgba: {_f(KIT_REAL['pads_rgba'])}   # W2 what-if: training's light-blue pads")
        t = _replace_line_starting(t, "  adapters_rgba:", f"  adapters_rgba: {_f(KIT_REAL['holders_rgba'])}")
    t = _replace_line_starting(t, "    rgba: [0.96, 0.80, 0.08, 1.0]",
                               f"    rgba: {_f(er_rgba)}   # W2 robot: "
                               + ("training's lime eraser / light (what-if)" if "lime" in whatif else "the yellow object on the robot / light"))
    t = _replace_line_starting(t, "  floor_rgba:", "  floor_rgba: [0.45, 0.45, 0.45, 0.0]   # W2 robot: hidden (the backdrop has the floor)")
    t = _real_eraser_size(t, eraser_size_line("real" if whatif else "robot"))
    t = _replace_line_starting(t, "  skybox:", "  skybox: {rgb_top: [0.72, 0.72, 0.72], rgb_bottom: [0.50, 0.50, 0.50]}   # W2: neutral grey")
    t = _replace_once(t, "    - {dir: [-0.25, 0.15, -1.0], diffuse: [0.50, 0.50, 0.50], specular: [0.05, 0.05, 0.05], castshadow: true}\n"
                         "    - {dir: [0.40, -0.50, -0.70], diffuse: [0.20, 0.20, 0.20], specular: [0.0, 0.0, 0.0], castshadow: false}\n",
                      f"    - {{dir: {_f(LIGHT_DIR)}, diffuse: {_f([DIFFUSE] * 3)}, specular: [0.0, 0.0, 0.0], castshadow: false}}   # W2\n")
    t = _replace_line_starting(t, "  headlight:", f"  headlight: {{ambient: {_f([AMBIENT] * 3)}, diffuse: [0.0, 0.0, 0.0], "
                                                  "specular: [0.0, 0.0, 0.0]}   # W2: soft room light")
    t = t.rstrip("\n") + "\n\n" + _sim_pi_block()
    return "\n".join(head) + t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", choices=("real", "real_bg", "real_kit", "real_bg_kit", "train", "robot", "real_nocam", "robot_nocam",
                     *ROBOT_WHATIF))
    a = ap.parse_args(argv)
    rc = 0
    for look in ("real", "real_bg", "real_kit", "real_bg_kit", "train", "robot", "real_nocam", "robot_nocam",
                     *ROBOT_WHATIF):
        if a.only and look != a.only:
            continue
        path = STATIONS / f"sim_rig_look_{look}.yaml"
        text = render_nocam(look[:-6]) if look.endswith("_nocam") else render(look)
        if a.check:
            same = path.is_file() and path.read_text() == text
            print(f"{path.name}: {'up to date' if same else 'DIFFERS'}")
            rc |= 0 if same else 1
        else:
            tmp = path.with_suffix(".yaml.tmp")
            tmp.write_text(text)
            tmp.replace(path)
            print(f"wrote {path}")
    if not a.only:                 # every station's eraser has the one width (sim_rig.yaml and the looks on disk)
        errs = eraser_mod.station_width_errors()
        for e in errs:
            print(f"ERASER WIDTH WRONG: {e}", file=sys.stderr)
        if errs:
            rc |= 2
        else:
            print(f"eraser width: {W_MM} mm in every station (sim_deploy/eraser.py WIDTH_M)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
