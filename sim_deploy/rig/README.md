# sim_deploy/rig: the CAD rig on the simulated Franka Hand

The team's SolidWorks assembly `franka_mount_with_rig.STEP` (mount + plate + top cover + IMX519, IMX219, IMU, Pi 5,
button and LED boards) turned into MuJoCo assets, camera and IMU poses, and mass properties, all in the **Franka Hand
frame** (menagerie `hand` body: z along the hand to the fingertips, 0 at the hand–arm joint = flange; x out of the
camera-side face). Frames and key numbers: `rig.yaml` and [../DESIGN.md](../DESIGN.md) "E2: as built".

| File | What it is |
|---|---|
| `extract.py` | STEP -> `assets/` + `rig.yaml`. Re-runnable; needs the CAD venv below. |
| `rig.yaml` | schema `sim_deploy.rig/v1`: STEP sha256, STEP->hand and hand->flange transforms, `cam_imx219` / `cam_imx519` optical centres + OpenCV and MuJoCo rotations with ±, IMU pose, camera->fingertip transforms (old kit and as a function of z_tip) vs training, per-part and total mass/COM/inertia (hand and flange frames), station tooling values (F_T_C, C_main_T_C_uw, boxes, payload), the 11-17 check. Generated: do not edit by hand. |
| `assets/<part>.stl` | visual meshes, hand frame, metres, ≤ 20k faces: mount, plate, cover, imx519_board, imx219_board (with lens), imu_board, pi5, button_board, led_strip. SMD parts are merged into their board. |
| `assets/collision/<part>.stl` | convex hulls of mount, plate, cover, pi5 |
| `add_rig.py` | `add_rig(spec, hand_body, rig_dir, intrinsics=None) -> dict`: adds the meshes (group 2), hulls (group 5, contype/conaffinity 1), body `rig` with the rig's inertial, and cameras `cam_imx519`, `cam_imx219` on the hand body. Also `load_rig`, `hand_T_cam`, `cam_T_tip`. Needs only mujoco + numpy + pyyaml. |
| `smoke_render.py` | franka_sim's `sim-rig` model + the rig: compile, print the camera poses read back and where the fingertips land, save PNGs. |

## Regenerate

CAD venv (one time; built and run with Python 3.13):

```bash
python3 -m venv ~/envs/cad
~/envs/cad/bin/pip install cadquery-ocp trimesh rtree scipy fast-simplification pyyaml
```

Tested with cadquery-ocp 8.0.1.1, trimesh 5.2.1, rtree 1.4.1, scipy 1.18.1, fast_simplification 0.2.0, PyYAML 6.0.3.

```bash
cd ~/Capstone-main-sim
~/envs/cad/bin/python -I sim_deploy/rig/extract.py [--step ~/Downloads/franka_mount_with_rig.STEP] [--max-faces 20000]
MUJOCO_GL=egl ~/envs/env_sim/bin/python sim_deploy/rig/smoke_render.py --out /tmp/rig_smoke
```

`extract.py` takes about 30 s (15 s of it reading the STEP). Assumptions it encodes (each is a constant at the top
of the file and is written into `rig.yaml`): the STEP->hand transform, the printed parts' fill factors, the board
masses, the 20 g of unmodelled screws and cables, and how far behind each lens front the optical centre sits.

## Use

```python
from sim_deploy.rig.add_rig import add_rig
info = add_rig(spec, "hand/hand", "sim_deploy/rig",
               intrinsics={"cam_imx519": (K519, 640, 480), "cam_imx219": (K219, 640, 480)})
model = spec.compile()
```

Mesh files are referenced by absolute path (`info["mesh_files"]`); a builder that writes `scene.xml` + `meshes/`
copies them. Rig geoms sit on a joint-free body under the hand, so they share link 7's weld: MuJoCo never collides
them with the hand, the fingers, link 7 or link 6 (filterparent), only with links 0–5, the table and objects.
