"""Planning reference (verified 2026-11-12). The executor turns these checks into deploy/tests/test_geometry.py.

Worked check of the gripper offsets (the "triangle").

Frames:
  F    flange: Franka convention, origin at the plate centre, z out of the plate toward the fingers.
  TCP  fingertip point: the Franka Hand TCP, F_T_TCP = Rz(-45 deg), 0.1034 m along flange z (stock fingers).
  C    ultrawide camera optical frame (OpenCV): x right in the image, y down in the image, z out of the lens.

Inputs that describe the mount (EXAMPLE numbers until the real CAD values are filled in):
  cam_pos_F     camera lens centre in flange coordinates (m)
  tilt_deg      how far the camera looks "down" toward the fingers, measured from the flange z axis
  up_axis_F     which flange direction is "up" in the camera image (the side the camera sits on)

Output:
  T_cam_tcp = inverse(F_T_cam) @ F_T_TCP, printed in rig-file form (euler_xyz_deg + translation_m), and checks.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def T(Rm, t):
    M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = t; return M


def inv(M):
    Ri, t = M[:3, :3].T, M[:3, 3]
    return T(Ri, -Ri @ t)


# --- fixed by the hardware: flange -> fingertip point (Franka Hand nominal; printed fingers may lengthen it) ---
finger_extra_m = 0.0                       # TODO: extra length of the printed fingers vs stock Franka fingers
F_T_TCP = T(R.from_euler("z", -45, degrees=True).as_matrix(), [0, 0, 0.1034 + finger_extra_m])

# --- the camera mount (EXAMPLE values; replace with CAD) ---
cam_pos_F = np.array([0.0, 0.07, 0.04])    # 7 cm off-axis toward flange +y, 4 cm out from the plate
up_axis_F = np.array([0.0, 1.0, 0.0])      # the camera sits on the flange +y side; image "up" points that way
tilt_deg = 30.0                            # looks 30 deg "down" (toward the flange axis / fingers)


def camera_pose(cam_pos_F, up_axis_F, tilt_deg):
    """F_T_cam for a camera at cam_pos_F, looking along flange z tilted by tilt_deg toward the axis."""
    up = up_axis_F / np.linalg.norm(up_axis_F)
    z_f = np.array([0.0, 0.0, 1.0])
    # viewing direction: flange z, rotated toward -up (i.e. looking "down" at the fingers) by tilt_deg
    a = np.radians(tilt_deg)
    z_c = np.cos(a) * z_f - np.sin(a) * up
    y_c = -(up - (up @ z_c) * z_c); y_c /= np.linalg.norm(y_c)   # image y points down = away from "up"
    x_c = np.cross(y_c, z_c)                                      # right-handed: x = y cross z
    return T(np.column_stack([x_c, y_c, z_c]), cam_pos_F)


F_T_cam = camera_pose(cam_pos_F, up_axis_F, tilt_deg)
T_cam_tcp = inv(F_T_cam) @ F_T_TCP

# --- checks ---
print("F_T_cam (camera's constant pose on the flange):\n", np.round(F_T_cam, 4))
print("\nT_cam_tcp (what goes in the rig file):")
print("  rotation euler_xyz_deg:", np.round(R.from_matrix(T_cam_tcp[:3, :3]).as_euler("xyz", degrees=True), 3))
print("  translation_m:        ", np.round(T_cam_tcp[:3, 3], 4))
print("  distance camera -> fingertips: %.4f m" % np.linalg.norm(T_cam_tcp[:3, 3]))

# 1. the triangle closes
assert np.allclose(F_T_cam @ T_cam_tcp, F_T_TCP, atol=1e-12)
# 2. rig-file round trip: euler_xyz_deg written and read back gives the same matrix (dataproc uses from_euler("xyz"))
e = R.from_matrix(T_cam_tcp[:3, :3]).as_euler("xyz", degrees=True)
assert np.allclose(R.from_euler("xyz", e, degrees=True).as_matrix(), T_cam_tcp[:3, :3], atol=1e-12)
# 3. the fingertips are in front of the camera (positive z in the optical frame) and below image centre (y > 0)
assert T_cam_tcp[2, 3] > 0 and T_cam_tcp[1, 3] > 0
# 4. a training pose and a robot pose describe the same physical point and axes:
#    pick any flange pose O_T_F; the camera is at O_T_F @ F_T_cam; dataproc's TCP = O_T_cam @ T_cam_tcp;
#    franka_control's TCP = O_T_F @ F_T_TCP. They must be identical (position AND axes).
rng = np.random.default_rng(0)
for _ in range(1000):
    O_T_F = T(R.random(random_state=rng).as_matrix(), rng.uniform(-1, 1, 3))
    assert np.allclose((O_T_F @ F_T_cam) @ T_cam_tcp, O_T_F @ F_T_TCP, atol=1e-9)
# 5. a relative motion the policy outputs (in the TCP frame) produces the same fingertip motion on both sides
dT = T(R.from_euler("xyz", [5, -10, 20], degrees=True).as_matrix(), [0.01, -0.02, 0.005])
O_T_F = T(R.random(random_state=rng).as_matrix(), [0.5, 0, 0.3])
tcp_now = O_T_F @ F_T_TCP
tcp_next = tcp_now @ dT                                   # what message_comms sends (absolute TCP target)
flange_next = tcp_next @ inv(F_T_TCP)                     # what franka_control does internally
assert np.allclose(flange_next @ F_T_cam @ T_cam_tcp, tcp_next, atol=1e-9)
print("\nAll checks passed: the triangle closes, the rig-file form round-trips, and training and robot describe the")
print("same fingertip point and axes for any arm pose and any policy motion.")

# --- what the CURRENT stand-in rig file implies ---
T_cam_tcp_standin = T(R.from_euler("xyz", [15, 0, 0], degrees=True).as_matrix(), [0.0, 0.060, 0.200])
F_T_cam_implied = F_T_TCP @ inv(T_cam_tcp_standin)
print("\nCurrent stand-in T_cam_tcp (15 deg about x, (0, 0.06, 0.20) m) implies this camera pose on the flange:")
print("  position (m):", np.round(F_T_cam_implied[:3, 3], 4), " camera z axis in flange:", np.round(F_T_cam_implied[:3, 2], 3))
