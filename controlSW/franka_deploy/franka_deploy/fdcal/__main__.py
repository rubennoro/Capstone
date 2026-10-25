"""python -m franka_deploy fdcal {handeye|pivot|width|capture|stream-intrinsics} ...  (named fdcal so it never collides with sensor_rig's calib-*)

    capture handeye|pivot|width|stream   data capture (hardware day)
    handeye | pivot                      analysis of the captured data
    width [--linear]                     gripper width map (servo table, or the Franka Hand's linear a, b)
    stream-intrinsics                    ChArUco K of the deployment stream vs the station K and the training variants
"""
import importlib
import sys

COMMANDS = {"handeye": "handeye", "pivot": "pivot", "width": "width", "capture": "capture",
            "stream-intrinsics": "stream_intrinsics"}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 1
    return importlib.import_module(f"franka_deploy.fdcal.{COMMANDS[argv[0]]}").main(argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())
