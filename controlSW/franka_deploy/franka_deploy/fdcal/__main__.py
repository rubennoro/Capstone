"""python -m franka_deploy fdcal {handeye|pivot|width|capture} ...  (named fdcal so it never collides with sensor_rig's calib-*)"""
import importlib
import sys


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("handeye", "pivot", "width", "capture"):
        print(__doc__)
        return 1
    return importlib.import_module(f"franka_deploy.fdcal.{argv[0]}").main(argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())
