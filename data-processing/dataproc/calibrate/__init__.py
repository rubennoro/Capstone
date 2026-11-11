"""``dataproc calibrate TASK EPISODE_DIR``: fit and check the gripper calibration of one Pi recording, then (only
if it passes) create a new task folder holding it. ``task``: the steps; ``diagnostics``: failure messages and
hints; ``cli``: printing and exit codes."""

from dataproc.calibrate.task import CalibrateError, CalibrateResult, calibrate_task

__all__ = ["calibrate_task", "CalibrateResult", "CalibrateError"]
