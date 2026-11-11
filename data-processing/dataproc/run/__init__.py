"""``dataproc run TASK_FOLDER START_DIR END_DIR``: process the Pi recordings START..END for a calibrated task folder,
one at a time; only the ones that pass every stage become ``episode_N``. ``plan``: the range and what happens to
each recording (``--dry-run``); ``staging``: the ``.staging/`` and ``failed/`` moves and crash repair; ``episode``:
one recording; ``range``: the whole run; ``cli``: printing and exit codes."""

from dataproc.run.plan import RunError
from dataproc.run.range import EXIT_FAILED, EXIT_OK, EXIT_WAITING_FOR_SLAM, RunResult, run_range

__all__ = ["run_range", "RunResult", "RunError", "EXIT_OK", "EXIT_FAILED", "EXIT_WAITING_FOR_SLAM"]
