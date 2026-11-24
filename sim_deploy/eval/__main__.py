"""python -m sim_deploy.eval run --policy <RUN_FOLDER | hold_still | replay:<zarr>#<idx>> --rollouts N --seed S
[--tag T] [--timeout 60] [--tap] [--dry-run] ...   (README.md)
python -m sim_deploy.eval batch --plan <yaml> [--resume BATCH_ID] [--dry-run]                (batch.py)
python -m sim_deploy.eval starts build|show|check [FILE]                                        (start_set.py)"""

import argparse
import sys

from sim_deploy.eval.driver import EvalError, add_arguments, main_run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_arguments(sub.add_parser("run", help="run (or --dry-run) an eval"))
    from sim_deploy.eval import batch
    batch.add_arguments(sub.add_parser("batch", help="unattended evals from a plan file, one sim at a time"))
    from sim_deploy.eval import start_set
    start_set.add_arguments(sub.add_parser("starts", help="the fixed start set (--starts set): the shared demos and "
                                           "the real-trial poses; build, show, check the file"))
    a = ap.parse_args(argv)
    try:
        if a.cmd == "starts":
            return start_set.main(a)
        return batch.main_batch(a) if a.cmd == "batch" else main_run(a)
    except (EvalError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[eval] error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
