"""python -m sim_deploy.evaltool {score,report,frames,prune-index,import-real,migrate-rollouts} ...   (README.md)"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _score(a) -> int:
    from sim_deploy.evaltool.score import write_score
    import json
    rc = 0
    for d in a.eval_dir:
        try:
            p = write_score(Path(d))
        except Exception as e:                            # noqa: BLE001 - one bad folder must not stop the rest
            print(f"[evaltool] {d}: {type(e).__name__}: {e}", file=sys.stderr)
            rc = 2
            continue
        s = json.loads(p.read_text())["summary"]
        su = s["success"]
        print(f"{p}: success {su['k']}/{su['n']} (CI {su['ci95'][0]:.2f}-{su['ci95'][1]:.2f}), "
              f"lift without grasp {s['lift_without_grasp']['k']}, {s['invalid']} invalid, "
              f"{s['unlabelled']} unlabelled, missing {s['missing']}")
    return rc


def _report(a) -> int:
    from sim_deploy.evaltool.report import SIM_EVALS, build_report
    out = Path(a.out) if a.out else SIM_EVALS / "_reports" / "report"
    res = build_report(Path(a.root), a.select, a.batch, a.protocol, [g for g in a.group.split(",") if g], out)
    print(f"[evaltool] {res['evals']} evals, {res['rollouts']} rollouts ({res['invalid']} invalid, "
          f"{res['lift_without_grasp']} lift without grasp) -> {res['out']}")
    return 0


def _frames(a) -> int:
    from sim_deploy.evaltool.frames import frames_eval
    out = frames_eval(Path(a.eval_dir), a.rollout)
    for p in out:
        print(p)
    if not out:
        print("[evaltool] no rollout with external.mp4", file=sys.stderr)
        return 1
    return 0


def _prune_index(a) -> int:
    """Drop sim_evals/index.csv rows whose eval_dir no longer exists (deleted eval folders); backup in _logs/ first."""
    import csv
    import datetime as dt
    import shutil
    from sim_deploy.evaltool.report import SIM_EVALS
    idx = SIM_EVALS / "index.csv"
    with idx.open(newline="") as f:
        r = csv.DictReader(f)
        fields, rows = r.fieldnames, list(r)
    gone = [x for x in rows if not (SIM_EVALS / x.get("eval_dir", "")).is_dir() or not x.get("eval_dir")]
    for x in gone:
        print(f"{'would drop' if a.dry_run else 'drop'}: {x.get('eval_dir')}")
    if a.dry_run or not gone:
        print(f"[evaltool] {len(gone)} of {len(rows)} rows point to missing eval folders" + (" (dry run)" if a.dry_run else ""))
        return 0
    bak = SIM_EVALS / "_logs" / f"index_before_prune_{dt.datetime.now():%Y-%m-%d_%H%M%S}.csv"
    bak.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(idx, bak)
    keep = [x for x in rows if x not in gone]
    tmp = idx.with_name("index.csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    tmp.replace(idx)
    print(f"[evaltool] dropped {len(gone)} of {len(rows)} rows; backup {bak}")
    return 0


def _import_real(rest: list) -> int:
    from sim_deploy.evaltool.import_real import main
    return main(rest) or 0


def _migrate_rollouts(rest: list) -> int:
    from sim_deploy.evaltool.migrate_rollouts import main
    return main(rest) or 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "import-real":                 # agent D's tool takes its own arguments
        return _import_real(argv[1:])
    if argv and argv[0] == "migrate-rollouts":            # one-off (SCHEMA.md "Versions"), its own arguments
        return _migrate_rollouts(argv[1:])
    from sim_deploy.evaltool.report import SIM_EVALS
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.evaltool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score", help="write <eval>/score.json")
    p.add_argument("eval_dir", nargs="+")
    p.set_defaults(fn=_score)
    p = sub.add_parser("report", help="cross-eval report into --out")
    p.add_argument("--select", default=None, help="regex searched in eval.yaml tag")
    p.add_argument("--batch", default=None, help="eval.yaml batch_id")
    p.add_argument("--protocol", default=None, help="eval.yaml protocol_sha (prefix)")
    p.add_argument("--group", default="checkpoint,kind,look",
                   help="comma list of checkpoint, kind, look, origin, tag, batch, protocol")
    p.add_argument("--out", default=None, help="output folder (default sim_evals/_reports/report)")
    p.add_argument("--root", default=str(SIM_EVALS), help="where to look for eval folders")
    p.set_defaults(fn=_report)
    p = sub.add_parser("frames", help="key-frame strip into rollouts/NNN/frames.jpg")
    p.add_argument("eval_dir")
    p.add_argument("--rollout", type=int, default=None)
    p.set_defaults(fn=_frames)
    p = sub.add_parser("prune-index", help="drop sim_evals/index.csv rows whose eval folder is gone (backup in _logs/)")
    p.add_argument("--dry-run", action="store_true", help="list the rows only")
    p.set_defaults(fn=_prune_index)
    sub.add_parser("import-real", help="import real-arm trials (import_real.py; its own --help)")
    sub.add_parser("migrate-rollouts", help="one-off: sim_evals/ episode names -> rollout names (migrate_rollouts.py; "
                   "its own --help)")
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
