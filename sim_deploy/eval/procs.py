"""Find and stop the processes a sim run started, whatever session or process group they moved to.

Why: the driver starts `policy_deploy run` through pexpect, which puts the launcher in a new session (pty); the
launcher starts frames and policy with ``start_new_session=True`` (deployment/policy_deploy launcher/managed_process.py).
None of them is in run.sh's process group, so a group signal (batch.py stop_group, Ctrl-C) never reaches them; only
the launcher's own stop (on q / SIGINT / SIGTERM) ends frames and policy. A SIGHUP (pexpect terminate, or the pty
closing when the driver dies) or SIGKILL ends the launcher without that stop, and frames / policy are left running
(reparented to the user's systemd), holding their ports (frames: 5610).

Two ways to find them anyway:
- ``tree(pid)``: a process and its descendants by ppid (while the parent is alive);
- ``tagged(tag)``: every process whose environment has ``SIM_DEPLOY_RUN_TAG=<tag>``. run.sh exports the tag (batch.py
  sets one per run.sh call); every process started under run.sh inherits it (pexpect and the launcher pass the
  environment on), including those reparented after their parent died.

    python -m sim_deploy.eval.procs stop-tagged <TAG> [--grace 10]     (run.sh's EXIT trap)
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

TAG_VAR = "SIM_DEPLOY_RUN_TAG"


def _stat(pid: int) -> tuple[int, int] | None:
    """(ppid, start time in clock ticks) of a live pid, or None."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            s = f.read().decode("utf-8", "replace")
        rest = s[s.rfind(")") + 2:].split()
        return int(rest[1]), int(rest[19])
    except (OSError, ValueError, IndexError):
        return None


def _pids() -> list[int]:
    return [int(d) for d in os.listdir("/proc") if d.isdigit()]


def ancestors(pid: int | None = None) -> set[int]:
    """pid (default: this process) and its parents up to 1."""
    out, p = set(), os.getpid() if pid is None else pid
    while p > 1 and p not in out:
        out.add(p)
        st = _stat(p)
        if st is None:
            break
        p = st[0]
    return out


def tree(root: int) -> list[tuple[int, int]]:
    """[(pid, start time)] of root and every descendant (by ppid), any session."""
    kids: dict[int, list[int]] = {}
    start: dict[int, int] = {}
    for pid in _pids():
        st = _stat(pid)
        if st is not None:
            kids.setdefault(st[0], []).append(pid)
            start[pid] = st[1]
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        if pid in start:
            out.append((pid, start[pid]))
        todo += kids.get(pid, [])
    return out


def tagged(tag: str) -> list[tuple[int, int]]:
    """[(pid, start time)] of this user's processes with SIM_DEPLOY_RUN_TAG=<tag>, except this process and its
    ancestors (the caller: run.sh, batch)."""
    if not tag:
        return []
    want = f"{TAG_VAR}={tag}".encode()
    skip = ancestors()
    out = []
    for pid in _pids():
        if pid in skip:
            continue
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                env = f.read().split(b"\0")
        except OSError:                                  # another user's process, or gone
            continue
        if want in env:
            st = _stat(pid)
            if st is not None:
                out.append((pid, st[1]))
    return out


def _alive(pid: int, start: int) -> bool:
    st = _stat(pid)
    return st is not None and st[1] == start                     # same start time: not a reused pid


def _cmd(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return " ".join(f.read().decode("utf-8", "replace").split("\0")).strip()[:120]
    except OSError:
        return "?"


def stop(procs: list[tuple[int, int]], grace_s: float = 10.0, first: signal.Signals = signal.SIGTERM) -> list[str]:
    """``first`` (SIGTERM: policy_deploy's launcher then stops frames / policy itself) to every live process of
    procs, up to grace_s for them to exit, then SIGKILL. Returns one line per process that was still running."""
    live = [(p, s) for p, s in procs if _alive(p, s)]
    done = [f"{p} {_cmd(p)}" for p, _ in live]
    for p, _ in live:
        try:
            os.kill(p, first)
        except (ProcessLookupError, PermissionError):
            pass
    end = time.monotonic() + grace_s
    while time.monotonic() < end and any(_alive(p, s) for p, s in live):
        time.sleep(0.1)
    for p, s in live:
        if _alive(p, s):
            try:
                os.kill(p, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return done


def stop_tagged(tag: str, grace_s: float = 10.0) -> list[str]:
    return stop(tagged(tag), grace_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.eval.procs", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stop-tagged", help=f"stop every process with {TAG_VAR}=<tag> (not this one's ancestors)")
    p.add_argument("tag")
    p.add_argument("--grace", type=float, default=10.0, help="s between SIGTERM and SIGKILL")
    p.add_argument("--prefix", default="[run.sh]", help="the printed lines' prefix")
    a = ap.parse_args(argv)
    for line in stop_tagged(a.tag, a.grace):
        print(f"{a.prefix} still running after the stop, ended: {line}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
