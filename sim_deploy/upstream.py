"""The upstream deployment branch the sim compares itself with (read only): ONE place for its worktree and ref.

SIM_DEPLOY_UPSTREAM_REPO  the git worktree to read from (default ~/Capstone-final-deployment; when it does not exist,
                          this checkout is used: the ref is then looked up in its git, which shares the branches)
SIM_DEPLOY_UPSTREAM_REF   the branch/commit (default rnoro/final-deployment)

Users: scene/start_pose.py (compare_git, from_git, `scene start-pose --from-git`), run.sh's start-pose line (through
compare_git), ui/server.py (through compare_git), offline/deploy_path.py (the f632576 gripper_map fallback)."""

import os
from pathlib import Path

from sim_deploy import REPO

DEFAULT_REPO = "~/Capstone-final-deployment"
DEFAULT_REF = "rnoro/final-deployment"


def repo_setting() -> Path:
    """The configured upstream worktree (expanded; may not exist)."""
    return Path(os.environ.get("SIM_DEPLOY_UPSTREAM_REPO") or DEFAULT_REPO).expanduser()


def repo() -> Path:
    """The git directory to read upstream from: the configured worktree when it exists, else this checkout."""
    p = repo_setting()
    return p if p.is_dir() else REPO


def ref() -> str:
    return os.environ.get("SIM_DEPLOY_UPSTREAM_REF") or DEFAULT_REF


def describe() -> str:
    """'REF in REPO' for log lines."""
    p = repo_setting()
    where = str(p) if p.is_dir() else f"{REPO} ({p} not found)"
    return f"{ref()} in {where}"
