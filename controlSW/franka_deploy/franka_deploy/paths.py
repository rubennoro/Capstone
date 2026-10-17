"""Path guards: upstream detached_inference writes `<ckpt>.yaml` next to the checkpoint path it is given and
`tmp_vis/` into the cwd. Those writes must never land in the repo's protected or result areas."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Nothing in franka_deploy may write work/log/cache files anywhere inside the repository (incl. the repo root,
# results/, data/, sensor_rig/ and controlSW/ itself). Work dirs belong outside the repo (e.g. a scratch dir).
FORBIDDEN_WRITE_DIRS = (REPO_ROOT,)


class ForbiddenPathError(ValueError):
    pass


def _is_within(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def check_writable_dir(path: str | os.PathLike, what: str = "directory") -> Path:
    """Absolute, non-resolved AND resolved forms must both be outside the repository (FORBIDDEN_WRITE_DIRS)."""
    p = Path(os.path.abspath(path))
    for cand in (p, p.resolve()):
        for root in FORBIDDEN_WRITE_DIRS:
            if _is_within(cand, root.resolve()) or _is_within(cand, root):
                raise ForbiddenPathError(f"refusing to use {what} {p}: it is inside {root} (no writes allowed there)")
    return p


def stage_checkpoint(ckpt: str | os.PathLike, work_dir: str | os.PathLike) -> Path:
    """Return `<work_dir>/ckpt/<name>.ckpt`, a symlink to `ckpt`, so the server's `<ckpt>.yaml` lands in work_dir.

    The checkpoint itself is only read. An existing link must already point to the same file.
    """
    src = Path(ckpt)
    if not str(src).endswith(".ckpt"):
        raise ValueError(f"checkpoint must be a .ckpt file, got {src}")
    if not src.resolve().is_file():
        raise FileNotFoundError(src)
    work = check_writable_dir(work_dir, "work dir")
    link_dir = work / "ckpt"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / src.name
    if Path(os.path.abspath(src)).parent == link_dir:  # already staged
        return link
    if link.is_symlink() or link.exists():
        if link.resolve() != src.resolve():
            raise FileExistsError(f"{link} exists and points to {link.resolve()}, not {src.resolve()}")
    else:
        link.symlink_to(src.resolve())
    check_writable_dir(link.parent, "checkpoint staging dir")
    return link
