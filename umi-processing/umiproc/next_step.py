"""The ``Next:`` block every umiproc command ends its output with: the next command in DIRECTIONS.md's order, ready
to copy. A blank line, ``Next: <sentence> (DIRECTIONS.md section N)``, then at most two commands indented by two
spaces. ``--source``/``--root`` are carried over when the user passed non-default values, so the next command works
on the laptop too; recordings not made yet are DIRECTIONS.md's placeholders (``<MAP_EPISODE>``, ...)."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

__all__ = ["DEFAULT_SOURCE", "carried", "command", "block", "copy_command"]

DEFAULT_SOURCE = "/data/incoming"
_PLACEHOLDER = re.compile(r"<[A-Za-z_]+>")


def _abs(p) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(p))))


def carried(args, *, source: bool = True, root: bool = True) -> list[str]:
    """``--source DIR`` / ``--root DIR`` of ``args`` when they are not the defaults (absolute paths)."""
    from umiproc.paths import DEFAULT_ROOT

    out = []
    s = getattr(args, "source", None) if source else None
    if s is not None and _abs(s) != DEFAULT_SOURCE:
        out += ["--source", _abs(s)]
    r = getattr(args, "root", None) if root else None
    if r is not None and _abs(r) != _abs(DEFAULT_ROOT):
        out += ["--root", _abs(r)]
    return out


def command(*parts) -> str:
    """The parts joined for a shell; placeholders like ``<FIRST_DEMO>`` stay unquoted."""
    return " ".join(str(p) if _PLACEHOLDER.fullmatch(str(p)) else shlex.quote(str(p)) for p in parts)


def block(sentence: str, section: str, *commands: str) -> str:
    assert len(commands) <= 2
    return "\n".join(["", f"Next: {sentence} (DIRECTIONS.md section {section})"] + [f"  {c}" for c in commands])


def copy_command(root, folder: str) -> str:
    """DIRECTIONS.md 8.2's rsync, run on the laptop, for ``<root>/tasks/<folder>``."""
    root = Path(_abs(root))
    try:
        remote = str(root.relative_to(Path.home()))
    except ValueError:
        remote = str(root)
    return f"rsync -av --exclude .staging <jetson>:{remote}/tasks/{folder}/ ~/Capstone/data/{folder}/"
