"""Filesystem plumbing shared by the ``dataproc`` commands: durable writes, moves that never copy, small checks.

Nothing here knows about episodes or tasks. Every function either succeeds completely or raises; the only
best-effort calls are :func:`fsync_dir` and :func:`rmdir_if_empty`, which say so.

- :func:`write_atomic`: dot temp file in the same directory + fsync + rename + fsync of the directory.
- :func:`rename_no_copy`: ``os.rename`` that raises :class:`CrossDeviceError` on ``EXDEV`` instead of copying.
- :class:`FileLock`: non-blocking exclusive ``flock`` on a lock file.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
from pathlib import Path

__all__ = [
    "CrossDeviceError", "FileLock", "EFFECTIVE_IDS", "sha256_file", "is_regular_file", "fsync_dir", "tmp_path",
    "write_atomic", "rmdir_if_empty", "rename_no_copy", "free_name", "nearest_existing", "not_writable",
    "checked_uid",
]


class CrossDeviceError(OSError):
    """``os.rename`` failed with ``EXDEV``: source and destination are on different filesystems."""

    def __init__(self, src: Path, dst: Path):
        super().__init__(errno.EXDEV, os.strerror(errno.EXDEV))
        self.src, self.dst = src, dst


# --------------------------------------------------------------------------------------------- reading

def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_regular_file(path: Path) -> bool:
    """True if ``path`` itself (not a symlink target) is a regular file."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


# --------------------------------------------------------------------------------------------- durable writes

def fsync_dir(path: Path) -> None:
    """Best effort: make a rename/creation in ``path`` durable."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def tmp_path(path: Path) -> Path:
    """The dot temp file an atomic write of ``path`` goes through: ``.<name>.tmp-<pid>`` next to it."""
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a dot temp file in the same directory + fsync + rename."""
    tmp = tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)
    fsync_dir(path.parent)


def rmdir_if_empty(d: Path) -> None:
    """Remove a directory only if it is empty (``os.rmdir``); any failure is ignored. Never removes data."""
    try:
        os.rmdir(d)
    except OSError:
        pass


def rename_no_copy(src: Path, dst: Path) -> None:
    """``os.rename`` that never falls back to copying (``EXDEV`` -> :class:`CrossDeviceError`), then fsyncs both
    parent directories."""
    try:
        os.rename(src, dst)
    except OSError as e:
        if e.errno == errno.EXDEV:
            raise CrossDeviceError(src, dst) from None
        raise
    fsync_dir(dst.parent)
    fsync_dir(src.parent)


# --------------------------------------------------------------------------------------------- names and paths

def free_name(parent: Path, name: str) -> Path:
    """``parent/name``, or ``name__2``, ``name__3``, ... if taken (any entry counts)."""
    p, k = parent / name, 2
    while os.path.lexists(p):
        p, k = parent / f"{name}__{k}", k + 1
    return p


def nearest_existing(d: Path) -> Path:
    """``d``, or its nearest existing ancestor (where ``d`` would be created)."""
    while not d.exists() and d.parent != d:
        d = d.parent
    return d


# --------------------------------------------------------------------------------------------- permissions

#: os.access checks the effective ids where the platform supports it (Linux does), else the real ones.
EFFECTIVE_IDS = os.access in os.supports_effective_ids


def not_writable(d: Path) -> str | None:
    """None if this process can create/rename entries in ``d`` (W_OK | X_OK), else a description of the dir."""
    if os.access(d, os.W_OK | os.X_OK, effective_ids=EFFECTIVE_IDS):
        return None
    try:
        st = os.stat(d)
        return f"owner uid {st.st_uid}, mode {stat.filemode(st.st_mode)}"
    except OSError as e:
        return str(e)


def checked_uid() -> str:
    """The uid :func:`not_writable` checks against, for messages."""
    return f"effective uid {os.geteuid()}" if EFFECTIVE_IDS else f"real uid {os.getuid()}"


# --------------------------------------------------------------------------------------------- locking

class FileLock:
    """Non-blocking exclusive ``flock`` on ``path`` (created ``0o644`` if missing; never deleted).

    :meth:`acquire` raises ``BlockingIOError`` if another process holds it. Also usable as a context manager.
    """

    def __init__(self, path: Path):
        self.path, self.fd = path, None

    def acquire(self) -> None:
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            raise

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
