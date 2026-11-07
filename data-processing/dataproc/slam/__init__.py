"""Pluggable SLAM backends (interface and conventions: :mod:`dataproc.slam.base`).

``get_backend("oracle" | "orbslam3" | "none", **options)`` returns a :class:`~dataproc.slam.base.SlamBackend`.
``none`` is the placeholder used until a real backend is installed: the slam stage stops every episode with
``slam_unavailable``. ``resolve_backend_name("auto")`` picks ``orbslam3`` when its native build is present, else
``none``.
"""

from __future__ import annotations

from dataproc.slam.base import SlamBackend, SlamError, SlamResult

__all__ = ["BACKENDS", "NONE", "AUTO", "BACKEND_CHOICES", "get_backend", "orbslam3_available",
           "resolve_backend_name", "NoSlamBackend", "SlamBackend", "SlamError", "SlamResult"]

NONE = "none"
AUTO = "auto"


class NoSlamBackend(SlamBackend):
    """``--slam none``: no SLAM installed yet. The slam stage turns this into ``slam_unavailable`` before calling
    :meth:`run`; calling it anyway raises :class:`SlamError`."""

    name = NONE

    def run(self, episode, rig, work_dir) -> SlamResult:
        raise SlamError("no SLAM backend installed (--slam none); build ORB-SLAM3 (docs/ORBSLAM3.md) and re-run")


def _oracle(**opts):
    from dataproc.slam.oracle import OracleBackend
    return OracleBackend(**opts)


def _orbslam3(**opts):
    from dataproc.slam.orbslam3 import Orbslam3Backend
    return Orbslam3Backend(**opts)


BACKENDS = {"oracle": _oracle, "orbslam3": _orbslam3, NONE: lambda **opts: NoSlamBackend(**opts)}
#: What ``--slam`` accepts: every backend name plus ``auto``.
BACKEND_CHOICES = ("auto", "orbslam3", "oracle", NONE)


def orbslam3_available(orb_slam3_dir=None) -> bool:
    """True if the native ORB-SLAM3 build (``mono_inertial_euroc`` + ``ORBvoc.txt``) exists in ``orb_slam3_dir``
    (default ``$ORB_SLAM3_DIR`` or ``~/ORB_SLAM3``)."""
    import os
    from pathlib import Path

    from dataproc.slam.orbslam3 import ORB_BIN, ORB_VOC, default_orb_slam3_dir

    d = Path(orb_slam3_dir).expanduser() if orb_slam3_dir else default_orb_slam3_dir()
    exe = d / ORB_BIN
    return exe.is_file() and os.access(exe, os.X_OK) and (d / ORB_VOC).is_file()


def resolve_backend_name(name: str) -> str:
    """``auto`` -> ``orbslam3`` if :func:`orbslam3_available`, else ``none``; any other name is returned as is."""
    if name != AUTO:
        return name
    return "orbslam3" if orbslam3_available() else NONE


def get_backend(name: str, **opts) -> SlamBackend:
    """A backend instance by name; ``opts`` go to its constructor (unknown options raise TypeError)."""
    if name not in BACKENDS:
        raise ValueError(f"unknown SLAM backend {name!r}; expected one of {sorted(BACKENDS)}")
    return BACKENDS[name](**opts)
