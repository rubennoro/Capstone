"""Pluggable SLAM backends (interface and conventions: :mod:`dataproc.slam.base`).

``get_backend("oracle" | "orbslam3", **options)`` returns a :class:`~dataproc.slam.base.SlamBackend`.
"""

from __future__ import annotations

from dataproc.slam.base import SlamBackend, SlamError, SlamResult

__all__ = ["BACKENDS", "get_backend", "SlamBackend", "SlamError", "SlamResult"]


def _oracle(**opts):
    from dataproc.slam.oracle import OracleBackend
    return OracleBackend(**opts)


def _orbslam3(**opts):
    from dataproc.slam.orbslam3 import Orbslam3Backend
    return Orbslam3Backend(**opts)


BACKENDS = {"oracle": _oracle, "orbslam3": _orbslam3}


def get_backend(name: str, **opts) -> SlamBackend:
    """A backend instance by name; ``opts`` go to its constructor (unknown options raise TypeError)."""
    if name not in BACKENDS:
        raise ValueError(f"unknown SLAM backend {name!r}; expected one of {sorted(BACKENDS)}")
    return BACKENDS[name](**opts)
