"""load_process_class(): imports <folder>/launch.py by file path (STRUCTURE.md section 3.6).

By path, because launch.py is not part of its folder's package, which may need libraries the launcher's environment
doesn't have (torch for policy, PyAV for process_frames).
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

from deploy.errors import LaunchError
from deploy.managed_process import ManagedProcess


def load_process_class(folder: Path) -> type[ManagedProcess]:
    folder = Path(folder)
    path = folder / "launch.py"
    if not path.is_file():
        raise LaunchError(f"{folder.name}: no launch.py at {path}")
    module_name = f"launch_{folder.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise LaunchError(f"{folder.name}: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module            # so dataclasses and pickling inside launch.py find their module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                     # any error in someone else's file: report it, naming the file
        del sys.modules[module_name]
        raise LaunchError(f"{folder.name}: {path} failed to import: {type(exc).__name__}: {exc}") from exc
    cls = getattr(module, "PROCESS_CLASS", None)
    if cls is None:
        raise LaunchError(f"{folder.name}: {path} does not define PROCESS_CLASS")
    if not (inspect.isclass(cls) and issubclass(cls, ManagedProcess)):
        raise LaunchError(f"{folder.name}: PROCESS_CLASS in {path} is not a ManagedProcess subclass: {cls!r}")
    if inspect.isabstract(cls):
        missing = ", ".join(sorted(cls.__abstractmethods__))
        raise LaunchError(f"{folder.name}: PROCESS_CLASS {cls.__name__} does not write: {missing}")
    if getattr(cls, "NAME", None) != folder.name:
        raise LaunchError(f"{folder.name}: PROCESS_CLASS.NAME is {getattr(cls, 'NAME', None)!r}, "
                          f"expected the folder name {folder.name!r}")
    return cls
