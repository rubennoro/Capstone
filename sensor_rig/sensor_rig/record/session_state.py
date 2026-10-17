"""Remembers the last complete gripper calibration per (session, side, gripper_id) (P1 §3.3).

Stored in ``<raw_root>/.sensor_rig_state.json`` as ``{"<session>|<side>|<gripper_id>": "<NAME>"}``.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

STATE_FILE = ".sensor_rig_state.json"


class SessionState:
    def __init__(self, raw_root):
        self.path = Path(raw_root) / STATE_FILE
        self._lock = threading.Lock()

    @staticmethod
    def key(session: str, side: str, gripper_id: str = "default") -> str:
        return f"{session}|{side}|{gripper_id}"

    def _load(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            with open(self.path) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def get_calibration(self, session: str, side: str, gripper_id: str = "default") -> str | None:
        with self._lock:
            return self._load().get(self.key(session, side, gripper_id))

    def set_calibration(self, session: str, side: str, gripper_id: str, name: str) -> None:
        with self._lock:
            d = self._load()
            d[self.key(session, side, gripper_id)] = name
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w") as f:
                json.dump(d, f, indent=1, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self.path)

    def clear_calibration(self, session: str, side: str, gripper_id: str = "default", name: str | None = None) -> None:
        """Forget the stored calibration (only if it is ``name``, when given): used when ``collect`` discards or
        rejects the calibration episode it just recorded."""
        with self._lock:
            d = self._load()
            k = self.key(session, side, gripper_id)
            if k not in d or (name is not None and d[k] != name):
                return
            del d[k]
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w") as f:
                json.dump(d, f, indent=1, sort_keys=True)
                f.write("\n")
            os.replace(tmp, self.path)
