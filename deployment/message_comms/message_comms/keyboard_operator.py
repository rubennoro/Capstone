"""KeyboardOperator: single keys from the terminal (cbreak mode), or first letters of lines from a pipe.

main-sim stop_watcher.py / _input_thread were the reference only (STRUCTURE.md 4.9).
"""
from __future__ import annotations

import atexit
import queue
import threading
from typing import TextIO

from message_comms.operator import KEY_LIST, Key, Operator

_KEYS = {key.value: key for key in Key}


class KeyboardOperator(Operator):
    def __init__(self, stdin: TextIO, stdout: TextIO) -> None:
        super().__init__(stdout)
        self._in = stdin
        self._keys: queue.Queue[str] = queue.Queue()
        self._saved_attrs = None
        self._tty = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self._tty = self._in.isatty()
        except (AttributeError, ValueError):
            self._tty = False
        if self._tty:
            import termios
            import tty
            fd = self._in.fileno()
            self._saved_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            atexit.register(self.close)
        self._thread = threading.Thread(target=self._read, name="keyboard", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._saved_attrs is not None:
            import termios
            attrs, self._saved_attrs = self._saved_attrs, None
            try:
                termios.tcsetattr(self._in.fileno(), termios.TCSADRAIN, attrs)
            except (OSError, ValueError):
                pass
            self._write("\n")

    def poll_key(self, progress) -> Key | None:
        while True:
            try:
                char = self._keys.get_nowait()
            except queue.Empty:
                return None
            if char.isspace():
                continue
            key = _KEYS.get(char.lower())
            if key is None:
                self.show_message(f"{char!r}: not a key. {KEY_LIST}")
                continue
            return key

    def _read(self) -> None:
        try:
            while True:
                if self._tty:
                    char = self._in.read(1)
                    if not char:
                        return
                    self._keys.put(char)
                else:
                    line = self._in.readline()
                    if not line:
                        return
                    line = line.strip()
                    if line:
                        self._keys.put(line[0])
        except (OSError, ValueError):
            return
