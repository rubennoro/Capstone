# adapted from rnoro/deployment:deployment/message_comms/message_comms/operator.py @ 91b178b
# changes: Key lookup through a table (KEYS_BY_CHAR, key_for); imports, logger name
"""Operator: where keys come from and where the status line goes (STRUCTURE.md 4.9)."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from policy_deploy.manager.episode_loop import LoopProgress

log = logging.getLogger("manager.operator")
CLEAR_LINE = "\x1b[K"
KEY_LIST = "keys: c start  y success  n fail  s stop  h home  q quit"


class Key(Enum):
    """The keys; EpisodeLoop dispatches them through its KEY_HANDLERS table (a later jog key: one member + one entry)."""
    START = "c"
    SUCCESS = "y"
    FAIL = "n"
    STOP = "s"
    HOME = "h"
    QUIT = "q"


KEYS_BY_CHAR: dict[str, Key] = {key.value: key for key in Key}


def key_for(char: str) -> Key | None:
    """The Key of one character (lower-cased), through the table; None if it is not a key."""
    return KEYS_BY_CHAR.get(char.lower())


class Operator(ABC):
    def __init__(self, stdout: TextIO) -> None:
        self._out = stdout
        self._status = ""

    def start(self) -> None:
        """E.g. the terminal into single-key mode."""

    def close(self) -> None:
        """Always restores the terminal."""

    @abstractmethod
    def poll_key(self, progress: LoopProgress) -> Key | None:
        """Never blocks. May raise ScriptError (a ScriptedOperator whose wait ran out): EpisodeLoop concludes."""

    def key_refused(self, key: Key, hint: str) -> None:
        """EpisodeLoop ignored the last key returned by poll_key (the hint was shown). The keyboard does nothing (the
        operator reads the hint); a ScriptedOperator retries the key."""

    def show_status(self, line: str) -> None:
        """Rewrites the one status line in place."""
        self._status = line
        self._write(f"\r{line}{CLEAR_LINE}")

    def show_message(self, text: str) -> None:
        """A full line above the status line (also logged)."""
        log.info("%s", text)
        self._write(f"\r{CLEAR_LINE}{text}\n{self._status}")

    def _write(self, text: str) -> None:
        try:
            self._out.write(text)
            self._out.flush()
        except (OSError, ValueError):          # a closed terminal must not stop the arm's shutdown
            pass
