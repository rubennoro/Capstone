"""ScriptedOperator: keys from a YAML script instead of the keyboard (tests, milestone 3; STRUCTURE.md 4.9)."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import yaml

from message_comms.episode_loop import LoopProgress, Phase
from message_comms.errors import ScriptError
from message_comms.operator import Key, Operator

KINDS = ("key", "wait_phase", "wait_steps", "wait_s")
NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class ScriptEntry:
    kind: str                     # one of KINDS
    value: object                 # Key | phase name | int | float
    line: int                     # line in the script file (1-based), for messages


class ScriptedOperator(Operator):
    def __init__(self, entries: Sequence[ScriptEntry], stdout: TextIO, clock: Callable[[], int]) -> None:
        super().__init__(stdout)
        self._entries = list(entries)
        self._clock = clock
        self._index = 0
        self._wait_since: int | None = None

    @classmethod
    def load(cls, path: Path, stdout: TextIO, clock: Callable[[], int]) -> ScriptedOperator:
        """Raises ScriptError naming the line."""
        try:
            text = Path(path).read_text()
        except OSError as exc:
            raise ScriptError(f"{path}: cannot read: {exc}") from exc
        return cls(parse_script(text, str(path)), stdout, clock)

    def done(self) -> bool:
        return self._index >= len(self._entries)

    def poll_key(self, progress: LoopProgress) -> Key | None:
        while self._index < len(self._entries):
            entry = self._entries[self._index]
            if entry.kind == "key":
                self._index += 1
                return entry.value
            if entry.kind == "wait_phase":
                if progress.phase.value != entry.value:
                    return None
            elif entry.kind == "wait_steps":
                if progress.phase is not Phase.RUNNING or progress.steps_sent < entry.value:
                    return None
            else:                                           # wait_s, by the injected clock
                if self._wait_since is None:
                    self._wait_since = self._clock()
                if self._clock() - self._wait_since < entry.value * NS_PER_S:
                    return None
                self._wait_since = None
            self._index += 1
        return None


def parse_script(text: str, name: str) -> list[ScriptEntry]:
    loader = yaml.SafeLoader(text)
    try:
        root = loader.get_single_node()
        if root is None or not isinstance(root, yaml.SequenceNode):
            raise ScriptError(f"{name}: a script is a YAML list of entries")
        entries = [_entry(loader, node, name) for node in root.value]
    except yaml.YAMLError as exc:
        raise ScriptError(f"{name}: not valid YAML: {exc}") from exc
    finally:
        loader.dispose()
    if not entries or entries[-1].kind != "key" or entries[-1].value is not Key.QUIT:
        line = entries[-1].line if entries else 1
        raise ScriptError(f"{name}:{line}: the last entry must be {{key: q}} (s only ends an episode)")
    return entries


def _entry(loader: yaml.SafeLoader, node: yaml.Node, name: str) -> ScriptEntry:
    line = node.start_mark.line + 1
    where = f"{name}:{line}"
    data = loader.construct_object(node, deep=True)
    if not isinstance(data, dict) or len(data) != 1:
        raise ScriptError(f"{where}: an entry is a map with one key, one of {', '.join(KINDS)}: got {data!r}")
    (kind, value), = data.items()
    if kind not in KINDS:
        raise ScriptError(f"{where}: unknown entry {kind!r} (expected one of {', '.join(KINDS)})")
    if kind == "key":
        keys = {key.value: key for key in Key}
        if value not in keys:
            raise ScriptError(f"{where}: unknown key {value!r} (expected one of {', '.join(keys)})")
        value = keys[value]
    elif kind == "wait_phase":
        phases = [phase.value for phase in Phase]
        if value not in phases:
            raise ScriptError(f"{where}: unknown phase {value!r} (expected one of {', '.join(phases)})")
    elif kind == "wait_steps":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ScriptError(f"{where}: wait_steps needs an integer >= 0, got {value!r}")
    else:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ScriptError(f"{where}: wait_s needs a number >= 0, got {value!r}")
        value = float(value)
    return ScriptEntry(kind, value, line)
