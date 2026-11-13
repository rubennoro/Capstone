"""ScriptedOperator: keys from a YAML script instead of the keyboard (tests, milestone 3; STRUCTURE.md 4.9)."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import yaml

from message_comms.episode_loop import LoopProgress, Phase
from message_comms.errors import ScriptError, ScriptTimeout
from message_comms.operator import Key, Operator

KINDS = ("key", "wait_phase", "wait_steps", "wait_s")
NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class ScriptEntry:
    kind: str                     # one of KINDS
    value: object                 # Key | phase name | int | float
    line: int                     # line in the script file (1-based), for messages


class ScriptedOperator(Operator):
    """Runs the entries in order. `wait_phase`, `wait_steps` and a key that EpisodeLoop refuses each have the time
    limit `wait_limit_s` (message_comms.operator_script_wait_s, by the injected clock); past it, poll_key raises
    ScriptTimeout naming the entry, and EpisodeLoop concludes (exit 1, FATAL line). A refused key is retried as soon
    as anything the refusal depends on (phase, episode, steps sent, worker busy) has changed."""

    def __init__(self, entries: Sequence[ScriptEntry], stdout: TextIO, clock: Callable[[], int],
                 wait_limit_s: float, name: str = "script") -> None:
        super().__init__(stdout)
        self._entries = list(entries)
        self._clock = clock
        self._limit_s = float(wait_limit_s)
        self._limit_ns = int(round(wait_limit_s * NS_PER_S))
        self._name = name
        self._index = 0
        self._since: int | None = None             # when the current entry began (wait_s and the time limit)
        self._returned: tuple[int, int, tuple] | None = None   # the key just returned: (index, since, state)
        self._refused: tuple | None = None         # the state the current key was refused in
        self._hint = ""                            # the refusal's hint, for the timeout message

    @classmethod
    def load(cls, path: Path, stdout: TextIO, clock: Callable[[], int], wait_limit_s: float) -> ScriptedOperator:
        """Raises ScriptError naming the line."""
        try:
            text = Path(path).read_text()
        except OSError as exc:
            raise ScriptError(f"{path}: cannot read: {exc}") from exc
        return cls(parse_script(text, str(path)), stdout, clock, wait_limit_s, str(path))

    def done(self) -> bool:
        return self._index >= len(self._entries)

    def key_refused(self, key: Key, hint: str) -> None:
        """Go back to the key just returned: it is retried (section 4.9)."""
        if self._returned is None or self._entries[self._returned[0]].value is not key:
            return
        self._index, self._since, self._refused = self._returned
        self._returned = None
        self._hint = hint

    def poll_key(self, progress: LoopProgress) -> Key | None:
        """Raises ScriptTimeout when the current entry is past its time limit."""
        self._returned = None
        now = self._clock()
        state = (progress.phase, progress.episode_id, progress.steps_sent, progress.worker_busy)
        while self._index < len(self._entries):
            entry = self._entries[self._index]
            if self._since is None:
                self._since = now
            if entry.kind == "key":
                if self._refused is not None:                              # refused before
                    self._check_limit(entry, now, f"key {entry.value.value}: still refused ({self._hint})")
                    if state == self._refused:                             # nothing changed: no retry yet
                        return None
                self._returned = (self._index, self._since, state)
                self._refused = None
                self._index += 1
                self._since = None
                return entry.value
            if entry.kind == "wait_phase":
                if progress.phase.value != entry.value:
                    self._check_limit(entry, now, f"wait_phase {entry.value}: not reached "
                                                  f"(phase {progress.phase.value})")
                    return None
            elif entry.kind == "wait_steps":
                if progress.phase is not Phase.RUNNING or progress.steps_sent < entry.value:
                    self._check_limit(entry, now, f"wait_steps {entry.value}: not reached "
                                                  f"(phase {progress.phase.value}, {progress.steps_sent} sent)")
                    return None
            elif now - self._since < entry.value * NS_PER_S:              # wait_s, by the injected clock
                return None
            self._index += 1
            self._since = None
        return None

    def _check_limit(self, entry: ScriptEntry, now: int, what: str) -> None:
        if now - self._since >= self._limit_ns:
            raise ScriptTimeout(f"{self._name}:{entry.line}: {what} after {self._limit_s:.1f} s "
                                "(message_comms.operator_script_wait_s)")


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
