"""Rig session names (ROADMAP §0.3): one shared check for collect, record and process-session.

The session is baked into every episode NAME at record start, so a name that process-session or the cluster would
refuse must be refused before the sensors open, with the same message everywhere:

* rig-only: the name starts with ``rig`` (e.g. ``rig1020``), so rig data can never mix with iPhone sessions in
  iPhUMI's shared ``tmp_demonstrations`` cache or in ``~/CapstoneProject/datasets``;
* unused: no iPhone episode in ``data/raw`` (checked only where that folder exists, i.e. in the repo on the laptop
  or cluster; on the Jetson it usually does not).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_IPHONE_RAW = REPO / "data" / "raw"


def session_name_problems(session: str, iphone_raw: Path | None = DEFAULT_IPHONE_RAW) -> list[str]:
    """Empty if ``session`` (already sanitised) is a valid rig session name."""
    out = []
    if not session.startswith("rig"):
        out.append(f"session {session!r} is not a rig-only name: rig sessions must start with 'rig' (e.g. rig1020) "
                   "so they can never mix with iPhone sessions in tmp_demonstrations or on the cluster")
    if iphone_raw is not None and Path(iphone_raw).is_dir():
        hits = [p for p in Path(iphone_raw).glob("*/*") if len(p.name.split("_")) == 5
                and p.name.split("_")[2] == session]
        if hits:
            out.append(f"session {session!r} already exists in {iphone_raw} ({len(hits)} episodes, e.g. "
                       f"{hits[0].name}): pick another name")
    return out
