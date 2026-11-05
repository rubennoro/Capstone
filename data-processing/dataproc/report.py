"""Summaries shared by ``process`` (report.json, the printed table) and ``status``: one episode's state from its
``status.json``, and a plain-text table."""

from __future__ import annotations

__all__ = ["STAGE_NAMES", "episode_summary", "format_table"]

#: Stage names in processing order (== ``EpisodeProcessor.STAGES`` names; checked by a test).
STAGE_NAMES = ("check", "slam", "grid", "pose", "width", "validate", "images", "export")


def episode_summary(status: dict) -> dict:
    """``{status, last_stage, stages, reasons, n_steps}`` of one episode's ``status.json``.

    ``status``: ``exported`` | the first ``rejected`` / ``failed`` stage's status | ``pending`` (nothing recorded)
    | ``incomplete``. ``last_stage``: the last stage that ran (status not ``skipped``). ``reasons``:
    ``"<stage>: <reason code>"`` of every rejected / failed stage, plus its message when that says more."""
    stages = {s: status.get(s, {}).get("status", "pending") for s in STAGE_NAMES}
    reasons, state = [], None
    for s in STAGE_NAMES:
        if stages[s] in ("rejected", "failed"):
            rec = status[s]
            reasons += [f"{s}: {r}" for r in (rec.get("reasons") or [rec.get("reason") or stages[s]])]
            if rec.get("reason") and rec.get("reasons") and rec["reason"] != ", ".join(rec["reasons"]):
                reasons.append(f"{s}: {rec['reason']}")
            state = state or stages[s]
    if stages["export"] == "done":
        state = "exported"
    elif state is None:
        state = "pending" if all(v == "pending" for v in stages.values()) else "incomplete"
    ran = [s for s in STAGE_NAMES if stages[s] not in ("pending", "skipped")]
    n_steps = status.get("export", {}).get("outputs", {}).get("n_steps") if stages["export"] == "done" else None
    return {"status": state, "last_stage": ran[-1] if ran else None, "stages": stages, "reasons": reasons,
            "n_steps": n_steps}


def format_table(rows: list[list], header: list[str]) -> str:
    rows = [[str(c) for c in r] for r in rows]
    w = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(header)]

    def fmt(r):
        return "  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip()

    return "\n".join([fmt(header), fmt(["-" * x for x in w])] + [fmt(r) for r in rows])
