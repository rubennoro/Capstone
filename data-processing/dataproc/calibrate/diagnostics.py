"""What ``dataproc calibrate`` prints: the tag statistics of a fit and, for a failure, why and what to check.

::

    calibrate: FAILED for episode_000001 - detection rate 0.100 < 0.2 threshold
      ultrawide frames 452; tag 0 seen 45 (10%), tag 1 seen 401 (89%), both 40 (9%; fit needs 20% ...)
      both tags at the expected depth (z 0.018-0.178 m): 40; median z 0.097 m
      open/close cycles: 5; closed 0.044 m, open 0.124 m (tag-to-tag x)
      likely causes:
        - tags are seen in only 10% of frames: improve the lighting, reduce motion blur ...

The statistics come from :func:`~dataproc.stages.width.calibration_diagnostics`; the hints from the reason codes of
:class:`~dataproc.calibrate.task.CalibrateResult` and those statistics.
"""

from __future__ import annotations

__all__ = ["format_stats", "hints", "format_failure"]


def _pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def _m(x) -> str:
    return "-" if x is None else f"{x:.3f} m"


def format_stats(diag: dict) -> list[str]:
    """The tag statistics as indented lines (empty when the tags were never looked for)."""
    if not diag:
        return []
    d = diag
    z0, z1 = d["z_window_m"]
    return [
        f"  ultrawide frames {d['n_frames']}; tag {d['left_id']} seen {d['n_left']} ({_pct(d['left_rate'])}), "
        f"tag {d['right_id']} seen {d['n_right']} ({_pct(d['right_rate'])}), both {d['n_both']} "
        f"({_pct(d['both_rate'])}; fit needs {_pct(d['tag_det_threshold'])} of frames with each tag)",
        f"  both tags at the expected depth (z {z0:.3f}-{z1:.3f} m): {d['n_in_window']}; median z "
        f"{_m(d['median_z_m'])} (aruco.nominal_z_m {d['nominal_z_m']:.3f} m)",
        f"  open/close cycles: {d['n_cycles']}; closed {_m(d['closed_width_m'])}, open {_m(d['open_width_m'])} "
        f"(tag-to-tag x; 3-D distance {_m(d['closed_distance_m'])} / {_m(d['open_distance_m'])})",
    ]


def _fit_hints(diag: dict, rig) -> list[str]:
    """Why ``fit_gripper_range`` failed, read from the statistics."""
    a = rig.aruco if rig is not None else None
    ids = f"{a.left_id}/{a.right_id}" if a else "left_id/right_id"
    dict_name = a.dict if a else "aruco.dict"
    if not diag or diag["n_left"] == 0 and diag["n_right"] == 0:
        return [f"the finger tags ({ids}) were never seen: check that both fingers are in the ultrawide view "
                f"(lens cap, camera roles), and that aruco.dict ({dict_name}) and aruco.left_id/right_id match the "
                "printed tags"]
    out = []
    for side in ("left", "right"):
        if diag[f"n_{side}"] == 0:
            out.append(f"tag {diag[f'{side}_id']} ({side} finger) was never seen while the other was: wrong "
                       f"aruco.{side}_id, or that finger is out of view / its tag is covered or damaged")
    if not out and diag["detection_rate"] < diag["tag_det_threshold"]:
        out.append(f"tags are seen in only {_pct(diag['detection_rate'])} of frames: improve the lighting, reduce "
                   "motion blur (open and close more slowly), check the lens focus")
    if not out and diag["n_in_window"] < 3:
        z0, z1 = diag["z_window_m"]
        out.append(f"the tags are not at the expected depth (median z {_m(diag['median_z_m'])}, window "
                   f"{z0:.3f}-{z1:.3f} m): fix aruco.nominal_z_m in the rig yaml, or the camera was remounted; a "
                   "wrong aruco.marker_size_m also scales z")
    return out or ["too few usable both-tag frames: record again with both fingers clearly in view"]


_SOURCE_HINTS = {
    "source_missing": "check the directory name: it must be a recording directly inside --source",
    "still_syncing": "the recording is still being recorded or transferred (COMPLETE is written last): wait, then "
                     "run the same command again",
    "manifest_failed": "files do not match MANIFEST.sha256 (a broken transfer): re-send the recording from the Pi",
    "pi_invalid": "the Pi flagged this recording: record a new calibration episode",
    "meta_unreadable": "meta.json is missing or broken: re-send the recording, or record a new one",
    "already_taken": "every recording is taken in once: record a new calibration episode for a new session",
    "cannot_convert": "the recording does not match the Pi format dataproc expects (docs/PI_ADAPTER.md)",
    "invalid_episode": "the converted recording breaks the contract (see the error codes): re-send or re-record",
    "intake": "see the intake line above (e.g. a duplicate session/episode number)",
    "exception": "unexpected error (traceback below); please report it",
}


def hints(res, rig=None) -> list[str]:
    """Likely causes for a failed :class:`~dataproc.calibrate.task.CalibrateResult`, most specific first."""
    out = []
    diag = res.fit.diagnostics if res.fit is not None else {}
    for code in res.reasons:
        if code == "gripper_range_fit":
            out += _fit_hints(diag, rig)
        elif code.startswith("rig_"):
            out.append(f"wrong --rig {res.rig_id}: the recording's cameras (resolution, fps) or rig id do not match "
                       f"rigs/{res.rig_id}.yaml; pass the rig this recorder uses")
        elif code in _SOURCE_HINTS:
            out.append(_SOURCE_HINTS[code])
    return list(dict.fromkeys(out))


def format_failure(res, rig=None) -> str:
    """The whole failure block printed on stderr."""
    lines = [f"calibrate: FAILED for {res.source_name} - {res.message}"]
    if res.fit is not None:
        lines += format_stats(res.fit.diagnostics)
        if res.fit.details.get("contract_errors"):
            lines.append("  contract errors: " + "; ".join(
                f"{e.get('code')}: {e.get('message')}" for e in res.fit.details["contract_errors"][:5]))
    causes = hints(res, rig)
    if causes:
        lines.append("  likely causes:")
        lines += [f"    - {c}" for c in causes]
    if res.fit is not None and res.fit.error_text:
        lines.append("  " + res.fit.error_text.rstrip().replace("\n", "\n  "))
    lines.append(f"  nothing was created and {res.source_name} was not taken: fix the cause and run "
                 f"`dataproc calibrate {res.task} {res.source_name}` again")
    return "\n".join(lines)
