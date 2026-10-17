"""Calibration-episode tag check for ``collect`` (T3): are both finger ArUco tags visible?

iPhUMI's calibrate stage identifies the gripper from the tag pair ``(2g, 2g+1)`` seen in the ultrawide video and
fails below a detection threshold. After a calibration episode, ``collect`` remaps ~10 ultrawide frames to the
virtual iPhone ultrawide (``rig.remap("ultrawide", "export")``, exactly what export writes) and runs ArUco
(DICT_4X4_50, iPhUMI's ``aruco_config.yaml``). Reject only if no frame shows both tags of a pair; warn below a 0.2
fraction. Works with both cv2.aruco APIs: OpenCV < 4.7 (``detectMarkers`` + ``DetectorParameters_create``, e.g.
Ubuntu 22.04's 4.5.4) and >= 4.7 (``ArucoDetector``, e.g. JetPack 6's 4.8).
"""

from __future__ import annotations

import numpy as np

DICT_NAME = "DICT_4X4_50"
WARN_FRACTION = 0.2


def detect_marker_ids(gray: np.ndarray, aruco=None, dict_name: str = DICT_NAME) -> list[int]:
    if aruco is None:
        import cv2

        aruco = cv2.aruco
    did = getattr(aruco, dict_name)
    d = aruco.getPredefinedDictionary(did) if hasattr(aruco, "getPredefinedDictionary") else aruco.Dictionary_get(did)
    if hasattr(aruco, "ArucoDetector"):                       # OpenCV >= 4.7
        det = aruco.ArucoDetector(d, aruco.DetectorParameters())
        _, ids, _ = det.detectMarkers(gray)
    else:                                                     # OpenCV < 4.7
        params = aruco.DetectorParameters_create() if hasattr(aruco, "DetectorParameters_create") \
            else aruco.DetectorParameters()
        _, ids, _ = aruco.detectMarkers(gray, d, parameters=params)
    return [] if ids is None else sorted(int(i) for i in np.asarray(ids).ravel())


def both_tags(ids) -> bool:
    s = set(ids)
    return any(i % 2 == 0 and i + 1 in s for i in s)


def calibration_tag_check(episode_dir, rig=None, n_frames: int = 10, aruco=None) -> dict:
    """{"frames", "both", "fraction", "ids_seen", "ok", "warn"} for a raw calibration episode."""
    import cv2

    from sensor_rig.raw_episode import RawEpisode

    ep = RawEpisode(episode_dir)
    rig = rig or ep.rig()
    remap = rig.remap("ultrawide", "export")
    t = ep.camera_times("ultrawide")
    idx = np.unique(np.linspace(0, len(t) - 1, min(n_frames, len(t))).round().astype(int)) if len(t) else []
    both, seen = 0, set()
    for i in idx:
        img = remap.apply(ep.read_image("ultrawide", int(i)))
        ids = detect_marker_ids(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), aruco)
        seen.update(ids)
        both += both_tags(ids)
    n = len(idx)
    frac = both / n if n else 0.0
    return {"frames": n, "both": both, "fraction": frac, "ids_seen": sorted(seen), "ok": both > 0,
            "warn": both > 0 and frac < WARN_FRACTION}
