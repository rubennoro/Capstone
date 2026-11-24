"""W1 first-chunk "approach": where a chunk's last step (k = 15) puts the TCP, in the anchor TCP frame, as P3 measured it
(sim_evals/_start_pose_sweep/tools/offline_first_chunk.py: z = toward the fingertips, cm), and its projection on the
unit vector from the anchor TCP to the recorded grasp TCP position (grasp row = ``grasp_rows(widths)["end"]``, the
hold reached, sim_deploy/scene/placement.py). numpy only; used by policy_runner.py.

    approach_metrics(chunk_pos (16, 3) relative to the anchor, anchor_T (4, 4), meta, z) -> dict (cm)

The demo's own value: the same on the ground-truth action with the row-T pose as the anchor (policy_runner "gt").
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def grasp_point(dataset_path: str, index: int) -> tuple[int, tuple[float, float, float]]:
    """(grasp row, grasp TCP position in the package frame) of one episode."""
    from sim_deploy.scene.placement import grasp_rows
    from sim_deploy.scene.zarr_lowdim import load_episode
    ep = load_episode(dataset_path, index)
    g = grasp_rows(ep["widths"])
    return int(g["end"]), tuple(float(x) for x in ep["poses"][int(g["end"])][:3, 3])


def approach_metrics(chunk_pos: np.ndarray, anchor_T: np.ndarray, meta: dict, z: dict | None = None) -> dict:
    s = meta["sample"]
    ds_path = meta["deploy_setup"]["episode"]["dataset_path"]
    g_row, g_p = grasp_point(ds_path, int(s["index"]))
    A = np.asarray(anchor_T, np.float64)
    v = A[:3, :3].T @ (np.asarray(g_p) - A[:3, 3])          # anchor TCP -> grasp TCP, in the anchor frame
    dist = float(np.linalg.norm(v))
    u = v / dist if dist > 0 else v
    p15 = np.asarray(chunk_pos, np.float64)[15]
    return {"k15_cm": np.round(p15 * 100, 3).tolist(), "k15_z_cm": round(float(p15[2] * 100), 3),
            "k15_toward_grasp_cm": round(float(p15 @ u * 100), 3), "grasp_dist_cm": round(dist * 100, 3),
            "grasp_row": g_row, "row": int(s["row"])}
