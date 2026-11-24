"""Export the real training images and low-dim rows of the camera study B references (layout.json) for render_sets.

    ~/envs/sim_policy_deploy_torch/bin/python sim_deploy/camera_study/sets_real_export.py LAYOUT_JSON OUT_DIR

Needs ``zarr`` and the JPEG-XL codecs (``behavior_prompting``), so it runs in the torch env (render_sets.py calls it as
a subprocess when its cache is missing). For every reference (``ep<index:02d>_<label>``, row T in the episode) it
writes ``OUT_DIR/ep<ii>_<label>.npz``:

* ``main`` (2, 224, 224, 3): the package's IMX519 images at rows T-3, T (training's horizon 2 at 20 Hz), oldest first;
* ``uw`` (2, 224, 224, 3): the two newest IMX219 frames at row T (``upsample_index`` at T and the one before);
* ``main_rows``, ``uw_rows``: their rows in the episode; ``pos`` / ``rotvec`` / ``width`` (n, ...) of the whole episode
  (60 Hz low-dim rows, tag frame); ``name``, ``index``, ``T``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    layout = json.loads(Path(argv[0]).read_text())
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    import zarr
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    register_codecs()
    root = zarr.open(zarr.ZipStore(layout["dataset"], mode="r"), mode="r")
    ends = root["meta/episode_ends"][:]
    up = root["meta/upsample_index_camera_right_ultrawide_rgb"][:]
    down = root["meta/downsample_index_camera_right_ultrawide_rgb"][:]
    pos_all = root["data/gripper_right_eef_pos"]
    rot_all = root["data/gripper_right_eef_rot_axis_angle"]
    w_all = root["data/gripper_right_gripper_width"]
    for sel in layout["selection"]:
        i = int(sel["index"])
        s, e = (int(ends[i - 1]) if i else 0), int(ends[i])
        name = str(root["meta/episode_names"][i])
        for T, label in zip(sel["rows_in_episode"], sel["labels"]):
            T = int(T)
            mr = [T - 3, T]
            main = np.stack([root["data/camera_right_main_rgb"][s + r] for r in mr])
            k = int(up[s + T])
            ks = [k - 1, k]
            uw = np.stack([root["data/camera_right_ultrawide_rgb"][j] for j in ks])
            ur = [int(down[j]) - s for j in ks]
            f = out / f"ep{i:02d}_{label}.npz"
            np.savez_compressed(f, main=main, uw=uw, main_rows=np.array(mr), uw_rows=np.array(ur),
                                pos=pos_all[s:e].astype(np.float64), rotvec=rot_all[s:e].astype(np.float64),
                                width=w_all[s:e, 0].astype(np.float64), name=name, index=i, T=T)
            print(f"sets_real_export: {f.name}: {name} T {T}, main rows {mr}, IMX219 rows {ur}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
