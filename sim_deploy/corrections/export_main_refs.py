"""Export the training main (IMX519) 224x224 images of camera study B's reference frames (layout.json), for the
overlays of fit_main.py.

    unset PYTHONPATH; ~/envs/sim_policy_deploy_torch/bin/python sim_deploy/corrections/export_main_refs.py \
        sim_evals/_camera_study/B/layout.json sim_evals/_camera_study/B/corr_main/refs_main.npz

Needs zarr + the JPEG-XL codecs (behavior_prompting), so it runs in the torch env (CPU only). The package's main
images are stored at every 60 Hz data row: the frame of reference ``ep<index:02d>_<label>`` is row
``episode_start + rows_in_episode``. Writes one (224, 224, 3) uint8 array per reference name, plus ``<name>__row``
(global data row) and ``<name>__width_m``.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    layout_path, out = argv[0], argv[1]
    import zarr
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    register_codecs()
    lay = json.load(open(layout_path))
    root = zarr.open(zarr.ZipStore(lay["dataset"], mode="r"), mode="r")
    ends = root["meta/episode_ends"][:]
    width = root["data/gripper_right_gripper_width"]
    res = {}
    for sel in lay["selection"]:
        i = int(sel["index"])
        s = int(ends[i - 1]) if i else 0
        for row, label in zip(sel["rows_in_episode"], sel["labels"]):
            name = f"ep{i:02d}_{label}"
            g = s + int(row)
            res[name] = np.asarray(root["data/camera_right_main_rgb"][g], np.uint8)
            res[name + "__row"] = np.int64(g)
            res[name + "__width_m"] = np.float64(width[g, 0])
    np.savez_compressed(out, **res)
    print(f"export_main_refs: {sum(1 for k in res if '__' not in k)} main images -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
