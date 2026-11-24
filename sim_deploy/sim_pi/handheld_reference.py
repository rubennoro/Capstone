"""Export one training episode's images for the G3 check (handheld_check.py), from a packaged zarr.

    ~/envs/sim_policy_deploy_torch/bin/python sim_deploy/sim_pi/handheld_reference.py ZARR_ZIP INDEX OUT.npz

Needs ``zarr`` and the JPEG-XL codecs (``behavior_prompting``), so it runs in the torch env, not in ~/envs/sim_deploy;
handheld_check runs it as a subprocess. Writes: ``uw`` (n, 224, 224, 3) the episode's ultrawide images (10 Hz),
``rows`` their data rows, ``main`` (n, 224, 224, 3) the main image at those rows, ``width_m`` the recorded gripper
width at those rows (training units: the ArUco tag-centre distance), ``name`` the episode name.
"""

from __future__ import annotations

import sys

import numpy as np


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path, index, out = argv[0], int(argv[1]), argv[2]
    import zarr
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    register_codecs()
    root = zarr.open(zarr.ZipStore(path, mode="r"), mode="r")
    ue = root["meta/episode_ends_camera_right_ultrawide_rgb"][:]
    s, e = (int(ue[index - 1]) if index else 0), int(ue[index])
    rows = root["meta/downsample_index_camera_right_ultrawide_rgb"][s:e].astype(np.int64)
    uw = root["data/camera_right_ultrawide_rgb"][s:e]
    main_imgs = np.stack([root["data/camera_right_main_rgb"][int(r)] for r in rows])
    width = root["data/gripper_right_gripper_width"][:][rows, 0].astype(np.float64)
    name = str(root["meta/episode_names"][index])
    np.savez_compressed(out, uw=uw, main=main_imgs, rows=rows, width_m=width, name=name)
    print(f"handheld_reference: {name}: {len(rows)} ultrawide images, width {width.min():.4f}..{width.max():.4f} m "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
