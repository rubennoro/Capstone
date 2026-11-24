"""W2 step 4: the real training inputs at the NEIGHBOUR rows of the layout's 15 references (T-6, T-3, T+3, T+6 at
60 Hz, i.e. the nearby 20 Hz frames), exactly as training's sampler gives them (camera study B ``reference.py`` stage 1:
``_sample_obs`` + ``_convert_lowdim``, imported unchanged). They give the real response's spread across nearby frames,
from which ``attribution_table.py`` derives each checkpoint's tolerance.

Run (torch env: zarr + JPEG-XL codecs; CPU only, no GPU) from ``~/Capstone-main-sim`` with ``unset PYTHONPATH``::

    ~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.camera_study.real_neighbours [--out DIR] [--offsets -6 -3 3 6]

Output (default ``sim_evals/_pipeline/W2/attribution/real_neighbours/``): per reference and offset a folder
``<ref>_d<+-k>/`` with ``obs.npz`` (exactly the C4 Observation keys, oldest first, low-dim relative to the pose at the
neighbour row; ``eef_rot6d_wrt_start`` relative to the episode's first pose, without training's noise), ``main_1.png``,
``uw_1.png`` and ``meta.json`` (reference, offset, row, data row, frames used); plus ``index.json``. A neighbour row
below 0 is skipped (none of the layout's references needs it: the ``first`` rows are T 6).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim_deploy.camera_study import reference as R

OUT = R.REPO / "sim_evals/_pipeline/W2/attribution/real_neighbours"
OFFSETS = (-6, -3, 3, 6)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--offsets", type=int, nargs="+", default=list(OFFSETS))
    a = ap.parse_args(argv)

    import zarr
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    from behavior_prompting.common.replay_buffer import ReplayBuffer
    register_codecs(verbose=False)
    R._ensure_deploy_path()
    from policy_deploy.contracts.c4_policy import Observation

    layout = R._layout()
    dataset = Path(layout["dataset"]).expanduser()
    store = zarr.ZipStore(str(dataset), mode="r")
    rb = ReplayBuffer(root=zarr.open(store, mode="r"))
    names = [str(n) for n in rb.episode_names[:]]
    a.out.mkdir(parents=True, exist_ok=True)
    index = []
    for ref in R._refs(layout):
        i, T = ref["index"], ref["row"]
        # the reference row itself, re-sampled: must equal the B1 reference obs.npz bit for bit
        s, used = R._sample_obs(rb, i, T)
        low = R._convert_lowdim(rb, s, used["start"])
        b1 = np.load(R.OUT / ref["folder"] / "obs.npz")
        same = {k: bool(np.array_equal(b1[k], v)) for k, v in
                dict(low, main=s["main"].astype(np.uint8), ultrawide=s["uw"].astype(np.uint8)).items()}
        if not all(same.values()):
            raise SystemExit(f"{ref['folder']}: re-sampled reference differs from B1's obs.npz: {same}")
        for off in a.offsets:
            row = T + off
            if row < 0:
                print(f"{ref['folder']} d{off:+d}: row {row} < 0, skipped")
                continue
            s, used = R._sample_obs(rb, i, row)
            low = R._convert_lowdim(rb, s, used["start"])
            obs_d = dict(low, main=s["main"].astype(np.uint8), ultrawide=s["uw"].astype(np.uint8))
            Observation(**obs_d).check_values()
            key = f"{ref['folder']}_d{off:+d}"
            d = a.out / key
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "obs.npz", **obs_d)
            R._png(d / "main_1.png", obs_d["main"][1])
            R._png(d / "uw_1.png", obs_d["ultrawide"][1])
            start = used["start"]
            meta = {"key": key, "reference": ref["folder"], "offset_rows": off, "row_in_episode": row,
                    "reference_row_in_episode": T, "package_index": i, "pi_name": names[i], "dataset": str(dataset),
                    "data_row": used["cur"], "episode_start_row": start,
                    "main_rows_in_episode": [r - start for r in used["main_rows"]],
                    "lowdim_rows_in_episode": [r - start for r in used["lowdim_rows"]],
                    "uw_frames_local": [u - used["uw_episode_start"] for u in used["uw_global"]],
                    "widths_m": low["gripper_width_m"][:, 0]}
            R._write_json(d / "meta.json", meta)
            index.append({"key": key, "reference": ref["folder"], "offset_rows": off, "row_in_episode": row,
                          "folder": str(d)})
            print(f"{key:<28} row {row:3d} main rows {meta['main_rows_in_episode']} uw {meta['uw_frames_local']}")
    store.close()
    R._write_json(a.out / "index.json", {"dataset": str(dataset), "offsets": a.offsets, "rows": index,
                                         "reference_resample_check": "bit-identical to B1 obs.npz (all 15)"})
    print(f"{len(index)} neighbours -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
