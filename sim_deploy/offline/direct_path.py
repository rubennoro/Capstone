"""W1 direct path = what training saw: the training dataset class at (episode, row T), normalizer NOT applied (obs as
the policy receives them before its own normalisation), plus the ground-truth action. Torch env, CPU only
(``~/envs/sim_policy_deploy_torch/bin/python``; zarr + JPEG-XL codecs; no GPU, no model).

    cd ~/Capstone-main-sim && unset PYTHONPATH
    ~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.offline.direct_path --run RUN_FOLDER --index 14 --rows 6 9

The dataset is built exactly as the training repo's offline eval does it
(``controlSW/franka_deploy/franka_deploy/tools/offline_eval.py build_eval_datasets``): the run's resolved
``cfg.task.dataset`` (``run_config/config.yaml``, resolved with OmegaConf; ``policy_runner`` checks it against the
checkpoint's own cfg), ``cache_dir = None``, ``hydra.utils.instantiate(cfg, dataset_path=<package>,
allow_zip_file=True)`` -> ``UmiTaskDataset`` (its own seed-42 val split), ``get_validation_dataset()``. The prompt is
switched off (``set_ignore_prompt(True)``): only the receding obs and the action, as ``sample_sequence_without_prompting``
gives them for both the plain (episode sampling) and the prompting runs (task sampling; the task starts at the
episode start in these packages, checked).

The dataset index: ``sampler.indices`` rows are ``(current_within_segment_idx, episode_idx, task_idx)``
(``common/sampler.py setup_indices_without_prompting``: current_within_segment_idx = data row - segment start), so
(episode, T) is the row with ``[0] == T`` and ``[1] == episode``, searched in the train set, then the val set.

Returned per sample (``UmiTaskDataset.__getitem__``, then back to the C4 layout): images uint8 (T, 224, 224, 3)
(``__getitem__`` gives float32 k/255 in (T, 3, H, W); x*255 rounds back exactly), eef_pos / eef_rot6d relative to
the pose at T (rows T-3, T), eef_rot6d_wrt_start (with training's N(0, 0.05) start-pose noise, seeded; ignored by
the policy) and the same without the noise (the dataset's own pose functions, as camera_study/reference.py),
gripper_width_m (absolute, model space), action (16, 10) relative to the pose at T: action k is data row T + 3k
(``sample_sequence_without_prompting``: ``action[cur : cur + 15*3 + 1 : 3]``, padded with the last row at the
episode end), so action 0 is the pose at T itself; and the absolute recorded poses at rows T + 3k (for the waypoint
alignment check).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from sim_deploy.offline.common import set_cpu_threads
from sim_deploy.offline.episodes import dataset_path

MAIN_KEY, UW_KEY = "camera_right_main_rgb", "camera_right_ultrawide_rgb"
POS_KEY, ROT_KEY = "gripper_right_eef_pos", "gripper_right_eef_rot_axis_angle"
W_KEY, WRT_KEY = "gripper_right_gripper_width", "gripper_right_eef_rot_axis_angle_wrt_start"
C4_OF_MODEL = {MAIN_KEY: "main", UW_KEY: "ultrawide", POS_KEY: "eef_pos", ROT_KEY: "eef_rot6d",
               W_KEY: "gripper_width_m", WRT_KEY: "eef_rot6d_wrt_start"}


def run_dataset_cfg(run_folder: Path) -> dict:
    """The run's cfg.task.dataset, resolved (run_config/config.yaml is the full hydra config of the run)."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Path(run_folder) / "run_config" / "config.yaml")
    return OmegaConf.to_container(cfg.task.dataset, resolve=True)


def cfg_signature(dataset_cfg: dict) -> dict:
    """The parts of a dataset cfg that decide the receding obs and the action (compared across checkpoints)."""
    sm = dataset_cfg["shape_meta"]
    return {"obs": {k: {f: v.get(f) for f in ("shape", "horizon", "down_sample_steps", "type", "ignore_by_policy",
                                               "include_in_receding_obs", "in_replay_buffer", "rotation_rep")}
                    for k, v in sm["obs"].items()},
            "action": {f: sm["action"].get(f) for f in ("shape", "horizon", "down_sample_steps", "rotation_rep",
                                                         "components")},
            "pose_repr": dataset_cfg.get("pose_repr"), "action_padding": dataset_cfg.get("action_padding")}


class DirectDataset:
    """The training dataset of one package (built once); ``sample(episode, row)`` per decision."""

    def __init__(self, dataset: str, dataset_cfg: dict) -> None:
        import hydra
        from omegaconf import OmegaConf
        from behavior_prompting.common.imagecodecs_numcodecs import register_codecs

        set_cpu_threads()
        register_codecs(verbose=False)
        self.dataset = dataset
        self.path = dataset_path(dataset)
        self.cfg = dataset_cfg
        dcfg = OmegaConf.create(dataset_cfg)
        dcfg.cache_dir = None
        self.ds = hydra.utils.instantiate(dcfg, dataset_path=str(self.path), allow_zip_file=True)
        self.val = self.ds.get_validation_dataset()
        self.ds.set_ignore_prompt(True)
        self.val.set_ignore_prompt(True)
        rb = self.ds.replay_buffer
        self.rb = rb
        self.names = [str(n) for n in rb.episode_names[:]]
        self.episode_ends = np.asarray(rb.episode_ends[:], dtype=np.int64)
        if dataset_cfg.get("sample_type") == "task":
            t_start = np.asarray(rb.task_data_ends[:]) - np.asarray(rb.task_lengths[:])
            assert np.array_equal(t_start, np.r_[0, self.episode_ends[:-1]]), "task start != episode start"
        self._index = {}
        for split, d in (("train", self.ds), ("val", self.val)):
            for i, (row, ep, _task) in enumerate(d.sampler.indices):
                self._index[(int(ep), int(row))] = (split, i)

    def split_info(self) -> dict:
        return {str(k): bool(v) for k, v in self.ds.get_training_split_info().items()}

    def episode_start(self, ep: int) -> int:
        return int(self.episode_ends[ep - 1]) if ep > 0 else 0

    def sample(self, ep: int, row: int, noise_seed: int = 0) -> dict:
        from behavior_prompting.common.pose_util import mat_to_pose10d, pose_to_mat
        from behavior_prompting.train_network.common.pose_repr_util import convert_pose_mat_rep

        split, idx = self._index[(int(ep), int(row))]
        d = self.ds if split == "train" else self.val
        r, e, _ = d.sampler.indices[idx]
        assert int(r) == row and int(e) == ep
        np.random.seed(noise_seed)               # the wrt_start "HACK" noise draw (np.random.normal in __getitem__)
        item = d[idx]
        obs = {}
        for key, val in item["obs"].items():
            if key not in C4_OF_MODEL:
                continue
            a = val.numpy()
            if key in (MAIN_KEY, UW_KEY):
                u8 = np.round(np.moveaxis(a, 1, -1) * 255.0)
                assert np.max(np.abs(u8 / 255.0 - np.moveaxis(a, 1, -1))) < 1e-6, "image is not k/255"
                obs[C4_OF_MODEL[key]] = u8.astype(np.uint8)
            else:
                obs[C4_OF_MODEL[key]] = a.astype(np.float32)
        start = self.episode_start(ep)
        stop = int(self.episode_ends[ep])
        cur = start + row
        pos, rot = self.rb[POS_KEY], self.rb[ROT_KEY]
        lrows = [x for x in (cur - 3, cur) if x >= start]
        lrows = [lrows[0]] * (2 - len(lrows)) + lrows
        pmat = pose_to_mat(np.concatenate([pos[lrows], rot[lrows]], axis=-1))
        start_pose = np.concatenate([pos[start], rot[start]])
        clean = mat_to_pose10d(convert_pose_mat_rep(pmat, base_pose_mat=pose_to_mat(start_pose), pose_rep="relative",
                                                    backward=False))[:, 3:].astype(np.float32)
        fut = [min(cur + 3 * k, stop - 1) for k in range(16)]
        fut_T = pose_to_mat(np.concatenate([pos[fut], rot[fut]], axis=-1)).astype(np.float64)
        anchor_T = pose_to_mat(np.concatenate([pos[cur], rot[cur]])).astype(np.float64)
        return {"obs": obs, "action": item["action"].numpy().astype(np.float32),
                "wrt_start_clean": clean, "future_rows": [x - start for x in fut], "future_T": fut_T,
                "row_T": anchor_T, "lowdim_rows": [x - start for x in lrows],
                "split_in_dataset": split, "dataset_index": int(idx), "episode_name": self.names[ep],
                "data_row": int(cur), "episode_start_row": start, "n_rows": stop - start, "noise_seed": noise_seed}


def main(argv=None) -> int:
    import json
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dataset", default=None, help="default: the run's train_meta.json dataset")
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--rows", type=int, nargs="+", required=True)
    a = ap.parse_args(argv)
    name = a.dataset or json.loads((a.run / "train_meta.json").read_text())["dataset"]
    dd = DirectDataset(name, run_dataset_cfg(a.run))
    for r in a.rows:
        s = dd.sample(a.index, r)
        print(r, s["split_in_dataset"], s["dataset_index"], s["episode_name"], s["lowdim_rows"],
              {k: v.shape for k, v in s["obs"].items()}, "action0", np.round(s["action"][0], 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
