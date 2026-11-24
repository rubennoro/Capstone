# new (STRUCTURE.md 5.2): the prompting checkpoint path of kind diffusion
"""PromptableDiffusionPolicy and PromptSource with a tiny prompting model and a fake prompt dataset (built through
hydra.utils.instantiate, as the real UmiTaskDataset is). No real checkpoint needed.

Optional: POLICY_TEST_CHECKPOINT=<run_001's slim best.ckpt> and POLICY_TEST_PROMPT_DATASET=<its .zarr.zip> also load
the real checkpoint, prompt it with the longest demo and run one inference (~15 s, ~5 GB host RAM; on CUDA if there).
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch tests run in the policy environment")
pytest.importorskip("behavior_prompting", reason="needs behavior_prompting (the policy environment)")
pytest.importorskip("hydra", reason="needs hydra (the policy environment)")

from deploy_contracts import Ack, RemoteError, RpcClient  # noqa: E402
from policy_deploy.contracts import ActionChunk, PolicyConfig, PolicyConfigRequest, Prompt, Reset  # noqa: E402

from policy_fakes import RUN_001_SHAPE_META, ServeThread, plain_diffusion_shape_meta  # noqa: E402
from policy_fakes_torch import FakePromptDataset, TinyDiffusionModel, TinyPromptModel  # noqa: E402
from policy_deploy.policy.app import run_warm_up, run_warm_up_prompt  # noqa: E402
from policy_deploy.policy.checkpoint_loader import CheckpointLoader, LoadedModel  # noqa: E402
from policy_deploy.policy.diffusion_policy import DiffusionPolicy  # noqa: E402
from policy_deploy.policy.errors import BadRequest, PolicyLoadError  # noqa: E402
from policy_deploy.policy.neutral_observation import neutral_observation  # noqa: E402
from policy_deploy.policy.policy_server import PolicyServer  # noqa: E402
from policy_deploy.policy.prompt_source import PromptSource  # noqa: E402
from policy_deploy.policy.promptable_diffusion_policy import PromptableDiffusionPolicy  # noqa: E402
from policy_deploy.policy.promptable_policy import PromptablePolicy  # noqa: E402
from policy_deploy.policy.settings import Settings  # noqa: E402

TORCH_PY = {"light": sys.executable, "torch": sys.executable}
EXPECTED = {
    "gripper_right_eef_pos": (1, 2, 3),
    "gripper_right_eef_rot_axis_angle": (1, 2, 6),
    "gripper_right_gripper_width": (1, 2, 1),
    "camera_right_main_rgb": (1, 2, 3, 224, 224),
    "camera_right_ultrawide_rgb": (1, 2, 3, 224, 224),
}
# the shape of cfg.task.dataset (run_001's: a cluster dataset_path and cache_dir), with the fake dataset class
DATASET_CFG = {"_target_": "policy_fakes_torch.FakePromptDataset",
               "dataset_path": "/home/someone/cluster/replay_buffer.zarr.zip", "cache_dir": "/tmp/someone/zarr_cache",
               "val_ratio": 0.05, "seed": 42, "sample_type": "task"}
SPLIT_INFO = {f"ep{i}:{name}": True for i, name in enumerate(["block_raise"] * 3 + ["other"])}


@pytest.fixture
def demos(tmp_path) -> Path:
    path = tmp_path / "demos.zarr.zip"
    path.write_bytes(b"")
    return path


def _settings(policy_station, **section) -> Settings:
    return Settings.load(policy_station({"kind": "diffusion", "checkpoint": "b.ckpt", "device": "cpu", **section},
                                        python=TORCH_PY), mock=False)


def _loaded(prompting: bool = True, split_info=SPLIT_INFO) -> LoadedModel:
    if prompting:
        model = TinyPromptModel(EXPECTED)
        model.split_info = None if split_info is None else dict(split_info)
        meta = copy.deepcopy(RUN_001_SHAPE_META)
    else:
        model, meta = TinyDiffusionModel(EXPECTED), plain_diffusion_shape_meta()
    return LoadedModel(model=model, shape_meta=meta, uses_prompting=prompting, info={"epoch": 50},
                       dataset_cfg=dict(DATASET_CFG))


def _build(monkeypatch, policy_station, loaded: LoadedModel, **section):
    monkeypatch.setattr(CheckpointLoader, "load", lambda self, path, device, steps: loaded)
    return DiffusionPolicy.from_settings(_settings(policy_station, **section))


def _promptable(monkeypatch, policy_station, demos, **section) -> tuple[PromptableDiffusionPolicy, TinyPromptModel]:
    loaded = _loaded()
    policy = _build(monkeypatch, policy_station, loaded, prompt_dataset=str(demos), **section)
    assert isinstance(policy, PromptableDiffusionPolicy)
    return policy, loaded.model


# ------------------------------------------------------------------------------------------- which policy is built

def test_plain_checkpoint_gives_the_plain_policy(monkeypatch, policy_station):
    policy = _build(monkeypatch, policy_station, _loaded(prompting=False))
    assert type(policy) is DiffusionPolicy and not isinstance(policy, PromptablePolicy)
    config = policy.config()
    assert config.kind == "diffusion" and config.uses_prompt is False
    assert "prompt_dataset" not in policy.describe()


def test_prompting_checkpoint_gives_the_promptable_policy(monkeypatch, policy_station, demos):
    policy, _ = _promptable(monkeypatch, policy_station, demos)
    config = policy.config()
    assert config.kind == "diffusion" and config.uses_prompt is True
    assert list(config.inputs) == ["eef_pos", "eef_rot6d", "gripper_width_m", "main", "ultrawide"]
    kwargs = FakePromptDataset.last_kwargs             # what the dataset was built with
    assert kwargs["only_prompt"] is True and kwargs["allow_zip_file"] is True
    assert kwargs["dataset_path"] == str(demos.resolve())          # the laptop's file, not the cluster path
    assert kwargs["cache_dir"] is None and kwargs["val_ratio"] == 0
    assert dict(kwargs["training_split_info"]) == SPLIT_INFO       # train split: the model's own split info


def test_prompting_checkpoint_without_prompt_dataset_refused(monkeypatch, policy_station):
    with pytest.raises(PolicyLoadError) as e:
        _build(monkeypatch, policy_station, _loaded())
    assert str(e.value) == ("checkpoint uses prompting (shape_meta.use_prompting: true): set policy.prompt_dataset "
                            "(the processed .zarr.zip of the demos)")


def test_plain_checkpoint_with_prompt_dataset_refused(monkeypatch, policy_station, demos):
    with pytest.raises(PolicyLoadError) as e:
        _build(monkeypatch, policy_station, _loaded(prompting=False), prompt_dataset=str(demos))
    assert e.value.what.startswith("policy.prompt_dataset is set but the checkpoint does not use prompting")
    assert e.value.check == "remove policy.prompt_dataset from station.yaml"


def test_describe_has_the_prompt_dataset_and_the_warm_up_demo(monkeypatch, policy_station, demos):
    policy, _ = _promptable(monkeypatch, policy_station, demos)
    d = policy.describe()
    assert d["kind"] == "diffusion" and d["epoch"] == 50 and d["inputs"][0] == "eef_pos"
    assert d["prompt_dataset"] == str(demos.resolve()) and d["prompt_split"] == "train"
    assert d["prompt_tasks"] == {"block_raise": 3, "other": 1}
    assert d["prompt_warm_up_demo"] == {"task_name": "other", "prompt_index": 0, "steps": 300, "episode": "ep3"}


# ---------------------------------------------------------------------------------------------------- prompting

def test_prompt_gives_the_model_batched_tensors_under_inference_mode(monkeypatch, policy_station, demos):
    policy, model = _promptable(monkeypatch, policy_station, demos)
    policy.prompt("block_raise", 1)                                # demo 1: 250 steps -> 4 prompt steps
    prompt = model.prompts[-1]
    assert prompt["obs"]["camera_right_main_rgb"].shape == (1, 4, 3, 4, 4)
    assert float(prompt["obs"]["gripper_right_gripper_width"][0, 0, 0]) == 1.0       # dataset index 1
    assert prompt["action"].shape == (1, 4, 20, 10) and prompt["metadata"]["task_indices"].tolist() == [[1]]
    assert model.prompt_inference_mode == [True]
    assert policy.last_prompt == {"task_name": "block_raise", "prompt_index": 1, "steps": 250, "episode": "ep1",
                                  "prompt_s": policy.last_prompt["prompt_s"]}
    chunk = policy.infer(neutral_observation(0.05))
    assert isinstance(chunk, ActionChunk)


@pytest.mark.parametrize("task,index,text", [
    ("pick_place", 0, "prompt: unknown task 'pick_place'; the prompt dataset (train split) has 'block_raise', "
                      "'other'"),
    ("block_raise", 3, "prompt: prompt_index 3 out of range for task 'block_raise': 0..2 (3 demos in the train "
                       "split)"),
    ("block_raise", -1, "prompt_index -1 out of range"),
])
def test_unknown_task_or_bad_index_is_a_bad_request(monkeypatch, policy_station, demos, task, index, text):
    policy, model = _promptable(monkeypatch, policy_station, demos)
    with pytest.raises(BadRequest) as e:
        policy.prompt(task, index)
    assert text in str(e.value)
    assert model.prompts == [] and policy.last_prompt is None


def test_reset_clears_the_prompt_and_the_server_requires_a_prompt_again(monkeypatch, policy_station, demos):
    policy, model = _promptable(monkeypatch, policy_station, demos)
    with ServeThread(PolicyServer(policy, "tcp://127.0.0.1:*", reset_max_s=0.5)) as thread:
        client = RpcClient("policy", thread.server.endpoint)
        try:
            assert client.request(PolicyConfigRequest(), PolicyConfig, timeout_s=5.0).uses_prompt is True
            for index in (0, 2):
                assert client.request(Reset(), Ack, timeout_s=5.0).ok
                assert model.prompted is None and policy.last_prompt is None
                with pytest.raises(RemoteError) as e:
                    client.request(neutral_observation(0.05), ActionChunk, timeout_s=5.0)
                assert "infer before prompt" in e.value.reply.error
                with pytest.raises(RemoteError) as e:                   # refused, serving goes on
                    client.request(Prompt(task_name="nope", prompt_index=0), Ack, timeout_s=5.0)
                assert "unknown task 'nope'" in e.value.reply.error
                assert client.request(Prompt(task_name="block_raise", prompt_index=index), Ack, timeout_s=5.0).ok
                client.request(neutral_observation(0.05), ActionChunk, timeout_s=5.0)
        finally:
            client.close()
        assert thread.error is None
    assert model.events == ["reset", "prompt", "predict", "reset", "prompt", "predict"]


def test_warm_up_prompts_with_the_longest_demo_first(monkeypatch, policy_station, demos):
    policy, model = _promptable(monkeypatch, policy_station, demos)
    warm = run_warm_up_prompt(policy)
    times = run_warm_up(policy, neutral_observation(0.05), 2)
    assert warm["task_name"] == "other" and warm["prompt_index"] == 0 and warm["prompt_s"] >= 0
    assert len(times) == 2
    assert model.events == ["prompt", "predict", "predict", "reset"]
    assert model.prompted is None                                   # the warm-up's reset cleared it
    assert float(model.prompts[0]["obs"]["gripper_right_gripper_width"][0, 0, 0]) == 3.0      # dataset index 3


# ------------------------------------------------------------------------------------------- PromptSource.load

def test_val_and_all_splits(monkeypatch, policy_station, demos):
    policy, _ = _promptable(monkeypatch, policy_station, demos, prompt_split="val")
    assert policy.describe()["prompt_tasks"] == {"block_raise": 1}             # the fake's validation set
    policy, _ = _promptable(monkeypatch, policy_station, demos, prompt_split="all")
    assert FakePromptDataset.last_kwargs["training_split_info"] is None
    assert policy.describe()["prompt_tasks"] == {"block_raise": 3, "other": 1}


def test_load_errors(demos, tmp_path):
    def load(path=demos, split="train", cfg=DATASET_CFG, info=SPLIT_INFO):
        with pytest.raises(PolicyLoadError) as e:
            PromptSource.load(path, split, cfg, info, "cpu")
        return e.value

    e = load(path=tmp_path / "missing.zarr.zip")
    assert e.what.startswith("prompt dataset not found") and "policy.prompt_dataset" in e.check
    (tmp_path / "demos.ckpt").write_bytes(b"")
    assert "not a .zarr.zip" in load(path=tmp_path / "demos.ckpt").what
    assert "no task.dataset config" in load(cfg=None).what
    e = load(info=None)
    assert "needs the model's training split info" in e.what and e.check == "set policy.prompt_split: all"
    e = load(info={"ep0:block_raise": True})                  # a dataset the model never trained on
    assert "training split info does not know" in e.what and "policy.prompt_split: all" in e.check
    e = load(cfg={**DATASET_CFG, "_target_": "policy_fakes_torch.NoSuchDataset"})
    assert e.what.startswith("prompt dataset failed to load") and "policy.prompt_dataset" in e.check
    assert "unknown prompt split" in load(split="test").what


def test_only_a_split_info_key_gets_the_prompt_split_all_hint(demos):
    with pytest.raises(PolicyLoadError) as e:              # a demo the split info lacks: "<episode>:<task>"
        PromptSource.load(demos, "train", DATASET_CFG, {"ep0:block_raise": True}, "cpu")
    assert "training split info does not know (ep1:block_raise)" in e.value.what
    assert "policy.prompt_split: all" in e.value.check
    with pytest.raises(PolicyLoadError) as e:              # any other KeyError (a wrong or old dataset)
        PromptSource.load(demos, "train", {**DATASET_CFG, "missing_key": "meta"}, SPLIT_INFO, "cpu")
    assert e.value.what.startswith("prompt dataset failed to load") and "'meta'" in e.value.what
    assert "prompt_split" not in e.value.check and "policy.prompt_dataset" in e.value.check


def test_val_split_of_a_model_trained_without_one_refused(demos):
    with pytest.raises(PolicyLoadError) as e:
        PromptSource.load(demos, "val", {**DATASET_CFG, "empty_val": True}, SPLIT_INFO, "cpu")
    assert e.value.what.startswith("no demos in the val split of")
    assert e.value.check == "set policy.prompt_split: train or all"
    assert PromptSource.load(demos, "train", {**DATASET_CFG, "empty_val": True}, SPLIT_INFO, "cpu").tasks()


def test_empty_split_refused():
    with pytest.raises(PolicyLoadError) as e:
        PromptSource(FakePromptDataset(tasks={}), "cpu", "x.zarr.zip", "val")
    assert "no demos in split 'val'" in e.value.what


def test_longest_without_lengths_is_the_first_demo():
    dataset = FakePromptDataset()
    dataset.sampler = object()                              # no prompt_indices (sequence prompting)
    source = PromptSource(dataset, "cpu")
    assert source.longest() == ("block_raise", 0)
    assert source.demo("other", 0) == {"task_name": "other", "prompt_index": 0, "steps": None, "episode": None}


# ------------------------------------------------------------------------------------------------ real artifacts

@pytest.mark.skipif(not (os.environ.get("POLICY_TEST_CHECKPOINT") and os.environ.get("POLICY_TEST_PROMPT_DATASET")),
                    reason="set POLICY_TEST_CHECKPOINT (run_001's slim best.ckpt) and POLICY_TEST_PROMPT_DATASET "
                           "(its .zarr.zip) to prompt the real checkpoint")
def test_real_checkpoint_prompted_with_the_longest_demo(policy_station):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = _settings(policy_station, checkpoint=os.environ["POLICY_TEST_CHECKPOINT"], device=device,
                         prompt_dataset=os.environ["POLICY_TEST_PROMPT_DATASET"])
    policy = DiffusionPolicy.from_settings(settings)
    assert isinstance(policy, PromptableDiffusionPolicy) and policy.config().uses_prompt
    warm = run_warm_up_prompt(policy)
    assert warm["task_name"] in policy.describe()["prompt_tasks"]
    run_warm_up(policy, neutral_observation(0.05), 1)
    policy.prompt(warm["task_name"], 0)
    assert isinstance(policy.infer(neutral_observation(0.05)), ActionChunk)
