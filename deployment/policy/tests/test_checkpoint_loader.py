"""CheckpointLoader's load checks on hand-made cfg / shape_meta objects (no real checkpoint needed).

Optional: POLICY_TEST_CHECKPOINT=<run_001's slim best.ckpt> also loads a real checkpoint on the CPU.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch tests run in the policy environment")
pytest.importorskip("behavior_prompting", reason="needs behavior_prompting (the policy environment)")

from conftest import RUN_001_SHAPE_META, plain_diffusion_shape_meta  # noqa: E402
from fakes_torch import TinyDiffusionModel  # noqa: E402
from policy import checkpoint_loader as cl  # noqa: E402
from policy import light_workspace  # noqa: E402
from policy.checkpoint_loader import CheckpointLoader, LoadedModel  # noqa: E402
from policy.errors import PolicyLoadError  # noqa: E402

TIMING = {"obs_down_sample_steps": 3, "ultrawide_down_sample_steps": 1}


def _cfg(shape_meta=None, **task_overrides):
    task = {"obs_down_sample_steps": 3, "ultrawide_down_sample_steps": 1,
            "pose_repr": {"obs_pose_repr": "relative", "action_pose_repr": "relative"},
            "shape_meta": copy.deepcopy(shape_meta or plain_diffusion_shape_meta())}
    task.update(task_overrides)
    return {"name": "umi_policy_dunet", "task": task, "model": {"_target_": "x.DiffusionUnetPolicy"}}


def _model_with(scheduler):
    model = TinyDiffusionModel({})
    model.noise_scheduler = scheduler
    return model


def test_prompting_checkpoint_refused_by_diffusion_policy(monkeypatch, station_file, tmp_path):
    from policy.diffusion_policy import DiffusionPolicy
    from policy.settings import Settings
    loaded = LoadedModel(model=TinyDiffusionModel({}), shape_meta=copy.deepcopy(RUN_001_SHAPE_META),
                         uses_prompting=True, info={})
    monkeypatch.setattr(CheckpointLoader, "load", lambda self, path, device, steps: loaded)
    settings = Settings.load(station_file({"kind": "diffusion", "checkpoint": "b.ckpt", "device": "cpu"}), tmp_path,
                             mock=False)
    with pytest.raises(PolicyLoadError) as e:
        DiffusionPolicy.from_settings(settings)
    assert str(e.value) == ("checkpoint uses prompting (shape_meta.use_prompting: true): kind diffusion needs a plain "
                            "diffusion checkpoint")


def test_pose_representation_checks():
    CheckpointLoader.check_pose_repr(_cfg())
    with pytest.raises(PolicyLoadError) as e:
        CheckpointLoader.check_pose_repr(_cfg(pose_repr={"obs_pose_repr": "abs", "action_pose_repr": "relative"}))
    assert "task.pose_repr.obs_pose_repr is 'abs'" in e.value.what
    meta = plain_diffusion_shape_meta()
    meta["action"]["rotation_rep"] = "axis_angle"
    with pytest.raises(PolicyLoadError) as e:
        CheckpointLoader.check_pose_repr(_cfg(meta))
    assert "rotation_rep" in e.value.what


def test_timing_mismatch_refused():
    loader = CheckpointLoader(TIMING)
    loader.check_timing(_cfg())
    with pytest.raises(PolicyLoadError) as e:
        loader.check_timing(_cfg(obs_down_sample_steps=2))
    assert "task.obs_down_sample_steps is 2, expected 3" in e.value.what


def test_diffusion_steps_override():
    from diffusers import DDIMScheduler, DDPMScheduler
    model = _model_with(DDIMScheduler(num_train_timesteps=50))
    CheckpointLoader.apply_diffusion_steps(model, None)
    assert model.num_inference_steps == 16
    CheckpointLoader.apply_diffusion_steps(model, 8)
    assert model.num_inference_steps == 8
    with pytest.raises(PolicyLoadError) as e:
        CheckpointLoader.apply_diffusion_steps(model, 51)
    assert "outside 1..50" in e.value.what
    with pytest.raises(PolicyLoadError) as e:
        CheckpointLoader.apply_diffusion_steps(_model_with(DDPMScheduler(num_train_timesteps=50)), 8)
    assert "not DDIMScheduler" in e.value.what


def _encoder(cls, model_name: str, mean, std):
    """An encoder object of the real class with only the attributes the check reads (no backbone built)."""
    import torchvision
    enc = cls.__new__(cls)
    torch.nn.Module.__init__(enc)
    enc.model_name = model_name
    enc.use_vision_norm = True
    enc.share_rgb_model = False
    enc.model_normalization_transform = torchvision.transforms.Normalize(mean=mean, std=std)
    return enc


@pytest.mark.parametrize("encoder_path", [
    "behavior_prompting.train_network.model.vision.timm_obs_encoder.TimmObsEncoder",
    "behavior_prompting.train_network.model.vision.transformer_obs_encoder.TransformerObsEncoder",
])
def test_colour_normalisation_check_on_both_encoder_types(encoder_path):
    import importlib
    module, name = encoder_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    clip = "vit_base_patch16_clip_224.openai"
    policy = torch.nn.Module()
    policy.obs_encoder = _encoder(cls, clip, light_workspace.CLIP_MEAN, light_workspace.CLIP_STD)
    assert len(light_workspace.check_vision_normalization(policy)) == 1

    policy.obs_encoder = _encoder(cls, clip, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))     # ImageNet: wrong
    with pytest.raises(AssertionError):
        light_workspace.check_vision_normalization(policy)
    policy.obs_encoder = _encoder(cls, "resnet34.a1_in1k", light_workspace.CLIP_MEAN, light_workspace.CLIP_STD)
    with pytest.raises(AssertionError, match="REFERENCE_NORMALIZATION"):
        light_workspace.check_vision_normalization(policy)


def test_missing_file_and_wrong_suffix(tmp_path):
    with pytest.raises(PolicyLoadError) as e:
        CheckpointLoader.check_file(tmp_path / "best.ckpt")
    assert e.value.what.startswith("checkpoint not found")
    (tmp_path / "best.pt").write_bytes(b"x")
    with pytest.raises(PolicyLoadError) as e:
        CheckpointLoader.check_file(tmp_path / "best.pt")
    assert ".ckpt" in e.value.what


def test_cuda_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(cl.torch.cuda, "is_available", lambda: False)
    with pytest.raises(PolicyLoadError) as e:
        CheckpointLoader.check_device("cuda")
    assert e.value.what == "CUDA not available"
    assert e.value.check.startswith("check the GPU driver, or set policy.device: cpu")
    CheckpointLoader.check_device("cpu")


@pytest.mark.skipif(not os.environ.get("POLICY_TEST_CHECKPOINT"),
                    reason="set POLICY_TEST_CHECKPOINT to run_001's slim best.ckpt to load a real checkpoint")
def test_real_checkpoint_loads_on_cpu():
    loaded = CheckpointLoader(TIMING).load(Path(os.environ["POLICY_TEST_CHECKPOINT"]), "cpu", None)
    report = loaded.info["load_report"]
    assert report and all(v["missing"] == 0 and v["unexpected"] == 0 for v in report.values())
    assert loaded.info["colour_normalisation_checked"]
    assert light_workspace.loaded_heavy_modules() == []
    assert loaded.uses_prompting is True                                  # run_001 is a prompting checkpoint
