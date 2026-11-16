# new (STRUCTURE.md 11.1, 12); + --prompt-dataset
"""policy.bench (STRUCTURE.md 11.1): the CLI and the pure helpers run without torch; the model parts need the torch
env and behavior_prompting and are skipped here."""
from __future__ import annotations

import argparse
import io
import json

import pytest

from policy_deploy.policy import bench


def test_defaults_are_the_plain_config_16_steps_fp32():
    a = bench.parse_args([])
    assert a.config == "umi_policy_dunet" and a.checkpoint is None
    assert a.steps == [16] and a.precision == ["fp32"] and a.infers == 500 and a.warmup == 3
    assert a.device == "cuda" and a.out is None


def test_lists_and_checkpoint(tmp_path):
    a = bench.parse_args(["--checkpoint", str(tmp_path / "b.ckpt"), "--steps", "16,8,4", "--precision", "fp32,bf16",
                          "--infers", "200", "--warmup", "0", "--device", "cpu", "--out", str(tmp_path)])
    assert a.config is None and a.checkpoint == tmp_path / "b.ckpt"
    assert a.steps == [16, 8, 4] and a.precision == ["fp32", "bf16"] and a.infers == 200 and a.warmup == 0


@pytest.mark.parametrize("argv", [["--steps", "16,x"], ["--steps", "0"], ["--precision", "fp16"],
                                  ["--config", "a", "--checkpoint", "b"], ["--infers", "0"], ["--device", "tpu"],
                                  ["--prompt-dataset", "d.zarr.zip"], ["--checkpoint", "b", "--prompt-split", "x"]])
def test_bad_arguments_refused(argv):
    with pytest.raises(SystemExit):
        bench.parse_args(argv)


def test_prompt_dataset_with_a_checkpoint(tmp_path):
    a = bench.parse_args(["--checkpoint", "b.ckpt", "--prompt-dataset", str(tmp_path / "d.zarr.zip"),
                          "--prompt-split", "all"])
    assert a.prompt_dataset == tmp_path / "d.zarr.zip" and a.prompt_split == "all"
    assert bench.parse_args(["--checkpoint", "b.ckpt"]).prompt_split == "train"


def test_list_parsers():
    assert bench.int_list("16, 8") == [16, 8]
    assert bench.precision_list("bf16") == ["bf16"]
    with pytest.raises(argparse.ArgumentTypeError):
        bench.int_list("")
    with pytest.raises(argparse.ArgumentTypeError):
        bench.precision_list("fp32,int8")


def test_timing_ms():
    stats = bench.timing_ms([i / 1000 for i in range(1, 101)])            # 1 .. 100 ms
    assert stats == {"mean_ms": 50.5, "p50_ms": 50.5, "p95_ms": 95.05, "max_ms": 100.0}
    assert bench.timing_ms([]) == {}


def test_nvidia_smi_now_parsing():
    out = bench.nvidia_smi_now(lambda args: "3100, 4096, 34.5, 35.00, 1500, 71\n")
    assert out == {"smi_used_mb": 3100.0, "smi_total_mb": 4096.0, "power_draw_w": 34.5, "power_limit_w": 35.0,
                   "sm_clock_mhz": 1500.0, "temperature_c": 71.0}
    assert bench.nvidia_smi_now(lambda args: "1, 2, [N/A], [N/A], 3, 4")["power_draw_w"] is None

    def missing(args):
        raise FileNotFoundError("nvidia-smi")
    assert bench.nvidia_smi_now(missing) == {}


def test_format_line_for_both_record_kinds():
    line = bench.format_line({"what": "infer", "steps": 16, "precision": "bf16", "infers": 500, "warmup_s": [1.2, 0.2],
                              "mean_ms": 180.0, "p50_ms": 179.0, "p95_ms": 190.0, "max_ms": 210.0})
    assert line.startswith("steps 16 bf16: mean 180.0 ms, p50 179.0, p95 190.0, max 210.0 ms over 500")
    assert "(warm-up 1200, 200 ms)" in line
    line = bench.format_line({"what": "load", "source": "x.ckpt", "load_s": 12.5, "uses_prompting": True})
    assert line.startswith("load: x.ckpt: 12.5 s") and "prompting True" in line
    line = bench.format_line({"what": "prompt", "task_name": "block_raise", "prompt_index": 4, "steps": 1042,
                              "split": "train", "prompt_s": 0.8, "max_memory_allocated_mb": 2013.6})
    assert line == ("prompt: task 'block_raise', index 4 (1042 steps, train split): 0.8 s, VRAM allocated 2013.6 MB")


def test_host_ram_peak_is_positive():
    assert bench.host_ram_peak_mb() > 10


def test_main_without_torch_says_so(capsys):
    try:
        import torch  # noqa: F401
        pytest.skip("torch is installed here")
    except ImportError:
        pass
    assert bench.main(["--device", "cpu"]) == 1
    assert "python.torch" in capsys.readouterr().err


def test_run_with_a_tiny_model_on_cpu(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch", reason="torch tests run in the torch environment")
    pytest.importorskip("diffusers", reason="needs diffusers (the torch environment)")
    from diffusers import DDIMScheduler
    from policy_fakes import plain_diffusion_shape_meta
    from policy_fakes_torch import TinyDiffusionModel
    expected = {"gripper_right_eef_pos": (1, 2, 3), "gripper_right_eef_rot_axis_angle": (1, 2, 6),
                "gripper_right_gripper_width": (1, 2, 1), "camera_right_main_rgb": (1, 2, 3, 224, 224),
                "camera_right_ultrawide_rgb": (1, 2, 3, 224, 224)}
    model = TinyDiffusionModel(expected)
    model.noise_scheduler = DDIMScheduler(num_train_timesteps=50)
    monkeypatch.setattr(bench, "build_from_config", lambda name, device: (model, plain_diffusion_shape_meta()))
    out = io.StringIO()
    a = bench.parse_args(["--device", "cpu", "--steps", "16,8", "--precision", "fp32", "--infers", "3",
                          "--warmup", "1", "--out", str(tmp_path)])
    assert bench.run(a, out) == 0
    records = [json.loads(line) for line in (tmp_path / "bench.jsonl").read_text().splitlines()]
    assert [r["what"] for r in records] == ["load", "infer", "infer"]
    assert [r["steps"] for r in records[1:]] == [16, 8] and model.num_inference_steps == 8
    assert records[1]["inputs"] == "neutral_observation through ModelRunner" and records[1]["infers"] == 3
    assert len(out.getvalue().splitlines()) == 3
    del torch


def test_prompting_checkpoint_prompted_with_the_longest_demo_then_timed(monkeypatch, tmp_path):
    pytest.importorskip("torch", reason="torch tests run in the torch environment")
    pytest.importorskip("behavior_prompting", reason="needs behavior_prompting (the torch environment)")
    pytest.importorskip("diffusers", reason="needs diffusers (the torch environment)")
    import copy

    from diffusers import DDIMScheduler
    from policy_fakes import RUN_001_SHAPE_META
    from policy_fakes_torch import TinyPromptModel

    from policy_deploy.policy.checkpoint_loader import CheckpointLoader, LoadedModel
    expected = {"gripper_right_eef_pos": (1, 2, 3), "gripper_right_eef_rot_axis_angle": (1, 2, 6),
                "gripper_right_gripper_width": (1, 2, 1), "camera_right_main_rgb": (1, 2, 3, 224, 224),
                "camera_right_ultrawide_rgb": (1, 2, 3, 224, 224)}
    model = TinyPromptModel(expected)
    model.noise_scheduler = DDIMScheduler(num_train_timesteps=50)
    loaded = LoadedModel(model=model, shape_meta=copy.deepcopy(RUN_001_SHAPE_META), uses_prompting=True, info={},
                         dataset_cfg={"_target_": "policy_fakes_torch.FakePromptDataset"})
    monkeypatch.setattr(CheckpointLoader, "load", lambda self, path, device, steps: loaded)
    demos = tmp_path / "demos.zarr.zip"
    demos.write_bytes(b"")
    out = io.StringIO()
    a = bench.parse_args(["--checkpoint", "b.ckpt", "--prompt-dataset", str(demos), "--prompt-split", "all",
                          "--device", "cpu", "--infers", "2", "--warmup", "0", "--out", str(tmp_path)])
    assert bench.run(a, out) == 0
    records = [json.loads(line) for line in (tmp_path / "bench.jsonl").read_text().splitlines()]
    assert [r["what"] for r in records] == ["load", "prompt", "infer"]
    assert records[1]["task_name"] == "other" and records[1]["steps"] == 300 and records[1]["split"] == "all"
    assert model.events == ["prompt", "predict", "predict"]


def test_prompt_dataset_with_a_plain_checkpoint_refused(monkeypatch, tmp_path):
    pytest.importorskip("torch", reason="torch tests run in the torch environment")
    from policy_fakes import plain_diffusion_shape_meta
    from policy_fakes_torch import TinyDiffusionModel

    from policy_deploy.policy.checkpoint_loader import CheckpointLoader, LoadedModel
    loaded = LoadedModel(model=TinyDiffusionModel({}), shape_meta=plain_diffusion_shape_meta(), uses_prompting=False,
                         info={})
    monkeypatch.setattr(CheckpointLoader, "load", lambda self, path, device, steps: loaded)
    out = io.StringIO()
    a = bench.parse_args(["--checkpoint", "b.ckpt", "--prompt-dataset", str(tmp_path / "d.zarr.zip"), "--device",
                          "cpu", "--out", str(tmp_path)])
    assert bench.run(a, out) == 1
    assert out.getvalue().splitlines()[-1] == ("FAIL: --prompt-dataset is set but the checkpoint does not use "
                                               "prompting (shape_meta.use_prompting: false): remove --prompt-dataset")
    assert not (tmp_path / "bench.jsonl").exists()
