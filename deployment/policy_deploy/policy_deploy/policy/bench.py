# new (STRUCTURE.md 11.1); the model building follows light_workspace.py (no backbone download) and the timing
# follows probe.py (mean / p50 / p95 / max); --prompt-dataset prompts a prompting checkpoint with its longest demo
# (PromptSource, as PromptableDiffusionPolicy's warm-up) and then times it; refused with a plain checkpoint
"""`python -m policy_deploy.policy.bench`: time predict_action on this laptop without a socket (STRUCTURE.md 11.1).

    python -m policy_deploy.policy.bench [--config umi_policy_dunet | --checkpoint PATH [--prompt-dataset ZIP
           [--prompt-split train]]] [--steps 16,8,4] [--precision fp32,bf16] [--infers 500] [--warmup 3]
           [--device cuda] [--out DIR]

--config NAME (default umi_policy_dunet): composes behavior_prompting's training config NAME with hydra, builds
cfg.model (the plain diffusion policy: TimmObsEncoder on vit_base_patch16_clip_224.openai, UNet [256, 512, 1024],
DDIM 16 steps) at random weights without downloading the backbone, and feeds neutral_observation() through
ModelRunner's own conversion. Timing does not depend on the weights.

--checkpoint PATH: loads through CheckpointLoader and reports the load time, the host RAM peak and the VRAM of the
real weights. A plain checkpoint is then timed; a prompting one only with --prompt-dataset (the processed .zarr.zip of
its demos; --prompt-split train | val | all): prompted once with the longest demo (one "prompt" line), then timed.

For each steps x precision: warm-up times, mean/p50/p95/max over --infers sustained calls,
torch.cuda.max_memory_allocated/reserved, and nvidia-smi's used memory, power draw, SM clock and temperature at the end
(throttling at the 35 W cap). One line each, and bench.jsonl in --out (when given).

Torch environment only. torch & co are imported inside the functions, so the CLI and the pure helpers import without
torch (the light env's tests).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from policy_deploy.policy.errors import PolicyLoadError
from policy_deploy.policy.neutral_observation import neutral_observation
from policy_deploy.policy.settings import PRECISIONS, PROMPT_SPLITS

CONFIG_MODULE = "behavior_prompting.train_network.config"
DEFAULT_CONFIG = "umi_policy_dunet"
SMI_QUERY = "memory.used,memory.total,power.draw,power.limit,clocks.sm,temperature.gpu"


# ------------------------------------------------------------------------------------------------ pure helpers

def int_list(text: str) -> list[int]:
    """"16,8,4" -> [16, 8, 4]; each >= 1."""
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from None
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(f"expected integers >= 1, got {text!r}")
    return values


def precision_list(text: str) -> list[str]:
    """"fp32,bf16" -> ["fp32", "bf16"]; each in PRECISIONS."""
    values = [v.strip() for v in text.split(",") if v.strip()]
    bad = [v for v in values if v not in PRECISIONS]
    if not values or bad:
        raise argparse.ArgumentTypeError(f"expected a comma-separated list of {', '.join(PRECISIONS)}, got {text!r}")
    return values


def timing_ms(seconds: Sequence[float]) -> dict[str, float]:
    """{"mean_ms", "p50_ms", "p95_ms", "max_ms"} of the given durations (seconds)."""
    a = np.asarray(seconds, dtype=np.float64) * 1e3
    if a.size == 0:
        return {}
    return {"mean_ms": round(float(a.mean()), 2), "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2), "max_ms": round(float(a.max()), 2)}


def _num(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def nvidia_smi_now(run: Callable[[list[str]], str] | None = None) -> dict[str, float | None]:
    """{"smi_used_mb", "smi_total_mb", "power_draw_w", "power_limit_w", "sm_clock_mhz", "temperature_c"} of the first
    GPU, or {} without nvidia-smi."""
    def default(args: list[str]) -> str:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            raise FileNotFoundError("nvidia-smi")
        return subprocess.run([exe, *args], capture_output=True, text=True, timeout=5.0, check=True).stdout
    try:
        out = (run or default)([f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits"])
    except Exception:
        return {}
    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        return {}
    parts = lines[0].split(",")
    keys = ("smi_used_mb", "smi_total_mb", "power_draw_w", "power_limit_w", "sm_clock_mhz", "temperature_c")
    return {k: _num(v) for k, v in zip(keys, parts)}


def host_ram_peak_mb() -> float:
    """This process's peak resident memory (Linux ru_maxrss is in KiB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def format_line(record: Mapping[str, Any]) -> str:
    """One readable line of a bench record."""
    if record.get("what") == "prompt":
        return (f"prompt: task {record.get('task_name')!r}, index {record.get('prompt_index')} ({record.get('steps')} "
                f"steps, {record.get('split')} split): {record.get('prompt_s')} s, VRAM allocated "
                f"{record.get('max_memory_allocated_mb')} MB")
    if record.get("what") == "load":
        return (f"load: {record.get('source')}: {record.get('load_s')} s, host RAM peak {record.get('host_ram_peak_mb')}"
                f" MB, VRAM allocated {record.get('max_memory_allocated_mb')} MB, reserved "
                f"{record.get('max_memory_reserved_mb')} MB, prompting {record.get('uses_prompting')}")
    warm = ", ".join(f"{t * 1e3:.0f}" for t in record.get("warmup_s", []))
    return (f"steps {record.get('steps')} {record.get('precision')}: mean {record.get('mean_ms')} ms, p50 "
            f"{record.get('p50_ms')}, p95 {record.get('p95_ms')}, max {record.get('max_ms')} ms over "
            f"{record.get('infers')} (warm-up {warm} ms); VRAM allocated {record.get('max_memory_allocated_mb')} MB, "
            f"reserved {record.get('max_memory_reserved_mb')} MB; nvidia-smi {record.get('smi_used_mb')} MiB, "
            f"{record.get('power_draw_w')} W, {record.get('sm_clock_mhz')} MHz, {record.get('temperature_c')} C")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m policy_deploy.policy.bench", description=__doc__.split("\n\n")[0])
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--config", default=None,
                        help=f"behavior_prompting training config at random weights (default {DEFAULT_CONFIG})")
    source.add_argument("--checkpoint", type=Path, default=None, help="a training checkpoint (.ckpt)")
    ap.add_argument("--prompt-dataset", type=Path, default=None,
                    help="with a prompting --checkpoint: the processed .zarr.zip of its demos (prompts with the "
                         "longest demo, then times)")
    ap.add_argument("--prompt-split", choices=PROMPT_SPLITS, default="train",
                    help="the demos to prompt from (default train)")
    ap.add_argument("--steps", type=int_list, default=[16], help="diffusion steps to time, e.g. 16,8,4")
    ap.add_argument("--precision", type=precision_list, default=["fp32"], help="e.g. fp32,bf16")
    ap.add_argument("--infers", type=int, default=500, help="sustained timed calls per combination")
    ap.add_argument("--warmup", type=int, default=3, help="warm-up calls per combination (timed and printed, not in the statistics)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--out", type=Path, default=None, help="write bench.jsonl here")
    a = ap.parse_args(argv)
    if a.checkpoint is None and a.config is None:
        a.config = DEFAULT_CONFIG
    if a.prompt_dataset is not None and a.checkpoint is None:
        ap.error("--prompt-dataset needs --checkpoint")
    if a.infers < 1 or a.warmup < 0:
        ap.error("--infers must be >= 1 and --warmup >= 0")
    return a


# ----------------------------------------------------------------------------------------------------- torch

def _key_map():
    from policy_deploy.contracts import load_config
    from policy_deploy.policy.key_map import KeyMap
    return KeyMap.from_config(load_config("policy", None)["key_map"])


def _training_timing() -> dict[str, int]:
    from policy_deploy.contracts import load_config
    return dict(load_config("policy", None)["training_timing"])


def build_from_config(name: str, device: str):
    """(model on the device in eval mode, shape_meta) from behavior_prompting's training config `name`, at random
    weights. timm.create_model is called with pretrained=False (no download; the same architecture), as in
    light_workspace.py, so every pretrained-dependent branch of the encoder is the trained one."""
    import hydra
    import omegaconf
    import timm
    import torch

    if not omegaconf.OmegaConf.has_resolver("eval"):
        omegaconf.OmegaConf.register_new_resolver("eval", eval)
    with hydra.initialize_config_module(config_module=CONFIG_MODULE, version_base=None):
        cfg = hydra.compose(config_name=name)
    orig = timm.create_model

    def create_model_no_download(*args, **kwargs):
        kwargs["pretrained"] = False
        return orig(*args, **kwargs)

    timm.create_model = create_model_no_download
    try:
        model = hydra.utils.instantiate(cfg.model)
    finally:
        timm.create_model = orig
    shape_meta = omegaconf.OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    model = model.eval().to(torch.device(device))
    if hasattr(model, "reset"):
        model.reset()
    return model, shape_meta


def make_infer(model, shape_meta: Mapping, device: str, precision: str) -> tuple[Callable[[], None], str]:
    """A no-argument call that runs one predict_action, and how its inputs were made. Through ModelRunner (the
    deployment's own conversion of neutral_observation()) when the key map fits the config's shape_meta; otherwise
    zeros of the shape_meta shapes (the timing is the same; the line says so)."""
    import torch

    from policy_deploy.policy.model_runner import ModelRunner
    try:
        runner = ModelRunner(model, shape_meta, _key_map(), torch.device(device), precision=precision)
    except PolicyLoadError as e:
        tensors = {}
        for key, attr in shape_meta["obs"].items():
            if attr.get("ignore_by_policy", False):
                continue
            shape = (1, int(attr.get("horizon", 2)), *[int(s) for s in attr["shape"]])
            tensors[key] = torch.zeros(shape, dtype=torch.float32, device=torch.device(device))

        def raw() -> None:
            autocast = (torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
                        if precision == "bf16" else contextlib.nullcontext())
            with torch.inference_mode(), autocast:
                out = model.predict_action(tensors)
            out["action_pred"].float().cpu()                 # waits for the GPU, as ModelRunner does

        return raw, f"zeros of shape_meta (key map does not fit: {e.what[:120]})"
    obs = neutral_observation(0.05)
    return (lambda: runner.run(obs)), "neutral_observation through ModelRunner"


def time_combination(model, shape_meta: Mapping, device: str, steps: int, precision: str, infers: int,
                     warmup: int) -> dict[str, Any]:
    import torch

    from policy_deploy.policy.checkpoint_loader import CheckpointLoader
    CheckpointLoader.apply_diffusion_steps(model, steps)
    infer, inputs = make_infer(model, shape_meta, device, precision)
    cuda = device == "cuda" and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    warm = []
    for _ in range(warmup):
        t0 = time.perf_counter()
        infer()
        warm.append(time.perf_counter() - t0)
    times = []
    for _ in range(infers):
        t0 = time.perf_counter()
        infer()
        times.append(time.perf_counter() - t0)
    record: dict[str, Any] = {"what": "infer", "steps": steps, "precision": precision, "device": device,
                              "infers": infers, "inputs": inputs, "warmup_s": [round(t, 4) for t in warm],
                              **timing_ms(times)}
    if cuda:
        record["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        record["max_memory_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    record.update(nvidia_smi_now())
    return record


def prompt_longest(loaded, path: Path, split: str, device: str) -> dict[str, Any]:
    """Prompt the loaded prompting model with the longest demo of `path` (as the policy's warm-up does)."""
    import torch

    from policy_deploy.policy.prompt_source import PromptSource
    source = PromptSource.load(path, split, loaded.dataset_cfg, loaded.model.get_training_split_info(), device)
    demo = source.demo(*source.longest())
    t0 = time.perf_counter()
    prompt = source.get(demo["task_name"], demo["prompt_index"])
    with torch.inference_mode():
        loaded.model.prompt(prompt)
    if device == "cuda":
        torch.cuda.synchronize()
    record: dict[str, Any] = {"what": "prompt", "split": split, **demo, "prompt_s": round(time.perf_counter() - t0, 3)}
    if device == "cuda":
        record["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return record


def run(a: argparse.Namespace, out=sys.stdout) -> int:
    import torch

    records: list[dict[str, Any]] = []

    def emit(record: dict[str, Any]) -> None:
        records.append(record)
        out.write(format_line(record) + "\n")
        out.flush()

    t0 = time.monotonic()
    if a.device == "cuda" and not torch.cuda.is_available():
        out.write("FAIL: CUDA not available (try --device cpu)\n")
        return 1
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if a.checkpoint is not None:
        from policy_deploy.policy.checkpoint_loader import CheckpointLoader
        loaded = CheckpointLoader(_training_timing()).load(a.checkpoint, a.device, None)
        model, shape_meta, plain = loaded.model, loaded.shape_meta, not loaded.uses_prompting
        source = str(a.checkpoint)
    else:
        model, shape_meta = build_from_config(a.config, a.device)
        plain, source = True, f"config {a.config} (random weights)"
    load = {"what": "load", "source": source, "load_s": round(time.monotonic() - t0, 2),
            "host_ram_peak_mb": host_ram_peak_mb(), "uses_prompting": not plain}
    if a.device == "cuda":
        load["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        load["max_memory_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    load.update(nvidia_smi_now())
    emit(load)
    if plain and a.prompt_dataset is not None:            # refused, as the policy refuses policy.prompt_dataset
        out.write("FAIL: --prompt-dataset is set but the checkpoint does not use prompting "
                  "(shape_meta.use_prompting: false): remove --prompt-dataset\n")
        return 1
    if not plain and a.prompt_dataset is not None:
        emit(prompt_longest(loaded, a.prompt_dataset, a.prompt_split, a.device))
        plain = True                                      # prompted: timed like a plain one
    if not plain:
        out.write("prompting checkpoint: inference not timed (give --prompt-dataset, or use --config)\n")
    else:
        for steps in a.steps:
            for precision in a.precision:
                emit(time_combination(model, shape_meta, a.device, steps, precision, a.infers, a.warmup))
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        with open(a.out / "bench.jsonl", "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("FAIL: torch is not importable: run the bench with python.torch, the torch environment", file=sys.stderr)
        return 1
    try:
        return run(a)
    except PolicyLoadError as e:
        print(f"FAIL: {e.what}: {e.check}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
