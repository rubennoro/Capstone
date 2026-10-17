"""Light workspace (no checkpoint needed): registration and import isolation, each in a fresh interpreter."""
import os
import subprocess
import sys

import pytest

from conftest import PKG_ROOT

pytest.importorskip("torch")
pytest.importorskip("behavior_prompting")


def run_py(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(PKG_ROOT), HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)


def test_install_resolves_target_without_heavy_imports():
    r = run_py(
        "import hydra, sys\n"
        "from franka_deploy.inference import light_workspace as lw\n"
        "cls = lw.install(); assert lw.install() is cls\n"
        "import behavior_prompting.deployment.detached_inference\n"
        "got = hydra.utils.get_class(lw.UPSTREAM_MODULE + '.TrainPolicyWorkspace')\n"
        "assert got is cls, got\n"
        "print('HEAVY', lw.loaded_heavy_modules())\n"
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "HEAVY []" in r.stdout


def test_install_refuses_after_real_import():
    r = run_py(
        "import sys, types\n"
        "sys.modules['behavior_prompting.train_network.workspace.train_policy_workspace'] = types.ModuleType('x')\n"
        "from franka_deploy.inference import light_workspace as lw\n"
        "try:\n    lw.install()\nexcept RuntimeError as e:\n    print('REFUSED', e)\n"
    )
    assert "REFUSED" in r.stdout, r.stdout + r.stderr[-2000:]


def test_normalization_check_rejects_wrong_constants():
    r = run_py(
        "import torch, torchvision\n"
        "from franka_deploy.inference import light_workspace as lw\n"
        "from behavior_prompting.train_network.model.vision.transformer_obs_encoder import TransformerObsEncoder\n"
        "class Fake(TransformerObsEncoder):\n"
        "    def __init__(self, mean):\n"
        "        torch.nn.Module.__init__(self)\n"
        "        self.use_vision_norm = True; self.ordered_rgb_keys = ['a']; self.share_rgb_model = False\n"
        "        self.model_normalization_transform = torchvision.transforms.Normalize(mean, lw.CLIP_STD)\n"
        "print('OK', lw.check_vision_normalization(torch.nn.Sequential(Fake(lw.CLIP_MEAN))))\n"
        "try:\n    lw.check_vision_normalization(torch.nn.Sequential(Fake((0.485, 0.456, 0.406))))\n"
        "except AssertionError as e:\n    print('REJECTED')\n"
    )
    assert "OK" in r.stdout and "REJECTED" in r.stdout, r.stdout + r.stderr[-2000:]
