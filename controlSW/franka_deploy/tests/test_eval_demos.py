"""Opt-in (env_bp + the prompt zarr): phase detection works on >= 43/45 firstreal demos."""
import pytest

from conftest import PROMPT_ZARR

pytestmark = pytest.mark.skipif(not PROMPT_ZARR, reason="needs FRANKA_DEPLOY_PROMPT_ZARR")


def test_phase_detection_on_demos():
    from franka_deploy.eval.demos import demo_phases, load_demos

    demos = load_demos(PROMPT_ZARR)
    ph = demo_phases(demos)
    ok = [p is not None and p["lifted"] for p in ph]
    assert len(demos) == 45 and sum(ok) >= 43, sum(ok)
    lifts = sorted(p["lift_height_m"] for p in ph if p)
    assert 0.08 < lifts[len(lifts) // 2] < 0.3     # median lift ~16 cm
