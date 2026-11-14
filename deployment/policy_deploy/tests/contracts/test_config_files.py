"""The config files and the C4 limits file: every config/*.yaml loads, defines no contract key, and C4Limits is
strict (STRUCTURE.md 2.5, 3.3)."""
from __future__ import annotations

import pytest

from policy_deploy.contracts import C4Limits, paths
from policy_deploy.contracts.config_merge import load_config, read_config
from policy_deploy.contracts.station import refused_keys


@pytest.mark.parametrize("name", paths.CONFIG_NAMES)
def test_every_config_loads_without_contract_keys(name):
    assert isinstance(load_config(name, None), dict)
    assert refused_keys(read_config(name)) == []


def test_paths_need_an_editable_install(monkeypatch, tmp_path):
    (tmp_path / "pkg").mkdir()
    monkeypatch.setattr(paths, "_package_file", lambda: tmp_path / "pkg" / "__init__.py")
    with pytest.raises(FileNotFoundError, match="installed editable"):
        paths.folder()


def test_documented_defaults():
    frames = load_config("frames", None)
    assert frames["select"]["pair_max_age_ms"] == 60 and frames["select"]["pair_max_lead_ms"] == 25
    assert frames["select"]["pair_max_reuse"] == 2
    manager = load_config("manager", None)
    assert manager["timeouts"]["pc_homing_wait_s"] == 60.0 and manager["timeouts"]["home_s"] == 90.0
    assert manager["clock"]["min_samples"] == 10 and manager["clock"]["future_state_ms"] == 20.0
    launcher = load_config("launcher", None)
    assert launcher["ready_timeout_s"]["manager"] == 130.0 and launcher["fake_pc"] is True
    policy = load_config("policy", None)
    assert policy["precision"] == "fp32" and policy["mock_kind"] == "hold_still"


def test_c4_limits_strict(tmp_path):
    lim = C4Limits.load()
    assert (lim.anchor_row_tol, lim.rot6d_norm_tol, lim.rot6d_dot_max) == (1.0e-5, 0.1, 0.1)
    assert C4Limits.load() is lim
    for text, match in (("anchor_row_tol: 1.0e-5\nrot6d_norm_tol: 0.1\n", "missing key 'rot6d_dot_max'"),
                        ("anchor_row_tol: 1.0e-5\nrot6d_norm_tol: 0.1\nrot6d_dot_max: 0.1\nextra: 1\n", "unknown key"),
                        ("anchor_row_tol: 0\nrot6d_norm_tol: 0.1\nrot6d_dot_max: 0.1\n", "anchor_row_tol"),
                        ("anchor_row_tol: x\nrot6d_norm_tol: 0.1\nrot6d_dot_max: 0.1\n", "anchor_row_tol")):
        p = tmp_path / "l.yaml"
        p.write_text(text)
        with pytest.raises(ValueError, match=match):
            C4Limits.load(p)
