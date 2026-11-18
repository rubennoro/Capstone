"""Shared pytest configuration for umi-processing's tests."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs a native binary on real recordings (minutes); deselect with "
                                       "-m 'not slow'")


import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _analysis_in_tmp(tmp_path_factory):
    """No test writes into the real ~/data/analysis (umiproc.analysis), module-scoped fixtures included."""
    mp = pytest.MonkeyPatch()
    mp.setenv("UMIPROC_ANALYSIS", str(tmp_path_factory.mktemp("analysis")))
    yield
    mp.undo()
