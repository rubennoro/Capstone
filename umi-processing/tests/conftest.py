"""Shared pytest configuration for umi-processing's tests."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs a native binary on real recordings (minutes); deselect with "
                                       "-m 'not slow'")
