"""Test selection: GPU integration tests run only on request.

``pytest -m gpu`` runs tests marked ``gpu`` (tiny or small models on a local or Pod GPU); a plain
``pytest`` skips them. The marker lives here rather than in pyproject.toml because the container
launcher requires pyproject.toml to match the dependency image byte for byte.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU; run with `pytest -m gpu`")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if "gpu" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="GPU test: run with `pytest -m gpu`")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
