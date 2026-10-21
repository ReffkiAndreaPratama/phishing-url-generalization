from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo_data import generate_demo_urls  # noqa: E402
from src.features import engineer_dataframe  # noqa: E402


@pytest.fixture(scope="session")
def demo_urls():
    return generate_demo_urls(n_urls=900, seed=7)


@pytest.fixture(scope="session")
def engineered(demo_urls):
    frame, invalid = engineer_dataframe(demo_urls)
    assert invalid == 0
    return frame
