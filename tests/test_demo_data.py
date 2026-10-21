from __future__ import annotations

from urllib.parse import urlsplit

import pandas as pd
import pytest

from src.config import TARGET_COL, TIME_COL
from src.demo_data import generate_demo_urls


def test_demo_is_deterministic():
    first = generate_demo_urls(n_urls=600, seed=5)
    second = generate_demo_urls(n_urls=600, seed=5)
    assert first.equals(second)


def test_demo_uses_reserved_domains_only(demo_urls):
    for url in demo_urls["url"].head(300):
        host = urlsplit(url).hostname or ""
        assert (
            host.endswith("example.com")
            or host.endswith("example.net")
            or host.endswith("example.org")
            or host.endswith("example.test")
            or host.startswith(("192.0.2.", "198.51.100.", "203.0.113."))
        ), url


def test_demo_has_sources_and_classes(demo_urls):
    assert demo_urls["source"].nunique() == 3
    assert set(demo_urls[TARGET_COL].unique()) == {0, 1}
    assert 0.2 < demo_urls[TARGET_COL].mean() < 0.5


def test_demo_timestamps_are_parseable(demo_urls):
    timestamps = pd.to_datetime(demo_urls[TIME_COL])
    assert timestamps.notna().all()
    assert timestamps.min() < timestamps.max()


def test_demo_rejects_too_few_rows():
    with pytest.raises(ValueError):
        generate_demo_urls(n_urls=10)
