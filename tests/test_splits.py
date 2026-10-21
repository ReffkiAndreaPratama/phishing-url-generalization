from __future__ import annotations

import pandas as pd
import pytest

from src.config import SOURCE_COL, TARGET_COL, TIME_COL
from src.splits import (
    split_cross_source,
    split_random,
    split_summary,
    split_temporal,
    validate_splits,
)


def test_random_split_sizes_and_disjoint_ids(engineered):
    train, val, test = split_random(engineered)
    assert abs(len(train) / len(engineered) - 0.7) < 0.02
    assert abs(len(val) / len(engineered) - 0.1) < 0.02
    assert set(train["url_id"]).isdisjoint(val["url_id"])
    assert set(train["url_id"]).isdisjoint(test["url_id"])


def test_temporal_split_is_chronological(engineered):
    train, val, test = split_temporal(engineered)
    assert pd.to_datetime(train[TIME_COL]).max() <= pd.to_datetime(val[TIME_COL]).min()
    assert pd.to_datetime(val[TIME_COL]).max() <= pd.to_datetime(test[TIME_COL]).min()


def test_cross_source_is_disjoint(engineered):
    source = sorted(engineered[SOURCE_COL].unique())[0]
    train, val, test = split_cross_source(engineered, source)
    assert set(test[SOURCE_COL]) == {source}
    assert source not in set(train[SOURCE_COL])
    assert source not in set(val[SOURCE_COL])


def test_validate_detects_id_overlap(engineered):
    train, val, test = split_random(engineered)
    leaked = pd.concat([train, val.head(1)], ignore_index=True)
    with pytest.raises(ValueError):
        validate_splits(leaked, val, test, "random")


def test_validate_detects_source_overlap(engineered):
    train, val, test = split_random(engineered)
    with pytest.raises(ValueError):
        validate_splits(train, val, test, "cross_source")


def test_split_summary_reports_domain_overlap(engineered):
    train, val, test = split_random(engineered)
    summary = split_summary(train, val, test)
    assert summary["test"]["domain_overlap_with_train"] >= 0
    assert summary["train"]["n"] == len(train)
    assert "sources" in summary["validation"]


def test_split_requires_enough_rows():
    small = pd.DataFrame(
        {
            "url_id": [f"u{i}" for i in range(20)],
            "url": ["https://example.com"] * 20,
            "label": [0, 1] * 10,
            "source": ["demo-feed-a"] * 20,
            "timestamp": pd.date_range("2024-01-01", periods=20),
        }
    )
    with pytest.raises(ValueError):
        split_temporal(small)
