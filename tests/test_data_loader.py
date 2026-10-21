from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    DataNotFoundError,
    audit_dataset,
    load_dataset,
    validate_dataframe,
    write_demo_dataset,
)


def test_load_demo_metadata_and_audit():
    df, meta = load_dataset(source="demo", n_demo=600, seed=3)
    assert meta["is_demo"] is True
    assert "synthetic" in meta["warning"].lower()
    audit = audit_dataset(df)
    assert audit["n_rows"] == len(df)
    assert 0 < audit["positive_rate"] < 1
    assert audit["n_sources"] == 3


def test_validate_rejects_bad_frames():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame())
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame({"url": ["https://example.com"]}))


def test_validate_drops_bad_labels_and_dates():
    frame = pd.DataFrame(
        {
            "url": ["https://example.com/a", "https://example.com/b", "https://example.com/c"],
            "label": [0, 2, 1],
            "source": ["s"] * 3,
            "timestamp": ["2024-01-01", "2024-01-02", "not-a-date"],
        }
    )
    with pytest.raises(ValueError):
        validate_dataframe(frame)
    good = frame.iloc[:2].copy()
    good.loc[1, "label"] = 1
    out = validate_dataframe(good)
    assert len(out) == 2


def test_csv_roundtrip(tmp_path):
    df, _ = load_dataset(source="demo", n_demo=400, seed=2)
    path = write_demo_dataset(df, tmp_path)
    loaded, meta = load_dataset(source="csv", raw_dir=path)
    assert len(loaded) == len(df)
    assert "CSV" in meta["data_source"]


def test_missing_csv_raises(tmp_path):
    with pytest.raises(DataNotFoundError):
        load_dataset(source="csv", raw_dir=tmp_path / "missing.csv")
