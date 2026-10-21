"""Random, temporal, and cross-source splits with leakage diagnostics."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, SOURCE_COL, TARGET_COL, TIME_COL
from src.features import domain_overlap


def _check_classes(frame: pd.DataFrame, name: str) -> None:
    counts = frame[TARGET_COL].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(f"{name} must contain both classes with at least 2 rows each: {counts.to_dict()}")


def split_random(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty dataset")
    if not 0.5 <= train_size < 1.0:
        raise ValueError("train_size must be within [0.5, 1)")
    if not 0.0 < val_size < 1.0 - train_size:
        raise ValueError("val_size leaves no room for test")
    test_size = 1.0 - train_size - val_size
    train, rest = train_test_split(
        df, test_size=val_size + test_size, stratify=df[TARGET_COL], random_state=random_state
    )
    val, test = train_test_split(
        rest,
        test_size=test_size / (val_size + test_size),
        stratify=rest[TARGET_COL],
        random_state=random_state,
    )
    train, val, test = (frame.reset_index(drop=True) for frame in (train, val, test))
    validate_splits(train, val, test, "random")
    return train, val, test


def split_temporal(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty dataset")
    if TIME_COL not in df.columns:
        raise ValueError(f"Temporal split requires '{TIME_COL}'")
    if not 0.5 <= train_size < 1.0:
        raise ValueError("train_size must be within [0.5, 1)")
    if not 0.0 < val_size < 1.0 - train_size:
        raise ValueError("val_size leaves no room for test")
    ordered = df.copy()
    ordered[TIME_COL] = pd.to_datetime(ordered[TIME_COL], errors="coerce")
    ordered = ordered.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    if len(ordered) < 300:
        raise ValueError("Temporal split needs at least 300 labelled URLs")
    n = len(ordered)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    train = ordered.iloc[:train_end].reset_index(drop=True)
    val = ordered.iloc[train_end:val_end].reset_index(drop=True)
    test = ordered.iloc[val_end:].reset_index(drop=True)
    validate_splits(train, val, test, "temporal")
    return train, val, test


def split_cross_source(
    df: pd.DataFrame,
    test_source: str,
    val_fraction: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-one-source-out: train/validate on other sources, test on one."""
    if test_source not in set(df[SOURCE_COL]):
        raise ValueError(f"Unknown source: {test_source}")
    test = df[df[SOURCE_COL] == test_source].reset_index(drop=True)
    remaining = df[df[SOURCE_COL] != test_source].reset_index(drop=True)
    if remaining[SOURCE_COL].nunique() < 1:
        raise ValueError("Cross-source split needs at least two sources")
    train, val = train_test_split(
        remaining,
        test_size=val_fraction,
        stratify=remaining[TARGET_COL],
        random_state=random_state,
    )
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    validate_splits(train, val, test, "cross_source")
    return train, val, test


def validate_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    split_type: str = "random",
) -> None:
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        if frame.empty:
            raise ValueError(f"{name} split is empty")
        _check_classes(frame, name)

    if "url_id" in train.columns:
        ids = [set(frame["url_id"]) for frame in (train, val, test)]
        if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("url_id overlap between splits")

    if split_type == "temporal":
        train_max = pd.to_datetime(train[TIME_COL]).max()
        val_min = pd.to_datetime(val[TIME_COL]).min()
        val_max = pd.to_datetime(val[TIME_COL]).max()
        test_min = pd.to_datetime(test[TIME_COL]).min()
        if not (train_max <= val_min <= val_max <= test_min):
            raise ValueError("Temporal split is not chronological")

    if split_type == "cross_source":
        train_sources = set(train[SOURCE_COL])
        test_sources = set(test[SOURCE_COL])
        if train_sources & test_sources:
            raise ValueError("Train and test share a source in cross-source split")
        if set(val[SOURCE_COL]) & test_sources:
            raise ValueError("Validation and test share a source in cross-source split")


def split_summary(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        summary[name] = {
            "n": int(len(frame)),
            "positive_rate": round(float(frame[TARGET_COL].mean()), 4),
            "sources": sorted(frame[SOURCE_COL].unique().tolist()),
            "date_min": str(pd.to_datetime(frame[TIME_COL]).min()),
            "date_max": str(pd.to_datetime(frame[TIME_COL]).max()),
            "domain_overlap_with_train": round(domain_overlap(train, frame), 4) if len(frame) else 0.0,
        }
    return summary
