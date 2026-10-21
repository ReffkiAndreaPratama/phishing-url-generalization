"""Dataset loading, validation, and audit for phishing URL experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DEMO_DATA_DIR,
    DEMO_N_URLS,
    RANDOM_STATE,
    SOURCE_COL,
    TARGET_COL,
    TIME_COL,
    URL_COL,
)
from src.demo_data import generate_demo_urls, write_demo_csv


class DataNotFoundError(FileNotFoundError):
    pass


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = [URL_COL, TARGET_COL, SOURCE_COL, TIME_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")
    out = df.copy()
    out[URL_COL] = out[URL_COL].astype(str).str.strip()
    out = out[out[URL_COL].str.len() > 0]
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out[out[TARGET_COL].isin([0, 1])].copy()
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    out = out.dropna(subset=[TIME_COL])
    out[SOURCE_COL] = out[SOURCE_COL].fillna("unknown").astype(str)
    if "url_id" not in out.columns:
        out["url_id"] = [f"row-{i:08d}" for i in range(len(out))]
    out = out.drop_duplicates(subset=["url_id"]).reset_index(drop=True)
    if out[TARGET_COL].nunique() < 2:
        raise ValueError("Dataset must contain both classes")
    return out


def load_dataset(
    source: str = "demo",
    raw_dir: Path | str | None = None,
    n_demo: int = DEMO_N_URLS,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if source == "demo":
        frame = validate_dataframe(generate_demo_urls(n_urls=n_demo, seed=seed))
        meta = {
            "data_source": "DEMO",
            "is_demo": True,
            "warning": (
                "SYNTHETIC DEMO URLS: labels come from a lexical rule plus noise. "
                "Metrics are not evidence about real phishing detection."
            ),
            "n_rows": len(frame),
        }
        return frame, meta
    if source == "csv":
        raw = Path(raw_dir) if raw_dir is not None else DEMO_DATA_DIR
        path = raw / "phishing_urls.csv" if raw.is_dir() else raw
        if not path.exists():
            raise DataNotFoundError(f"Dataset not found: {path}")
        frame = validate_dataframe(pd.read_csv(path))
        meta = {
            "data_source": f"CSV:{path.name}",
            "is_demo": "demo" in path.name.lower(),
            "warning": "Verify provenance, licence, and label quality of this local dataset.",
            "n_rows": len(frame),
        }
        return frame, meta
    raise ValueError("source must be 'demo' or 'csv'")


def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        raise ValueError("Cannot audit an empty dataset")
    by_source = (
        df.groupby(SOURCE_COL)[TARGET_COL]
        .agg(["size", "mean"])
        .reset_index()
        .rename(columns={"size": "n", "mean": "positive_rate"})
    )
    return {
        "n_rows": int(len(df)),
        "n_unique_urls": int(df[URL_COL].nunique()),
        "n_duplicate_urls": int(df[URL_COL].duplicated().sum()),
        "n_sources": int(df[SOURCE_COL].nunique()),
        "sources": sorted(df[SOURCE_COL].unique().tolist()),
        "positive_rate": round(float(df[TARGET_COL].mean()), 4),
        "date_min": str(df[TIME_COL].min()),
        "date_max": str(df[TIME_COL].max()),
        "by_source": by_source.round(4).to_dict(orient="records"),
        "warning": "DEMO labels are synthetic if data_source is DEMO.",
    }


def save_audit(audit: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")


def write_demo_dataset(frame: pd.DataFrame, directory: Path | str = DEMO_DATA_DIR) -> Path:
    return write_demo_csv(frame, Path(directory) / "phishing_urls.csv")
