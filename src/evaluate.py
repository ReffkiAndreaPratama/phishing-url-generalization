"""Metrics, per-source breakdowns, and bootstrap uncertainty."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import SOURCE_COL, TARGET_COL


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if y.size == 0 or y.size != p.size:
        raise ValueError("y_true and y_prob must be non-empty and aligned")
    if not np.isfinite(p).all():
        raise ValueError("y_prob contains non-finite values")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must be binary")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = np.unique(y).size == 2
    return {
        "n_test": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4) if two_classes else None,
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if two_classes else None,
        "brier": round(float(brier_score_loss(y, p)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)), 4) if fp + tn else 0.0,
        "false_negative_rate": round(float(fn / (fn + tp)), 4) if fn + tp else 0.0,
        "predicted_positive_rate": round(float(pred.mean()), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def bootstrap_pr_auc(
    y_true,
    y_prob,
    n_boot: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval for average precision."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if np.unique(y).size < 2:
        raise ValueError("Bootstrap requires both classes")
    if n_boot < 20:
        raise ValueError("n_boot must be at least 20")
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        scores.append(average_precision_score(y[idx], p[idx]))
    if not scores:
        return (float("nan"), float("nan"))
    return (round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4))


def per_source_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if SOURCE_COL not in predictions.columns:
        raise ValueError(f"Predictions need a '{SOURCE_COL}' column")
    rows = []
    for source, group in predictions.groupby(SOURCE_COL):
        metrics = compute_metrics(group[TARGET_COL], group["probability"], group["threshold"].iloc[0])
        rows.append({"source": source, **metrics})
    return pd.DataFrame(rows)


def add_predictions(frame: pd.DataFrame, probability, threshold: float) -> pd.DataFrame:
    if len(frame) != len(probability):
        raise ValueError("Prediction vector does not match frame length")
    out = frame.copy()
    out["probability"] = np.asarray(probability, dtype=float)
    out["predicted"] = (out["probability"] >= threshold).astype(int)
    out["threshold"] = float(threshold)
    out["error_type"] = "correct"
    out.loc[(out[TARGET_COL] == 1) & (out["predicted"] == 0), "error_type"] = "false_negative"
    out.loc[(out[TARGET_COL] == 0) & (out["predicted"] == 1), "error_type"] = "false_positive"
    return out


def save_json(payload: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def summarise_results(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {}
    safe = results.copy()
    summary: dict[str, Any] = {"n_configurations": int(len(safe))}
    if "val_pr_auc" in safe.columns:
        best = safe.sort_values(["val_pr_auc", "model"], ascending=[False, True]).iloc[0]
        summary["best_by_validation"] = {
            "split": best["split"],
            "test_source": best.get("test_source", "all"),
            "model": best["model"],
            "strategy": best["strategy"],
            "val_pr_auc": float(best["val_pr_auc"]),
            "test_pr_auc": float(best["pr_auc"]),
            "test_f1": float(best["f1"]),
            "threshold": float(best["threshold"]),
        }
    for split in ["random", "temporal"]:
        subset = safe[safe["split"] == split]
        if not subset.empty:
            summary[f"{split}_mean_pr_auc"] = round(float(subset["pr_auc"].mean()), 4)
    cross = safe[safe["split"] == "cross_source"]
    if not cross.empty:
        summary["cross_source"] = {
            "mean_pr_auc": round(float(cross["pr_auc"].mean()), 4),
            "worst_pr_auc": round(float(cross["pr_auc"].min()), 4),
            "worst_source": str(cross.sort_values("pr_auc").iloc[0]["test_source"]),
        }
    if {"random", "temporal"}.issubset(set(safe["split"])):
        random_mean = float(safe[safe["split"] == "random"]["pr_auc"].mean())
        temporal_mean = float(safe[safe["split"] == "temporal"]["pr_auc"].mean())
        summary["temporal_drop"] = round(random_mean - temporal_mean, 4)
    return summary
