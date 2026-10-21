from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    add_predictions,
    bootstrap_pr_auc,
    compute_metrics,
    per_source_metrics,
    summarise_results,
)


def test_perfect_and_worst_metrics():
    perfect = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert perfect["precision"] == 1.0
    assert perfect["recall"] == 1.0
    assert perfect["pr_auc"] == 1.0
    worst = compute_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.5)
    assert worst["f1"] == 0.0
    assert worst["false_positive_rate"] == 1.0


def test_metric_validation():
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.5])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.5, np.nan])
    with pytest.raises(ValueError):
        compute_metrics([0, 2], [0.2, 0.8])


def test_bootstrap_interval_and_validation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.2, 200), 0, 1)
    low, high = bootstrap_pr_auc(y, p, n_boot=80, seed=1)
    assert 0 <= low <= high <= 1
    with pytest.raises(ValueError):
        bootstrap_pr_auc(np.zeros(10), np.linspace(0, 1, 10), n_boot=50)


def test_predictions_and_per_source_table():
    frame = pd.DataFrame(
        {
            "label": [0, 1, 1, 0],
            "source": ["a", "a", "b", "b"],
            "url": ["u1", "u2", "u3", "u4"],
        }
    )
    predicted = add_predictions(frame, np.array([0.1, 0.4, 0.9, 0.8]), 0.5)
    assert predicted["error_type"].tolist() == ["correct", "false_negative", "correct", "false_positive"]
    table = per_source_metrics(predicted)
    assert set(table["source"]) == {"a", "b"}


def test_summary_from_results():
    results = pd.DataFrame(
        [
            {"split": "random", "test_source": "all", "model": "logistic_regression", "strategy": "none",
             "val_pr_auc": 0.8, "pr_auc": 0.75, "f1": 0.7, "threshold": 0.4},
            {"split": "temporal", "test_source": "all", "model": "logistic_regression", "strategy": "none",
             "val_pr_auc": 0.6, "pr_auc": 0.55, "f1": 0.5, "threshold": 0.4},
            {"split": "cross_source", "test_source": "demo-feed-a", "model": "logistic_regression", "strategy": "none",
             "val_pr_auc": 0.7, "pr_auc": 0.4, "f1": 0.4, "threshold": 0.4},
            {"split": "cross_source", "test_source": "demo-feed-b", "model": "logistic_regression", "strategy": "none",
             "val_pr_auc": 0.7, "pr_auc": 0.6, "f1": 0.5, "threshold": 0.4},
        ]
    )
    summary = summarise_results(results)
    assert summary["temporal_drop"] == pytest.approx(0.2)
    assert summary["cross_source"]["worst_source"] == "demo-feed-a"
    assert summary["best_by_validation"]["split"] == "random"
