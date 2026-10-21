from __future__ import annotations

import numpy as np
import pytest

from src.features import feature_matrix
from src.splits import split_random
from src.train import (
    build_model,
    fit_model,
    predict_proba,
    select_threshold,
)


def test_build_model_validation():
    with pytest.raises(ValueError):
        build_model("xgboost")
    with pytest.raises(ValueError):
        build_model("logistic_regression", strategy="smote")


def test_logistic_pipeline_fits_and_predicts(engineered):
    train, val, _ = split_random(engineered)
    X_train, y_train = feature_matrix(train)
    X_val, _ = feature_matrix(val)
    pipeline = build_model("logistic_regression", strategy="class_weight")
    fitted, seconds = fit_model(pipeline, X_train, y_train)
    proba = predict_proba(fitted, X_val)
    assert seconds >= 0
    assert proba.shape == (len(val),)
    assert np.isfinite(proba).all()
    assert proba.min() >= 0 and proba.max() <= 1


def test_threshold_selection_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    result = select_threshold(y, p)
    assert result["f1"] == pytest.approx(1.0)
    assert result["threshold"] == pytest.approx(0.5)


def test_threshold_selection_requires_both_classes():
    with pytest.raises(ValueError):
        select_threshold(np.zeros(10), np.linspace(0, 1, 10))


def test_fit_rejects_single_class(engineered):
    train, _, _ = split_random(engineered)
    X, _ = feature_matrix(train)
    with pytest.raises(ValueError):
        fit_model(build_model("logistic_regression"), X, np.zeros(len(X)))
