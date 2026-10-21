"""Models and validation-only threshold selection."""

from __future__ import annotations

import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, SUPPORTED_MODELS, SUPPORTED_STRATEGIES
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def build_model(
    model_name: str,
    strategy: str = "none",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model_name must be one of {SUPPORTED_MODELS}")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"strategy must be one of {SUPPORTED_STRATEGIES}")
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced" if strategy == "class_weight" else None,
            random_state=random_state,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample" if strategy == "class_weight" else None,
            n_jobs=-1,
            random_state=random_state,
        )
    return Pipeline([("preprocessor", make_preprocessor()), ("classifier", classifier)])


def fit_model(pipeline: Pipeline, X_train, y_train) -> tuple[Pipeline, float]:
    if len(X_train) == 0:
        raise ValueError("Training partition is empty")
    if np.asarray(y_train).min() == np.asarray(y_train).max():
        raise ValueError("Training target has a single class")
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    return pipeline, float(time.perf_counter() - start)


def predict_proba(pipeline: Pipeline, X) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("Fitted pipeline has no positive class")
    return proba[:, classes.index(1)]


def select_threshold(y_true, y_prob) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if y.size == 0 or y.size != p.size:
        raise ValueError("y_true and y_prob must be non-empty and aligned")
    if np.unique(y).size < 2:
        raise ValueError("Threshold selection requires both classes")
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"] + 1e-12 or (
            abs(f1 - best["f1"]) <= 1e-12 and abs(threshold - 0.5) < abs(best["threshold"] - 0.5)
        ):
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
    return best
