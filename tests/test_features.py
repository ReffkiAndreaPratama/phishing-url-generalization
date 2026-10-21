from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    domain_overlap,
    engineer_dataframe,
    extract_url_features,
    feature_matrix,
)
from src.config import TARGET_COL


def test_benign_features_are_lexical():
    features = extract_url_features("https://www.example.com/news/article?id=42")
    assert features["is_https"] == 1
    assert features["has_ip"] == 0
    assert features["suspicious_token_count"] == 0
    assert features["domain"] == "example.com"
    assert features["tld"] == "com"
    assert features["url_length"] > 0


def test_phishing_indicators():
    features = extract_url_features("http://198.51.100.7:8443/secure-login/verify.php?account=1")
    assert features["has_ip"] == 1
    assert features["has_port"] == 1
    assert features["is_https"] == 0
    assert features["suspicious_token_count"] >= 2
    assert features["domain"] == "198.51.100.7"


def test_query_and_path_counts():
    features = extract_url_features("https://example.org/a/b/c?x=1&y=2#frag")
    assert features["path_depth"] == 3
    assert features["has_query"] == 1
    assert features["has_fragment"] == 1
    assert features["equals_count"] == 2


def test_invalid_urls_raise():
    for url in ["", "not a url", "ftp://example.com/file", "https://", "http://[::1"]:
        with pytest.raises(ValueError):
            extract_url_features(url)


def test_engineer_dataframe_adds_features(engineered):
    assert len(engineered) > 0
    for column in FEATURE_COLUMNS + ["domain", "valid"]:
        assert column in engineered.columns
    assert engineered[FEATURE_COLUMNS].notna().all().all()


def test_engineer_dataframe_drops_invalid_rows(demo_urls):
    broken = demo_urls.head(20).copy()
    broken.loc[broken.index[0], "url"] = "not-a-url"
    frame, invalid = engineer_dataframe(broken)
    assert invalid == 1
    assert len(frame) == len(broken) - 1


def test_feature_matrix_shapes(engineered):
    X, y = feature_matrix(engineered)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(engineered)
    assert set(y.unique()) == {0, 1}


def test_domain_overlap(engineered):
    train = engineered.head(600)
    test = engineered.tail(100)
    overlap = domain_overlap(train, test)
    assert 0.0 <= overlap <= 1.0
    assert domain_overlap(train, train) == 1.0
    with pytest.raises(ValueError):
        domain_overlap(train, pd.DataFrame())
