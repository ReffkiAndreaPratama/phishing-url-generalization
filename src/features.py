"""Leakage-aware lexical URL features (no network calls)."""

from __future__ import annotations

import ipaddress
import math
import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from src.config import DOMAIN_COL, SOURCE_COL, SUSPICIOUS_TOKENS, TARGET_COL, TIME_COL, URL_COL

NUMERIC_FEATURES = [
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "digit_count",
    "digit_ratio",
    "special_count",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "slash_count",
    "at_count",
    "equals_count",
    "percent_count",
    "subdomain_count",
    "path_depth",
    "has_port",
    "has_ip",
    "has_punycode",
    "is_https",
    "has_query",
    "has_fragment",
    "double_slash_path",
    "is_shortener",
    "suspicious_token_count",
    "host_entropy",
    "path_entropy",
    "longest_token_length",
    "uppercase_ratio",
    "tld_length",
]
CATEGORICAL_FEATURES = ["tld"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = np.bincount(np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0] / max(1, counts.sum())
    return float(-(probabilities * np.log2(probabilities)).sum())


def _simple_domain(host: str) -> str:
    """Approximate registered domain using the last two labels.

    This deliberately avoids a Public Suffix List dependency; the limitation is
    documented. IP addresses are returned unchanged.
    """
    if not host:
        return ""
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    labels = [label for label in host.lower().strip(".").split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    return ".".join(labels[-2:])


def extract_url_features(url: str) -> dict[str, object]:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")
    candidate = url.strip()
    if len(candidate) > 4096:
        raise ValueError("URL is too long")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid URL: {exc}") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL has no hostname")

    path = parsed.path or ""
    query = parsed.query or ""
    lowered = candidate.lower()
    host_tokens = [token for token in host.replace(".", "-").split("-") if token]
    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0

    tld = host.rsplit(".", 1)[-1] if "." in host else "none"
    digits = sum(char.isdigit() for char in candidate)
    upper = sum(char.isupper() for char in candidate)
    special = sum(not char.isalnum() for char in candidate)
    tokens = _TOKEN_RE.findall(lowered)

    return {
        "url_length": len(candidate),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "digit_count": digits,
        "digit_ratio": digits / len(candidate),
        "special_count": special,
        "dot_count": candidate.count("."),
        "hyphen_count": candidate.count("-"),
        "underscore_count": candidate.count("_"),
        "slash_count": candidate.count("/"),
        "at_count": candidate.count("@"),
        "equals_count": candidate.count("="),
        "percent_count": candidate.count("%"),
        "subdomain_count": max(0, len(host.split(".")) - 2),
        "path_depth": len([part for part in path.split("/") if part]),
        "has_port": int(port is not None),
        "has_ip": has_ip,
        "has_punycode": int("xn--" in host),
        "is_https": int(parsed.scheme.lower() == "https"),
        "has_query": int(bool(query)),
        "has_fragment": int(bool(parsed.fragment)),
        "double_slash_path": int("//" in path),
        "is_shortener": int(host in {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rebrand.ly"}),
        "suspicious_token_count": sum(token in lowered for token in SUSPICIOUS_TOKENS),
        "host_entropy": round(_entropy(host), 4),
        "path_entropy": round(_entropy(path), 4),
        "longest_token_length": max((len(token) for token in host_tokens + tokens), default=0),
        "uppercase_ratio": upper / len(candidate),
        "tld_length": len(tld),
        "tld": tld,
        "domain": _simple_domain(host),
        "valid": True,
    }


def engineer_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add lexical features and a ``domain`` column; drop malformed URLs."""
    required = [URL_COL, TARGET_COL, SOURCE_COL, TIME_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    rows = []
    invalid = 0
    for _, row in df.iterrows():
        try:
            features = extract_url_features(str(row[URL_COL]))
        except ValueError:
            invalid += 1
            continue
        payload = row.to_dict()
        payload.update(features)
        payload[DOMAIN_COL] = features["domain"]
        rows.append(payload)
    if not rows:
        raise ValueError("No valid URLs remain after parsing")
    out = pd.DataFrame(rows)
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    out = out.dropna(subset=[TIME_COL]).reset_index(drop=True)
    if out[TARGET_COL].nunique() < 2:
        raise ValueError("Target has a single class after feature construction")
    return out, invalid


def feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns missing: {missing}")
    X = frame[FEATURE_COLUMNS].copy()
    y = frame[TARGET_COL].astype(int).copy()
    X.index = pd.RangeIndex(len(X))
    y.index = pd.RangeIndex(len(y))
    return X, y


def domain_overlap(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Fraction of test URLs whose approximate domain appeared in training."""
    if test.empty:
        raise ValueError("Test frame is empty")
    train_domains = set(train[DOMAIN_COL].astype(str))
    test_domains = test[DOMAIN_COL].astype(str)
    return float(test_domains.isin(train_domains).mean())
