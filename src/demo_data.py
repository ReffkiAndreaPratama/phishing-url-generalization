"""Generate synthetic DEMO URLs with controlled source shift.

Every URL uses reserved domains/documentation IP ranges (RFC 2606, RFC 5737).
None of them is a real phishing site. Labels come from a synthetic rule plus
noise, so metrics measure code behaviour, not real-world detection ability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEMO_N_URLS,
    DEMO_SOURCES,
    RANDOM_STATE,
    SOURCE_COL,
    SUSPICIOUS_TOKENS,
    TARGET_COL,
    TIME_COL,
    URL_COL,
)

BENIGN_WORDS = ("news", "shop", "docs", "blog", "catalog", "profile", "search", "article", "help")
PHISH_WORDS = ("secure", "account", "verify", "login", "update", "wallet", "invoice", "confirm")
DOC_IPS = ("192.0.2.", "198.51.100.", "203.0.113.")


def _random_word(rng: np.random.Generator) -> str:
    return BENIGN_WORDS[int(rng.integers(0, len(BENIGN_WORDS)))]


def _benign_url(rng: np.random.Generator, source_index: int) -> str:
    scheme = "https" if rng.random() > 0.04 else "http"
    host = f"www.example{['.com', '.net', '.org'][source_index % 3]}"
    path = "/".join(_random_word(rng) for _ in range(int(rng.integers(1, 4))))
    if rng.random() < 0.25:
        path += f"?id={int(rng.integers(1, 9999))}"
    return f"{scheme}://{host}/{path}"


def _phishing_url(rng: np.random.Generator, source_index: int) -> str:
    scheme = "http" if rng.random() < (0.55 if source_index == 1 else 0.3) else "https"
    token_a, token_b = rng.choice(PHISH_WORDS, size=2, replace=False)
    if source_index == 0:
        host = f"{token_a}-{token_b}.example.com"
        path = f"/{token_a}/{int(rng.integers(1000, 99999))}"
    elif source_index == 1:
        ip = DOC_IPS[int(rng.integers(0, len(DOC_IPS)))] + str(int(rng.integers(1, 255)))
        port = int(rng.integers(8000, 9999))
        host = f"{ip}:{port}"
        path = f"/{token_a}/{token_b}.php"
    else:
        sub = "-".join(rng.choice(PHISH_WORDS, size=3, replace=False))
        host = f"{sub}.{token_a}.example.test"
        path = f"/{'/'.join(str(rng.integers(100, 999)) for _ in range(3))}/{token_b}"
    extra = ""
    if rng.random() < 0.2:
        extra = f"?{token_b}={int(rng.integers(1, 9999))}&session={int(rng.integers(1, 9999))}"
    if rng.random() < 0.1:
        path = path.replace("/", "//", 1)
    if rng.random() < 0.08:
        path = f"/@{token_a}{path}"
    return f"{scheme}://{host}{path}{extra}"


def _is_suspicious(url: str) -> bool:
    lowered = url.lower()
    return any(token in lowered for token in SUSPICIOUS_TOKENS) or any(ip in lowered for ip in DOC_IPS)


def generate_demo_urls(n_urls: int = DEMO_N_URLS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    if n_urls < 300:
        raise ValueError("n_urls must be at least 300")
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_urls):
        source_index = i % len(DEMO_SOURCES)
        source = DEMO_SOURCES[source_index]
        # Synthetic label rule: lexical phishing pattern + 8% label noise.
        intended_phish = rng.random() < 0.32
        if intended_phish:
            url = _phishing_url(rng, source_index)
        else:
            url = _benign_url(rng, source_index)
        label = int(_is_suspicious(url))
        if rng.random() < 0.08:
            label = 1 - label
        day = int(rng.integers(0, 240))
        timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day, minutes=int(rng.integers(0, 1440)))
        rows.append(
            {
                "url_id": f"demo-{i:06d}",
                URL_COL: url,
                TARGET_COL: label,
                SOURCE_COL: source,
                TIME_COL: timestamp,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_demo_csv(frame: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
