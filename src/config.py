"""Configuration for the phishing URL generalisation study."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEMO_DATA_DIR = DATA_DIR / "demo"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
PREDICTION_DIR = RESULTS_DIR / "predictions"

RANDOM_STATE = 42
DEMO_N_URLS = 6000
DEMO_SOURCES = ("demo-feed-a", "demo-feed-b", "demo-feed-c")

SUPPORTED_SPLITS = ("random", "temporal", "cross_source")
SUPPORTED_MODELS = ("logistic_regression", "random_forest")
SUPPORTED_STRATEGIES = ("none", "class_weight")

TARGET_COL = "label"
URL_COL = "url"
SOURCE_COL = "source"
TIME_COL = "timestamp"
DOMAIN_COL = "domain"

SUSPICIOUS_TOKENS = (
    "login",
    "signin",
    "verify",
    "account",
    "update",
    "secure",
    "banking",
    "password",
    "confirm",
    "wallet",
    "invoice",
    "unlock",
    "support",
)

SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
STRATEGY_LABELS = {"none": "Tanpa bobot", "class_weight": "Class weight"}
SPLIT_LABELS = {
    "random": "Random split",
    "temporal": "Temporal split",
    "cross_source": "Cross-source (LOSO)",
}
