"""Figures for the phishing generalisation study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from src.config import MODEL_LABELS, SOURCE_COL, SPLIT_LABELS, STRATEGY_LABELS, TARGET_COL

COLORS = ["#2563eb", "#f97316", "#0f766e", "#b91c1c", "#7c3aed"]
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    }
)


def _save(fig, path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_class_balance(frame: pd.DataFrame, path) -> Path:
    table = frame.groupby(SOURCE_COL)[TARGET_COL].agg(["mean", "size"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(table[SOURCE_COL], table["mean"], color=COLORS[: len(table)])
    for bar, (_, row) in zip(bars, table.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['mean']:.1%}\nn={int(row['size'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Proporsi label positif")
    ax.set_ylim(0, min(1.0, table["mean"].max() * 1.35 + 0.05))
    ax.set_title("Komposisi kelas per sumber DEMO")
    return _save(fig, path)


def plot_source_timeline(frame: pd.DataFrame, path) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    for i, (source, group) in enumerate(frame.groupby(SOURCE_COL)):
        ax.scatter(
            pd.to_datetime(group["timestamp"]),
            np.full(len(group), i),
            s=5,
            alpha=0.35,
            color=COLORS[i % len(COLORS)],
        )
        ax.text(pd.to_datetime(group["timestamp"]).min(), i + 0.18, source, fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Timestamp")
    ax.set_title("Sebaran waktu per sumber (DEMO)")
    return _save(fig, path)


def plot_metric_comparison(results: pd.DataFrame, path, metric: str = "pr_auc") -> Path:
    frame = results[results["split"].isin(["random", "temporal"])].copy()
    if frame.empty:
        raise ValueError("No random/temporal results available")
    frame["label"] = (
        frame["model"].map(MODEL_LABELS).fillna(frame["model"])
        + " / "
        + frame["strategy"].map(STRATEGY_LABELS).fillna(frame["strategy"])
    )
    pivot = frame.pivot_table(index="label", columns="split", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.4, max(3.0, 0.4 * len(pivot))))
    pivot.plot(kind="barh", ax=ax, color=[COLORS[0], COLORS[1]])
    ax.set_xlabel(metric)
    ax.set_title("Random vs temporal split")
    ax.legend(frameon=False, title="")
    return _save(fig, path)


def plot_cross_source_heatmap(results: pd.DataFrame, path, metric: str = "pr_auc") -> Path:
    frame = results[results["split"] == "cross_source"].copy()
    if frame.empty:
        raise ValueError("No cross-source results available")
    frame["label"] = (
        frame["model"].map(MODEL_LABELS).fillna(frame["model"])
        + " / "
        + frame["strategy"].map(STRATEGY_LABELS).fillna(frame["strategy"])
    )
    pivot = frame.pivot_table(index="label", columns="test_source", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.45 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=metric)
    ax.set_title("Leave-one-source-out: performa pada sumber yang tidak dilihat")
    return _save(fig, path)


def plot_pr_curve(y_true, y_prob, path, label: str = "Model") -> Path:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(recall, precision, color=COLORS[0], label=label)
    ax.axhline(y.mean(), color="#64748b", linestyle="--", linewidth=1, label=f"Baseline ({y.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test)")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_calibration(y_true, y_prob, path, n_bins: int = 10) -> Path:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if np.unique(y).size < 2:
        raise ValueError("Calibration requires both classes")
    fraction, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Sempurna")
    ax.plot(predicted, fraction, marker="o", color=COLORS[2], label="Model")
    ax.set_xlabel("Rata-rata probabilitas")
    ax.set_ylabel("Fraksi positif aktual")
    ax.set_title("Calibration curve (test)")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_domain_overlap(summaries: dict[str, dict], path) -> Path:
    rows = []
    for split, summary in summaries.items():
        for part in ["validation", "test"]:
            rows.append(
                {
                    "split": SPLIT_LABELS.get(split, split),
                    "part": part,
                    "overlap": summary.get(part, {}).get("domain_overlap_with_train", 0.0),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No split summaries available")
    pivot = frame.pivot(index="split", columns="part", values="overlap")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    pivot.plot(kind="bar", ax=ax, color=[COLORS[1], COLORS[0]])
    ax.set_ylabel("Fraksi domain test yang terlihat saat training")
    ax.set_ylim(0, 1.05)
    ax.set_title("Kebocoran domain antar split (DEMO)")
    ax.legend(frameon=False)
    return _save(fig, path)
