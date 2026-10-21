"""Run random, temporal, and leave-one-source-out URL experiments.

Example:
    python scripts/run_experiments.py --source demo
    python scripts/run_experiments.py --source csv --raw-dir data/raw/phishing_urls.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEMO_N_URLS,
    FIGURE_DIR,
    MODEL_LABELS,
    PREDICTION_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SOURCE_COL,
    SPLIT_LABELS,
    STRATEGY_LABELS,
    SUPPORTED_MODELS,
    SUPPORTED_STRATEGIES,
    TARGET_COL,
    TIME_COL,
    URL_COL,
)
from src.data_loader import audit_dataset, load_dataset, save_audit  # noqa: E402
from src.evaluate import (  # noqa: E402
    add_predictions,
    bootstrap_pr_auc,
    compute_metrics,
    save_json,
    summarise_results,
)
from src.features import domain_overlap, engineer_dataframe, feature_matrix  # noqa: E402
from src.splits import (  # noqa: E402
    split_cross_source,
    split_random,
    split_summary,
    split_temporal,
)
from src.train import build_model, fit_model, predict_proba, select_threshold  # noqa: E402
from src import visualization as viz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["demo", "csv"], default="demo")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--n-demo", type=int, default=DEMO_N_URLS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models", nargs="+", default=list(SUPPORTED_MODELS), choices=SUPPORTED_MODELS)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--skip-cross-source", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    figures = FIGURE_DIR if args.output == RESULTS_DIR else output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    df, meta = load_dataset(source=args.source, raw_dir=args.raw_dir, n_demo=args.n_demo, seed=args.seed)
    engineered, n_invalid = engineer_dataframe(df)
    audit = audit_dataset(engineered)
    audit["n_invalid_urls_dropped"] = int(n_invalid)
    save_audit(audit, output / "dataset_audit.json")
    print(f"Dataset: {meta['data_source']} | {len(engineered):,} valid URLs | positives {audit['positive_rate']:.3%}")

    records: list[dict] = []
    split_summaries: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}
    best: dict | None = None

    def run_configuration(
        split_name: str,
        test_source: str,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        model_name: str,
        strategy: str,
    ) -> None:
        nonlocal best
        X_train, y_train = feature_matrix(train)
        X_val, y_val = feature_matrix(val)
        X_test, y_test = feature_matrix(test)
        pipeline = build_model(model_name, strategy=strategy, random_state=args.seed)
        pipeline, fit_seconds = fit_model(pipeline, X_train, y_train)
        val_prob = predict_proba(pipeline, X_val)
        selection = select_threshold(y_val, val_prob)
        val_metrics = compute_metrics(y_val, val_prob, selection["threshold"])
        test_prob = predict_proba(pipeline, X_test)
        metrics = compute_metrics(y_test, test_prob, selection["threshold"])
        record = {
            "split": split_name,
            "test_source": test_source,
            "model": model_name,
            "strategy": strategy,
            "threshold": selection["threshold"],
            "fit_seconds": round(fit_seconds, 3),
            "n_train": int(len(train)),
            "n_validation": int(len(val)),
            "n_test": int(len(test)),
            "val_pr_auc": val_metrics["pr_auc"],
            "val_f1": val_metrics["f1"],
            "domain_overlap_test": round(domain_overlap(train, test), 4),
            **metrics,
        }
        records.append(record)
        rank = (float(val_metrics["pr_auc"] or -1.0), val_metrics["f1"])
        if best is None or rank > best["rank"]:
            best = {
                "rank": rank,
                "record": record,
                "test_frame": test,
                "test_prob": test_prob,
                "y_test": y_test,
            }

    # Random and temporal evaluation.
    for split_name, splitter in (("random", split_random), ("temporal", split_temporal)):
        train, val, test = splitter(engineered)
        summary = split_summary(train, val, test)
        split_summaries[split_name] = summary
        save_json(summary, output / f"split_summary_{split_name}.json")
        for model_name in args.models:
            for strategy in SUPPORTED_STRATEGIES:
                print(f"[{split_name}] {model_name} / {strategy}", flush=True)
                run_configuration(split_name, "all", train, val, test, model_name, strategy)

    # Leave-one-source-out evaluation.
    if not args.skip_cross_source:
        for source in sorted(engineered[SOURCE_COL].unique()):
            train, val, test = split_cross_source(engineered, source, random_state=args.seed)
            summary = split_summary(train, val, test)
            split_summaries[f"cross_source:{source}"] = summary
            save_json(summary, output / f"split_summary_cross_source_{source}.json")
            for model_name in args.models:
                for strategy in SUPPORTED_STRATEGIES:
                    print(f"[cross->{source}] {model_name} / {strategy}", flush=True)
                    run_configuration("cross_source", source, train, val, test, model_name, strategy)

    results = pd.DataFrame(records)
    results.to_csv(output / "metrics.csv", index=False)

    if best is not None:
        record = best["record"]
        predictions = add_predictions(best["test_frame"], best["test_prob"], record["threshold"])
        prediction_dir = output / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        keep = [column for column in [URL_COL, SOURCE_COL, TIME_COL, TARGET_COL, "probability", "predicted", "threshold", "error_type"] if column in predictions.columns]
        predictions[keep].to_csv(prediction_dir / "best_model_test_predictions.csv", index=False)
        record["pr_auc_ci95"] = bootstrap_pr_auc(best["y_test"], best["test_prob"], n_boot=args.bootstrap, seed=args.seed)
        save_json(record, output / "best_model.json")
        viz.plot_pr_curve(best["y_test"], best["test_prob"], figures / "pr_curve_best.png")
        viz.plot_calibration(best["y_test"], best["test_prob"], figures / "calibration_best.png")

    summary = summarise_results(results)
    summary["dataset"] = audit
    summary["labels"] = {
        "models": MODEL_LABELS,
        "strategies": STRATEGY_LABELS,
        "splits": SPLIT_LABELS,
    }
    summary["warning"] = meta["warning"]
    save_json(summary, output / "summary.json")

    viz.plot_class_balance(engineered, figures / "class_balance.png")
    viz.plot_source_timeline(engineered, figures / "source_timeline.png")
    viz.plot_metric_comparison(results, figures / "random_vs_temporal.png")
    if (results["split"] == "cross_source").any():
        viz.plot_cross_source_heatmap(results, figures / "cross_source_heatmap.png")
    viz.plot_domain_overlap(split_summaries, figures / "domain_overlap.png")

    print(f"Saved {len(results)} configurations to {output / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
