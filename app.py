"""Streamlit dashboard for phishing URL generalisation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    FIGURE_DIR,
    MODEL_LABELS,
    RESULTS_DIR,
    SOURCE_COL,
    SPLIT_LABELS,
    STRATEGY_LABELS,
    TARGET_COL,
    TIME_COL,
    URL_COL,
)
from src.data_loader import audit_dataset, load_dataset
from src.features import FEATURE_COLUMNS, engineer_dataframe, extract_url_features

st.set_page_config(page_title="Phishing URL Generalization", page_icon="🔐", layout="wide")

PAGES = [
    "Ringkasan",
    "Komposisi sumber",
    "Inspektur fitur URL",
    "Hasil eksperimen",
    "Generalisasi & keterbatasan",
    "Unduh",
]


@st.cache_data(show_spinner="Mengekstrak fitur URL...")
def get_data(n_demo: int):
    df, meta = load_dataset(source="demo", n_demo=n_demo)
    engineered, invalid = engineer_dataframe(df)
    return engineered, meta, audit_dataset(engineered), invalid


@st.cache_data(ttl=30)
def get_results():
    path = RESULTS_DIR / "metrics.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(ttl=30)
def get_json(name: str):
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def figure(name: str, caption: str) -> None:
    path = FIGURE_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"`{name}` belum tersedia. Jalankan `python scripts/run_experiments.py --source demo`.")


def page_overview(df: pd.DataFrame, meta: dict, audit: dict, invalid: int) -> None:
    st.title("🔐 Generalisasi Deteksi URL Phishing")
    st.caption("Random split, temporal split, dan cross-source (leave-one-source-out).")
    st.error(
        "**DATA DEMO SINTETIS.** URL memakai domain/IP dokumentasi (RFC 2606/5737), "
        "bukan situs phishing nyata. Label dibuat oleh aturan leksikal + noise. "
        "Metrik tidak boleh dikutip sebagai kemampuan deteksi dunia nyata."
    )
    cols = st.columns(4)
    cols[0].metric("URL valid", f"{len(df):,}")
    cols[1].metric("Positif", f"{int(df[TARGET_COL].sum()):,}", f"{audit['positive_rate']:.1%}")
    cols[2].metric("Sumber", audit["n_sources"])
    cols[3].metric("URL tidak valid dibuang", invalid)

    st.subheader("Periode dan sumber")
    st.write(f"Timestamp: **{audit['date_min'][:10]}** sampai **{audit['date_max'][:10]}**")
    st.dataframe(
        pd.DataFrame(audit["by_source"]).rename(
            columns={"source": "Sumber", "n": "Jumlah", "positive_rate": "Proporsi positif"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Pertanyaan penelitian")
    st.markdown(
        "- Seberapa optimistis PR-AUC pada random split dibanding temporal split?\n"
        "- Seberapa jauh performa turun ketika model diuji pada **sumber yang belum pernah dilihat**?\n"
        "- Apakah domain yang sama muncul di train dan test (indikasi kebocoran)?\n"
        "- Apakah class weighting membantu recall tanpa merusak precision?"
    )

    results = get_results()
    if results is None:
        st.info(
            "Belum ada hasil. Jalankan:\n\n"
            "```bash\npython scripts/run_experiments.py --source demo\nstreamlit run app.py\n```"
        )
    else:
        summary = get_json("summary.json") or {}
        best = summary.get("best_by_validation")
        if best:
            c = st.columns(4)
            c[0].metric("Model (validation)", MODEL_LABELS.get(best["model"], best["model"]))
            c[1].metric("Test PR-AUC", f"{best['test_pr_auc']:.3f}")
            c[2].metric("Test F1", f"{best['test_f1']:.3f}")
            c[3].metric("Threshold", f"{best['threshold']:.2f}")
        st.caption("Metrik dipilih pada validation; test hanya dievaluasi sekali per konfigurasi.")


def page_sources(df: pd.DataFrame, audit: dict) -> None:
    st.title("Komposisi sumber")
    st.dataframe(
        df.groupby(SOURCE_COL)[TARGET_COL]
        .agg(["size", "mean"])
        .rename(columns={"size": "URL", "mean": "Proporsi positif"}),
        use_container_width=True,
    )
    figure("class_balance.png", "Proporsi kelas per sumber DEMO")
    figure("source_timeline.png", "Sebaran timestamp per sumber")

    st.subheader("Duplikasi dan domain")
    st.write(f"URL unik: **{audit['n_unique_urls']:,}** dari **{audit['n_rows']:,}** baris.")
    st.write(f"URL duplikat: **{audit['n_duplicate_urls']:,}**")
    st.caption(
        "Domain didekati dengan dua label terakhir (tanpa Public Suffix List). "
        "Domain bersama antara train dan test dapat membuat random split terlalu optimistis."
    )
    domains = df["domain"].value_counts().head(12)
    st.bar_chart(domains)


def page_inspector() -> None:
    st.title("Inspektur fitur URL")
    st.write(
        "Ekstraksi fitur sepenuhnya leksikal dan **tanpa jaringan** (tidak ada DNS, WHOIS, "
        "atau pengambilan halaman). Ini menjaga reproducibility dan menghindari fitur yang "
        "tidak tersedia saat deployment."
    )
    url = st.text_input("URL", value="https://secure-login.example.test/account/verify?id=123")
    try:
        features = extract_url_features(url)
        frame = pd.DataFrame(
            {"Fitur": FEATURE_COLUMNS, "Nilai": [features[column] for column in FEATURE_COLUMNS]}
        )
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.caption(f"Domain perkiraan: `{features['domain']}`")
    except ValueError as exc:
        st.error(f"URL tidak valid: {exc}")

    st.warning(
        "Fitur leksikal mudah diganggu penyerang (mis. domain pendek, HTTPS gratis). "
        "Model ini bukan pengganti blocklist, sandbox, atau analisis konten."
    )


def page_results(results: pd.DataFrame) -> None:
    st.title("Hasil eksperimen")
    split = st.selectbox(
        "Jenis split",
        [value for value in ["random", "temporal", "cross_source"] if value in set(results["split"])],
        format_func=lambda value: SPLIT_LABELS[value],
    )
    subset = results[results["split"] == split].copy()
    if split == "cross_source":
        source = st.selectbox("Sumber uji", sorted(subset["test_source"].unique()))
        subset = subset[subset["test_source"] == source]
    columns = [
        "test_source", "model", "strategy", "threshold", "val_pr_auc", "precision",
        "recall", "f1", "pr_auc", "roc_auc", "brier", "domain_overlap_test",
    ]
    view = subset[[column for column in columns if column in subset.columns]].copy()
    view["model"] = view["model"].map(lambda value: MODEL_LABELS.get(value, value))
    view["strategy"] = view["strategy"].map(lambda value: STRATEGY_LABELS.get(value, value))
    st.dataframe(view, hide_index=True, use_container_width=True)

    if split in {"random", "temporal"}:
        figure("random_vs_temporal.png", "Perbandingan random dan temporal split")
    if split == "cross_source":
        figure("cross_source_heatmap.png", "PR-AUC pada sumber yang tidak dilihat saat training")

    best = get_json("best_model.json")
    if best:
        st.subheader("Model terbaik (berdasarkan validation)")
        st.json(best, expanded=False)


def page_generalisation(results: pd.DataFrame | None, audit: dict) -> None:
    st.title("Generalisasi & keterbatasan")
    if results is not None:
        cross = results[results["split"] == "cross_source"]
        if not cross.empty:
            c = st.columns(3)
            c[0].metric("Rata-rata PR-AUC cross-source", f"{cross['pr_auc'].mean():.3f}")
            c[1].metric("PR-AUC terburuk", f"{cross['pr_auc'].min():.3f}")
            c[2].metric("Domain overlap rata-rata", f"{cross['domain_overlap_test'].mean():.1%}")
        figure("domain_overlap.png", "Fraksi domain test yang sudah terlihat saat training")

    st.subheader("Keterbatasan yang harus dinyatakan")
    st.markdown(
        "- **DEMO**: label sintetis, domain dokumentasi, dan noise buatan; bukan data phishing nyata.\n"
        "- Fitur hanya leksikal; tidak ada konten halaman, DNS, TLS, atau reputasi domain.\n"
        "- Domain didekati dua label terakhir; eTLD+1 sebenarnya memerlukan Public Suffix List.\n"
        "- Random split dapat membocorkan domain yang sama; cross-source menguji hal lain lagi.\n"
        "- Satu periode temporal tidak membuktikan stabilitas jangka panjang.\n"
        "- Threshold dipilih pada validation; metrik test tetap memiliki ketidakpastian sampling.\n"
        "- Klasifikasi URL bukan keputusan keamanan otonom; perlu analis dan sistem berlapis."
    )
    st.info(
        f"Dataset: {audit['n_rows']:,} URL, {audit['positive_rate']:.1%} positif, "
        f"{audit['n_sources']} sumber. Selalu laporkan konfigurasi split bersama metrik."
    )


def page_downloads(results, audit) -> None:
    st.title("Unduh hasil")
    if results is not None:
        st.download_button(
            "metrics.csv",
            results.to_csv(index=False).encode("utf-8"),
            "metrics.csv",
            "text/csv",
            use_container_width=True,
        )
    for name in ["summary.json", "dataset_audit.json", "best_model.json"]:
        path = RESULTS_DIR / name
        if path.exists():
            st.download_button(name, path.read_bytes(), name, "application/json", use_container_width=True)
    pred = RESULTS_DIR / "predictions" / "best_model_test_predictions.csv"
    if pred.exists():
        st.download_button(
            pred.name, pred.read_bytes(), pred.name, "text/csv", use_container_width=True
        )
    st.download_button(
        "dataset_audit_live.json",
        json.dumps(audit, indent=2).encode("utf-8"),
        "dataset_audit_live.json",
        "application/json",
        use_container_width=True,
    )


def main() -> None:
    with st.sidebar:
        st.header("Navigasi")
        page = st.radio("Halaman", PAGES)
        n_demo = st.number_input("Jumlah URL DEMO", min_value=500, max_value=30000, value=6000, step=500)
        st.caption("Hasil hanya dihasilkan oleh `scripts/run_experiments.py`.")
    try:
        df, meta, audit, invalid = get_data(int(n_demo))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal memuat data: {exc}")
        st.stop()
    results = get_results()

    if page == "Ringkasan":
        page_overview(df, meta, audit, invalid)
    elif page == "Komposisi sumber":
        page_sources(df, audit)
    elif page == "Inspektur fitur URL":
        page_inspector()
    elif page == "Hasil eksperimen":
        if results is None:
            st.info("Belum ada `results/metrics.csv`. Jalankan eksperimen terlebih dahulu.")
        else:
            page_results(results)
    elif page == "Generalisasi & keterbatasan":
        page_generalisation(results, audit)
    else:
        page_downloads(results, audit)


if __name__ == "__main__":
    main()
