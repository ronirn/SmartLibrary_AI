
"""
SmartLibrary AI
Evaluasi Precision@10 untuk bobot Query Expansion.

Prasyarat:
- query_expansion_weight_comparison.csv
- query_expansion_relevance_labels.csv
- Semua label relevance sudah diisi 0 atau 1.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "reports" / "evaluation"

RANKING_FILE = (
    EVAL_DIR / "query_expansion_weight_comparison.csv"
)

LABEL_FILE = (
    EVAL_DIR / "query_expansion_relevance_labels.csv"
)

DETAIL_FILE = (
    EVAL_DIR / "query_expansion_precision_at_10.csv"
)

SUMMARY_FILE = (
    EVAL_DIR / "query_expansion_summary.csv"
)


def main():

    rankings = pd.read_csv(
        RANKING_FILE,
        dtype={"query": str, "isbn13": str},
    )

    labels = pd.read_csv(
        LABEL_FILE,
        dtype={
            "query": str,
            "isbn13": str,
            "relevance": str,
        },
    )

    # Periksa kolom penilaian
    labels["relevance"] = (
        labels["relevance"]
        .fillna("")
        .str.strip()
    )

    invalid = labels[
        ~labels["relevance"].isin(["0", "1"])
    ]

    if not invalid.empty:
        print(
            f"Masih terdapat {len(invalid)} label "
            "yang kosong atau tidak valid."
        )
        print(
            "Isi semua kolom relevance "
            "dengan 0 atau 1."
        )
        return

    # Pastikan satu query-ISBN hanya dinilai sekali
    if labels.duplicated(
        subset=["query", "isbn13"]
    ).any():
        raise ValueError(
            "Terdapat label query-ISBN duplikat."
        )

    labels["relevance"] = (
        labels["relevance"].astype(int)
    )

    # Gabungkan ranking dan label relevansi
    merged = rankings.merge(
        labels[["query", "isbn13", "relevance"]],
        on=["query", "isbn13"],
        how="left",
        validate="many_to_one",
    )

    if merged["relevance"].isna().any():
        raise ValueError(
            "Ada buku hasil ranking yang belum "
            "memiliki label relevansi."
        )

    # Pastikan setiap konfigurasi menghasilkan 10 buku
    counts = merged.groupby(
        ["query", "configuration"]
    ).size()

    if not counts.eq(10).all():
        raise ValueError(
            "Ada konfigurasi dengan hasil bukan 10 buku."
        )

    # Precision@10 per query dan konfigurasi
    detail = (
        merged.groupby(
            ["query", "configuration"],
            as_index=False,
        )
        .agg(
            relevant_top10=("relevance", "sum"),
            precision_at_10=("relevance", "mean"),
        )
    )

    # Rata-rata antar-query
    summary = (
        detail.groupby(
            "configuration",
            as_index=False,
        )
        .agg(
            mean_precision_at_10=(
                "precision_at_10", "mean"
            ),
            queries_evaluated=("query", "nunique"),
        )
    )

    order = [
        "Baseline",
        "80_20",
        "70_30",
        "60_40",
        "50_50",
    ]

    summary["configuration"] = pd.Categorical(
        summary["configuration"],
        categories=order,
        ordered=True,
    )

    summary = summary.sort_values(
        "configuration"
    ).reset_index(drop=True)

    baseline_score = summary.loc[
        summary["configuration"] == "Baseline",
        "mean_precision_at_10",
    ].iloc[0]

    summary["delta_vs_baseline_pp"] = (
        summary["mean_precision_at_10"]
        - baseline_score
    ) * 100

    detail.to_csv(
        DETAIL_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 65)
    print("HASIL EVALUASI QUERY EXPANSION")
    print("=" * 65)

    print("\nPrecision@10 per konfigurasi:\n")

    for _, row in summary.iterrows():
        print(
            f"{str(row['configuration']):10} "
            f"P@10={row['mean_precision_at_10']:.3f} "
            f"Delta={row['delta_vs_baseline_pp']:+.1f} pp"
        )

    print("\nHasil per query:\n")

    pivot = detail.pivot(
        index="query",
        columns="configuration",
        values="precision_at_10",
    )

    print(pivot.to_string())

    print("\nFile ringkasan:", SUMMARY_FILE)
    print("File detail:", DETAIL_FILE)


if __name__ == "__main__":
    main()
