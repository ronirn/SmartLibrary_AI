
"""
SMARTLIBRARY AI
Evaluasi Bobot Query Expansion

Membandingkan:
1. Baseline
2. Original 80% + Expansion 20%
3. Original 70% + Expansion 30%
4. Original 60% + Expansion 40%
5. Original 50% + Expansion 50%

Tidak melatih ulang model.
Tidak mengubah app.py.
"""

import sys
import csv
from pathlib import Path


# ============================================================
# 1. KONFIGURASI PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.recommender import BookRecommender


MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "smartlibrary_tfidf.joblib"
)

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "processed"
    / "books_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "reports"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. QUERY PENGUJIAN
# ============================================================

TEST_QUERIES = [
    "self development",
    "psychology",
    "murder mystery",
    "world history",
    "criminal law",
    "machine learning",
]


# ============================================================
# 3. KONFIGURASI BOBOT
# ============================================================

WEIGHT_CONFIGS = [
    ("Baseline", 1.00, 0.00),
    ("80_20", 0.80, 0.20),
    ("70_30", 0.70, 0.30),
    ("60_40", 0.60, 0.40),
    ("50_50", 0.50, 0.50),
]


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\nMemuat model SmartLibrary AI...")

recommender = BookRecommender(
    model_path=MODEL_PATH,
    metadata_path=DATASET_PATH,
)

print("Model berhasil dimuat.")
print(
    "Jumlah buku:",
    len(recommender.book_isbns)
)


# ============================================================
# 5. PROSES PENGUJIAN
# ============================================================

all_results = []

# Pool unik untuk anotasi relevansi
annotation_pool = {}

for query in TEST_QUERIES:

    print("\n" + "=" * 75)
    print("QUERY:", query)
    print("=" * 75)

    for config_name, original_w, expansion_w in WEIGHT_CONFIGS:

        # Mengubah bobot sementara di instance ini saja
        recommender.ORIGINAL_WEIGHT = original_w
        recommender.EXPANSION_WEIGHT = expansion_w

        # Baseline tidak menggunakan expansion
        enable_expansion = (
            config_name != "Baseline"
        )

        output = recommender.recommend(
            query=query,
            language="en",
            top_k=10,
            enable_expansion=enable_expansion,
        )

        books = output["books"]
        details = output["query_details"]

        print(
            f"\n[{config_name}] "
            f"Original={original_w:.0%} "
            f"Expansion={expansion_w:.0%}"
        )

        print(
            "Expansion applied:",
            details.get(
                "expansion_applied",
                False
            )
        )

        for book in books:

            record = {
                "query": query,
                "configuration": config_name,
                "original_weight": original_w,
                "expansion_weight": expansion_w,
                "rank": book["rank"],
                "isbn13": book["isbn13"],
                "title": book["title"],
                "authors": book["authors"],
                "similarity_score": book[
                    "similarity_score"
                ],
                "intent": details.get(
                    "intent",
                    "general"
                ),
                "expansion_applied": details.get(
                    "expansion_applied",
                    False
                ),
            }

            all_results.append(record)

            # Satu buku untuk satu query hanya perlu
            # dinilai relevansinya satu kali.
            key = (
                query,
                book["isbn13"],
            )

            if key not in annotation_pool:

                annotation_pool[key] = {
                    "query": query,
                    "isbn13": book["isbn13"],
                    "title": book["title"],
                    "authors": book["authors"],
                    "description": book["description"],
                    "relevance": "",
                }

            # Output terminal
            print(
                f'{book["rank"]:2}. '
                f'{book["title"][:45]:45} '
                f'{book["similarity_score"] * 100:6.2f}%'
            )


# ============================================================
# 6. SIMPAN HASIL PERBANDINGAN
# ============================================================

result_path = (
    OUTPUT_DIR
    / "query_expansion_weight_comparison.csv"
)

if all_results:

    with open(
        result_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(all_results)


# ============================================================
# 7. BUAT TEMPLATE PENILAIAN RELEVANSI
# ============================================================

annotation_path = (
    OUTPUT_DIR
    / "query_expansion_relevance_labels.csv"
)

# Jangan menimpa penilaian manual yang sudah ada
if not annotation_path.exists():

    annotations = list(
        annotation_pool.values()
    )

    if annotations:

        with open(
            annotation_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=annotations[0].keys(),
            )

            writer.writeheader()
            writer.writerows(annotations)


# ============================================================
# 8. RINGKASAN
# ============================================================

print("\n" + "=" * 75)
print("PENGUJIAN SELESAI")
print("=" * 75)

print("Jumlah query:", len(TEST_QUERIES))
print("Jumlah konfigurasi:", len(WEIGHT_CONFIGS))
print("Jumlah record ranking:", len(all_results))
print(
    "Pasangan query-buku unik:",
    len(annotation_pool)
)

print("\nFile perbandingan:")
print(result_path)

print("\nTemplate penilaian:")
print(annotation_path)

print("\nTidak ada model yang dilatih ulang.")
