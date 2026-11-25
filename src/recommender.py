
"""
SmartLibrary AI - Recommendation Engine

Pipeline:
1. Bilingual preprocessing
2. Intent-aware query expansion
3. Weighted TF-IDF + Cosine Similarity
4. Metadata-based relevance scoring
5. Query-term coverage
6. Exact phrase matching
7. Subtopic-aware ranking
8. Catalog coverage checking

Model TF-IDF tidak dilatih ulang.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.bilingual import prepare_bilingual_query
from src.query_understanding import understand_query


# ============================================================
# 1. HELPERS
# ============================================================

def normalize_isbn(value):
    if value is None or pd.isna(value):
        return ""

    return re.sub(
        r"[^0-9X]",
        "",
        str(value).upper()
    )


def safe_text(value, fallback=""):
    if value is None or pd.isna(value):
        return fallback

    text = str(value).strip()

    if text.lower() in {"", "nan", "none"}:
        return fallback

    return text


# ============================================================
# 2. BOOK RECOMMENDER
# ============================================================

class BookRecommender:

    REQUIRED_KEYS = {
        "vectorizer",
        "tfidf_matrix",
        "book_isbns",
    }

    REQUIRED_COLUMNS = {
        "isbn13",
        "title",
        "authors",
        "description",
        "thumbnail",
    }

    # ========================================================
    # A. QUERY EXPANSION
    # ========================================================

    ORIGINAL_WEIGHT = 0.50
    EXPANSION_WEIGHT = 0.50

    # ========================================================
    # B. METADATA WEIGHTS
    # ========================================================

    TEXT_WEIGHT = 0.70
    TITLE_WEIGHT = 0.12
    CATEGORY_WEIGHT = 0.10
    DESCRIPTION_WEIGHT = 0.08

    CATEGORY_BONUS = 0.08
    FICTION_PENALTY = 0.05

    # ========================================================
    # C. QUERY RELEVANCE SIGNALS
    # ========================================================

    # Tambahan skor ketika istilah asli ditemukan
    TERM_COVERAGE_BONUS = 0.05

    # Tambahan ketika seluruh istilah penting ditemukan
    ALL_TERMS_BONUS = 0.045

    # Tambahan ketika frasa asli cocok
    EXACT_PHRASE_BONUS = 0.07

    # Penalti jika kata penentu subtopik tidak ditemukan
    MISSING_MODIFIER_PENALTY = 0.045

    # ========================================================
    # D. CATEGORY PATTERNS
    # ========================================================

    CATEGORY_PATTERNS = {

        "personal_growth": (
            r"self[\s-]?help|"
            r"personal development|"
            r"personal growth|"
            r"self[\s-]?improvement|"
            r"self[\s-]?esteem|"
            r"motivation"
        ),

        "motivation": (
            r"self[\s-]?help|"
            r"motivation|"
            r"personal growth"
        ),

        "psychology": (
            r"psycholog|"
            r"psychiatr|"
            r"psychoanalys|"
            r"mental health|"
            r"cognitive science"
        ),

        "law": (
            r"\blaw\b|"
            r"\blegal\b|"
            r"jurisprudence|"
            r"criminology|"
            r"criminal justice"
        ),

        "mystery": (
            r"mystery|"
            r"detective|"
            r"thriller|"
            r"suspense|"
            r"crime|"
            r"police procedural"
        ),

        "history": (
            r"history|"
            r"historical|"
            r"civilization"
        ),

        "philosophy": (
            r"philosoph|"
            r"ethics"
        ),
    }

    # ========================================================
    # E. TECHNICAL TOPICS
    # ========================================================

    TECHNICAL_TOPICS = {

        "machine learning": (
            r"\bmachine\s+learning\b"
        ),

        "deep learning": (
            r"\bdeep\s+learning\b"
        ),

        "data science": (
            r"\bdata\s+science\b"
        ),

        "artificial intelligence": (
            r"\bartificial\s+intelligence\b"
        ),

        "neural networks": (
            r"\bneural\s+networks?\b"
        ),
    }

    TECHNICAL_CATEGORY_PATTERN = (
        r"computer|"
        r"technology|"
        r"engineering|"
        r"mathematics|"
        r"statistics|"
        r"programming|"
        r"software|"
        r"data science|"
        r"artificial intelligence"
    )

    # ========================================================
    # F. EQUIVALENT TERMS
    # ========================================================

    TERM_ALIASES = {

        "criminal": (
            r"\b(?:criminal|crime|crimes|"
            r"criminology|criminological)\b"
        ),

        "murder": (
            r"\b(?:murder|murders|murdered|"
            r"homicide|homicides)\b"
        ),

        "child": (
            r"\b(?:child|children|childhood)\b"
        ),

        "psychology": (
            r"\b(?:psychology|psychological|"
            r"psychologist|psychologists)\b"
        ),

        "history": (
            r"\b(?:history|historical)\b"
        ),

        "self": (
            r"\b(?:self|personal|oneself)\b"
        ),

        "development": (
            r"\b(?:development|developing|"
            r"self.improvement|personal growth)\b"
        ),

        "law": (
            r"\b(?:law|legal|jurisprudence|judicial)\b"
        ),
    }

    SEARCH_FILLERS = {
        "book",
        "books",
        "novel",
        "novels",
        "about",
        "reading",
        "read",
        "recommend",
        "recommendation",
        "recommendations",
        "find",
        "search",
    }

    SUBTOPIC_INTENTS = {
        "law",
        "psychology",
        "history",
        "mystery",
    }

    # ========================================================
    # 3. INITIALIZATION
    # ========================================================

    def __init__(self, model_path, metadata_path):

        model_path = Path(model_path)
        metadata_path = Path(metadata_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}"
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan: {metadata_path}"
            )

        # Muat hanya model joblib milik sendiri/tepercaya.
        artifact = joblib.load(model_path)

        if (
            not isinstance(artifact, dict)
            or not self.REQUIRED_KEYS.issubset(artifact)
        ):
            raise ValueError(
                "Struktur artefak TF-IDF tidak sesuai."
            )

        self.vectorizer = artifact["vectorizer"]

        self.tfidf_matrix = artifact["tfidf_matrix"]

        self.book_isbns = [
            normalize_isbn(x)
            for x in artifact["book_isbns"]
        ]

        self.max_recommendations = int(
            artifact.get("max_recommendations", 10)
        )

        # ----------------------------------------------------
        # VALIDASI MODEL
        # ----------------------------------------------------

        if not hasattr(self.vectorizer, "transform"):
            raise ValueError(
                "Vectorizer tidak memiliki transform()."
            )

        if (
            len(self.book_isbns)
            != self.tfidf_matrix.shape[0]
        ):
            raise ValueError(
                "ISBN model dan matriks TF-IDF tidak cocok."
            )

        if (
            self.tfidf_matrix.shape[1]
            != len(self.vectorizer.vocabulary_)
        ):
            raise ValueError(
                "Jumlah fitur TF-IDF berbeda."
            )

        if (
            not all(self.book_isbns)
            or len(set(self.book_isbns))
            != len(self.book_isbns)
        ):
            raise ValueError(
                "ISBN model kosong atau duplikat."
            )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        df = pd.read_csv(
            metadata_path,
            dtype=str,
            low_memory=False,
        )

        missing = (
            self.REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"Kolom metadata tidak lengkap: {sorted(missing)}"
            )

        for col in (
            "categories",
            "simple_categories",
        ):
            if col not in df.columns:
                df[col] = ""

        df["_isbn"] = df["isbn13"].map(
            normalize_isbn
        )

        if (
            df["_isbn"].eq("").any()
            or df["_isbn"].duplicated().any()
        ):
            raise ValueError(
                "ISBN metadata kosong atau duplikat."
            )

        self.books_by_isbn = df.set_index(
            "_isbn",
            drop=False,
        )

        if (
            set(self.book_isbns)
            - set(self.books_by_isbn.index)
        ):
            raise ValueError(
                "Sebagian ISBN model tidak ditemukan."
            )

        # Urutan metadata sesuai dengan model
        self.books = (
            self.books_by_isbn
            .loc[self.book_isbns]
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # FIELDS
        # ----------------------------------------------------

        self.titles = (
            self.books["title"]
            .fillna("")
            .astype(str)
        )

        self.descriptions = (
            self.books["description"]
            .fillna("")
            .astype(str)
        )

        self.categories = (
            self.books["categories"]
            .fillna("")
            .astype(str)
            + " "
            + self.books["simple_categories"]
            .fillna("")
            .astype(str)
        )

        self.metadata_text = (
            self.titles
            + " "
            + self.categories
            + " "
            + self.descriptions
        ).str.lower()

        self.is_fiction = (
            self.books["simple_categories"]
            .fillna("")
            .str.fullmatch(
                r"\s*fiction\s*",
                case=False,
                na=False,
            )
            .to_numpy()
        )

        # ----------------------------------------------------
        # PRECOMPUTE METADATA VECTORS
        # ----------------------------------------------------

        # Menggunakan vectorizer yang sudah dilatih.
        # Tidak melakukan fit() ulang.

        self.title_matrix = (
            self.vectorizer.transform(
                self.titles.tolist()
            )
        )

        self.category_matrix = (
            self.vectorizer.transform(
                self.categories.tolist()
            )
        )

        self.description_matrix = (
            self.vectorizer.transform(
                self.descriptions.tolist()
            )
        )

    # ========================================================
    # 4. QUERY VECTOR
    # ========================================================

    def build_query_vector(
        self,
        normalized_query,
        enable_expansion=True,
    ):

        intent_info = understand_query(
            normalized_query,
            enable_expansion=enable_expansion,
        )

        original = self.vectorizer.transform(
            [normalized_query]
        )

        combined = original

        expanded = intent_info.get(
            "expanded_query"
        )

        applied = False

        reason = intent_info.get(
            "expansion_reason",
            "unknown",
        )

        actual_original = 1.0
        actual_expansion = 0.0

        if (
            enable_expansion
            and expanded
            and intent_info.get("expansion_applied")
        ):

            extra = self.vectorizer.transform(
                [expanded]
            )

            if (
                extra.nnz > 0
                and self.EXPANSION_WEIGHT > 0
            ):

                if original.nnz:

                    combined = normalize(
                        self.ORIGINAL_WEIGHT * original
                        + self.EXPANSION_WEIGHT * extra,
                        norm="l2",
                    )

                    actual_original = self.ORIGINAL_WEIGHT
                    actual_expansion = self.EXPANSION_WEIGHT

                else:

                    combined = extra

                    actual_original = 0.0
                    actual_expansion = 1.0

                applied = True

            else:
                reason = "expansion_out_of_vocabulary"

        return original, combined, {

            "intent": intent_info["intent"],

            "intent_label": intent_info["intent_label"],

            "expanded_query": expanded,

            "expansion_applied": applied,

            "expansion_reason": reason,

            "original_weight": actual_original,

            "expansion_weight": actual_expansion,
        }

    # ========================================================
    # 5. TECHNICAL COVERAGE
    # ========================================================

    def get_technical_coverage_mask(
        self,
        normalized_query,
    ):

        query = " ".join(
            str(normalized_query)
            .casefold()
            .split()
        )

        pattern = next(
            (
                p
                for _, p in sorted(
                    self.TECHNICAL_TOPICS.items(),
                    key=lambda item: -len(item[0]),
                )
                if re.search(
                    p,
                    query,
                    flags=re.IGNORECASE,
                )
            ),
            None,
        )

        if pattern is None:
            return None

        def matches(series, regex):

            return series.str.contains(
                regex,
                case=False,
                regex=True,
                na=False,
            ).to_numpy()

        title = matches(
            self.titles,
            pattern,
        )

        category = matches(
            self.categories,
            pattern,
        )

        description = matches(
            self.descriptions,
            pattern,
        )

        technical = matches(
            self.categories,
            self.TECHNICAL_CATEGORY_PATTERN,
        )

        return (
            ((title | category) & ~self.is_fiction)
            |
            (description & technical)
        )

    # ========================================================
    # 6. QUERY TERM MATCHING
    # ========================================================

    def _original_term_signals(
        self,
        query,
        intent,
    ):
        """
        Menghitung:
        - cakupan istilah asli pengguna
        - kecocokan seluruh istilah
        - kecocokan frasa
        - kelengkapan kata penentu subtopik
        """

        terms = list(
            dict.fromkeys(
                t
                for t in re.findall(
                    r"[a-z0-9]+",
                    query.casefold(),
                )
                if (
                    len(t) > 1
                    and t not in ENGLISH_STOP_WORDS
                    and t not in self.SEARCH_FILLERS
                )
            )
        )

        n = len(self.books)

        if not terms:

            return (
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                np.zeros(n, dtype=bool),
            )

        term_matches = []
        term_weights = []

        for term in terms:

            pattern = self.TERM_ALIASES.get(
                term,
                rf"\b{re.escape(term)}\b",
            )

            matched = (
                self.metadata_text
                .str.contains(
                    pattern,
                    case=False,
                    regex=True,
                    na=False,
                )
                .to_numpy()
            )

            term_matches.append(
                matched
            )

            # IDF memberikan bobot lebih tinggi
            # pada istilah yang lebih spesifik.
            idx = self.vectorizer.vocabulary_.get(
                term
            )

            if idx is not None:
                weight = float(
                    self.vectorizer.idf_[idx]
                )
            else:
                weight = 1.0

            term_weights.append(
                min(weight, 10.0)
            )

        weights = np.asarray(
            term_weights,
            dtype=float,
        )

        hit_matrix = (
            np.stack(term_matches)
            .astype(float)
        )

        # ----------------------------------------------------
        # TERM COVERAGE
        # ----------------------------------------------------

        weighted_coverage = np.average(
            hit_matrix,
            axis=0,
            weights=weights,
        )

        # ----------------------------------------------------
        # ALL TERMS MATCH
        # ----------------------------------------------------

        if len(terms) > 1:

            all_terms = (
                np.all(
                    hit_matrix.astype(bool),
                    axis=0,
                )
                .astype(float)
            )

        else:
            all_terms = np.zeros(n)

        # ----------------------------------------------------
        # EXACT PHRASE MATCH
        # ----------------------------------------------------

        phrase = (
            r"\b"
            + r"\s+".join(
                re.escape(t)
                for t in terms
            )
            + r"\b"
        )

        if len(terms) >= 2:

            phrase_title = (
                self.titles
                .str.contains(
                    phrase,
                    case=False,
                    regex=True,
                    na=False,
                )
                .to_numpy()
            )

            phrase_cat = (
                self.categories
                .str.contains(
                    phrase,
                    case=False,
                    regex=True,
                    na=False,
                )
                .to_numpy()
            )

            phrase_desc = (
                self.descriptions
                .str.contains(
                    phrase,
                    case=False,
                    regex=True,
                    na=False,
                )
                .to_numpy()
            )

            phrase_strength = np.maximum.reduce([
                phrase_title.astype(float),
                phrase_cat.astype(float),
                0.5 * phrase_desc.astype(float),
            ])

        else:
            phrase_strength = np.zeros(n)

        # ----------------------------------------------------
        # SUBTOPIC MODIFIER
        # ----------------------------------------------------

        # Contoh:
        # criminal law -> modifier: criminal
        # child psychology -> modifier: child
        # ancient history -> modifier: ancient

        modifier_missing = np.zeros(
            n,
            dtype=bool,
        )

        if (
            len(terms) == 2
            and intent in self.SUBTOPIC_INTENTS
        ):

            modifier_missing = (
                ~term_matches[0]
            )

        return (
            weighted_coverage,
            all_terms,
            phrase_strength,
            modifier_missing,
        )

    # ========================================================
    # 7. RECOMMENDATION
    # ========================================================

    def recommend(
        self,
        query,
        language="auto",
        top_k=10,
        enable_expansion=True,
        enable_rerank=True,
    ):

        details = prepare_bilingual_query(
            query,
            language=language,
        )

        normalized = details.get(
            "normalized_query",
            "",
        )

        details.update({

            "ranking_method": (
                "tfidf_cosine_similarity"
            ),

            "reranking_applied": False,

            "catalog_coverage": "not_checked",

            "candidate_count": 0,

            "recommendation_count": 0,
        })

        def empty():
            return {
                "query_details": details,
                "books": [],
            }

        if not normalized:
            return empty()

        # ----------------------------------------------------
        # QUERY VECTOR
        # ----------------------------------------------------

        (
            original_vector,
            mixed_vector,
            intent_details,
        ) = self.build_query_vector(
            normalized,
            enable_expansion=enable_expansion,
        )

        details.update(
            intent_details
        )

        if mixed_vector.nnz == 0:

            details["catalog_coverage"] = (
                "query_out_of_vocabulary"
            )

            return empty()

        # ----------------------------------------------------
        # TF-IDF COSINE SIMILARITY
        # ----------------------------------------------------

        base_scores = cosine_similarity(
            mixed_vector,
            self.tfidf_matrix,
        ).ravel()

        base_scores = np.nan_to_num(
            base_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ----------------------------------------------------
        # CATALOG COVERAGE
        # ----------------------------------------------------

        technical_mask = (
            self.get_technical_coverage_mask(
                normalized
            )
        )

        if technical_mask is not None:

            if not np.any(technical_mask):

                details["catalog_coverage"] = (
                    "insufficient_evidence"
                )

                return empty()

            details["catalog_coverage"] = "supported"

        # ----------------------------------------------------
        # METADATA RE-RANKING
        # ----------------------------------------------------

        if enable_rerank:

            # Seluruh koleksi diperiksa.
            # Tidak hanya kandidat Top 200.

            title_scores = cosine_similarity(
                mixed_vector,
                self.title_matrix,
            ).ravel()

            category_scores = cosine_similarity(
                mixed_vector,
                self.category_matrix,
            ).ravel()

            desc_scores = cosine_similarity(
                mixed_vector,
                self.description_matrix,
            ).ravel()

            final = (
                self.TEXT_WEIGHT * base_scores
                + self.TITLE_WEIGHT * title_scores
                + self.CATEGORY_WEIGHT * category_scores
                + self.DESCRIPTION_WEIGHT * desc_scores
            )

            intent = details.get(
                "intent",
                "general",
            )

            (
                coverage,
                all_terms,
                phrase,
                modifier_missing,
            ) = self._original_term_signals(
                normalized,
                intent,
            )

            # ------------------------------------------------
            # ORIGINAL QUERY ANCHORING
            # ------------------------------------------------

            final += (
                self.TERM_COVERAGE_BONUS
                * coverage
            )

            final += (
                self.ALL_TERMS_BONUS
                * all_terms
            )

            final += (
                self.EXACT_PHRASE_BONUS
                * phrase
            )

            # ------------------------------------------------
            # CATEGORY-AWARE BONUS
            # ------------------------------------------------

            pattern = (
                self.CATEGORY_PATTERNS.get(
                    intent
                )
            )

            if pattern:

                category_match = (
                    self.categories
                    .str.contains(
                        pattern,
                        case=False,
                        regex=True,
                        na=False,
                    )
                    .to_numpy()
                )

                # Bonus kategori disesuaikan
                # dengan kecocokan query asli.

                final += (
                    self.CATEGORY_BONUS
                    * category_match.astype(float)
                    * (0.40 + 0.60 * coverage)
                )

            # ------------------------------------------------
            # FICTION PENALTY
            # ------------------------------------------------

            if intent in {"psychology", "law"}:

                final -= (
                    self.FICTION_PENALTY
                    * self.is_fiction.astype(float)
                )

            # ------------------------------------------------
            # SUBTOPIC PENALTY
            # ------------------------------------------------

            final -= (
                self.MISSING_MODIFIER_PENALTY
                * modifier_missing.astype(float)
            )

            details["ranking_method"] = (
                "intent_anchored_metadata_reranking"
            )

            details["reranking_applied"] = True

        else:
            final = base_scores.copy()

        # ----------------------------------------------------
        # SCORE VALIDATION
        # ----------------------------------------------------

        final = np.nan_to_num(
            final,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        final = np.clip(
            final,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # RELEVANCE EVIDENCE
        # ----------------------------------------------------

        evidence = base_scores > 0

        if enable_rerank:

            evidence |= (
                (coverage > 0)
                | (title_scores > 0)
                | (category_scores > 0)
                | (desc_scores > 0)
            )

        if technical_mask is not None:

            evidence &= technical_mask

        final[~evidence] = 0.0

        details["candidate_count"] = int(
            np.count_nonzero(evidence)
        )

        # ----------------------------------------------------
        # TOP K
        # ----------------------------------------------------

        limit = min(
            max(int(top_k), 0),
            self.max_recommendations,
        )

        if limit <= 0:
            return empty()

        ordered = np.argsort(
            -final,
            kind="stable",
        )

        results = []

        for row_id in ordered:

            score = float(
                final[row_id]
            )

            if (
                score <= 0
                or len(results) >= limit
            ):
                break

            isbn = self.book_isbns[
                int(row_id)
            ]

            book = self.books_by_isbn.loc[
                isbn
            ]

            results.append({

                "rank": len(results) + 1,

                "isbn13": isbn,

                "title": safe_text(
                    book["title"],
                    "Judul tidak tersedia",
                ),

                "authors": safe_text(
                    book["authors"],
                    "Penulis tidak tersedia",
                ),

                "description": safe_text(
                    book["description"],
                    "Deskripsi tidak tersedia",
                ),

                "thumbnail": safe_text(
                    book["thumbnail"],
                ),

                "similarity_score": score,

                "base_similarity_score": float(
                    base_scores[row_id]
                ),
            })

        details["recommendation_count"] = (
            len(results)
        )

        return {
            "query_details": details,
            "books": results,
        }
