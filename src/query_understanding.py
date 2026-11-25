
"""
============================================================
SMARTLIBRARY AI - QUERY UNDERSTANDING
File: src/query_understanding.py

Fungsi:
1. Normalisasi teks untuk identifikasi intent.
2. Mengenali maksud pencarian berdasarkan aturan.
3. Memperluas query pendek dengan istilah terkait.
4. Menghindari query expansion yang terlalu luas.
5. Menyiapkan informasi untuk weighted TF-IDF.

Catatan:
- Tidak mengganti model TF-IDF.
- Tidak melakukan training ulang.
- Dipanggil setelah preprocessing bilingual.
============================================================
"""

import re
import unicodedata


# ============================================================
# 1. KONFIGURASI
# ============================================================

# Ekspansi hanya diterapkan pada query pendek.
# Tujuannya mengurangi risiko query drift.
MAX_QUERY_WORDS = 4


# ============================================================
# 2. KAMUS SEARCH INTENT
# ============================================================

INTENT_RULES = {

    # --------------------------------------------------------
    # A. PERSONAL GROWTH / SELF DEVELOPMENT
    # --------------------------------------------------------

    "personal_growth": {

        "label": "Personal Growth & Self-Help",

        "patterns": [
            r"\bself development\b",
            r"\bself improvement\b",
            r"\bself help\b",
            r"\bpersonal development\b",
            r"\bpersonal growth\b",
            r"\bself awareness\b",
            r"\bself confidence\b",
        ],

        "expansion": (
            "self help personal growth "
            "self improvement motivation "
            "positive habits confidence "
            "self awareness personal transformation"
        ),
    },

    # --------------------------------------------------------
    # B. PSYCHOLOGY
    # --------------------------------------------------------

    "psychology": {

        "label": "Psychology & Human Behavior",

        "patterns": [
            r"\bpsychology\b",
            r"\bhuman behavior\b",
            r"\bhuman behaviour\b",
            r"\bcognitive psychology\b",
            r"\bsocial psychology\b",
        ],

        "expansion": (
            "psychology human behavior "
            "human mind cognition emotions "
            "mental processes psychological research"
        ),
    },

    # --------------------------------------------------------
    # C. MYSTERY & DETECTIVE
    # --------------------------------------------------------

    "mystery": {

        "label": "Mystery & Detective Fiction",

        "patterns": [
            r"\bmystery\b",
            r"\bdetective\b",
            r"\bmurder mystery\b",
            r"\bcrime fiction\b",
            r"\bdetective novel\b",
        ],

        "expansion": (
            "mystery detective novel "
            "crime investigation suspense "
            "murder mystery thriller fiction"
        ),
    },

    # --------------------------------------------------------
    # D. HISTORY
    # --------------------------------------------------------

    "history": {

        "label": "History & Civilization",

        "patterns": [
            r"\bworld history\b",
            r"\bhistory books?\b",
            r"\bancient history\b",
            r"\bhistorical civilization\b",
            r"\bhistorical events\b",
        ],

        "expansion": (
            "world history historical events "
            "ancient civilizations world civilization "
            "historical societies history"
        ),
    },

    # --------------------------------------------------------
    # E. LAW
    # --------------------------------------------------------

    "law": {

        "label": "Law & Legal Studies",

        "patterns": [
            r"\blaw\b",
            r"\blegal studies\b",
            r"\bcriminal law\b",
            r"\blegal system\b",
            r"\bconstitutional law\b",
        ],

        "expansion": (
            "law legal system legal studies "
            "legislation justice courts "
            "legal principles"
        ),
    },

    # --------------------------------------------------------
    # F. PHILOSOPHY
    # --------------------------------------------------------

    "philosophy": {

        "label": "Philosophy & Ethics",

        "patterns": [
            r"\bphilosophy\b",
            r"\bphilosophical\b",
            r"\bethics\b",
            r"\bmoral philosophy\b",
        ],

        "expansion": (
            "philosophy ethics moral philosophy "
            "ethical theories philosophical thought "
            "morality human existence"
        ),
    },

    # --------------------------------------------------------
    # G. MOTIVATION
    # --------------------------------------------------------

    "motivation": {

        "label": "Motivation & Success",

        "patterns": [
            r"\bmotivation\b",
            r"\bmotivational books?\b",
            r"\bsuccess mindset\b",
            r"\bpositive thinking\b",
        ],

        "expansion": (
            "motivation personal growth "
            "success mindset inspiration "
            "positive thinking achievement "
            "self improvement"
        ),
    },

}


# ============================================================
# 3. NORMALISASI TEKS UNTUK INTENT
# ============================================================

def normalize_intent_text(text: str) -> str:
    """
    Menormalisasi teks khusus untuk pencocokan intent.

    Tidak mengubah query asli yang digunakan oleh model.
    """

    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    # Normalisasi Unicode
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.casefold()

    # Samakan tanda hubung dan underscore
    text = re.sub(r"[-_]+", " ", text)

    # Hilangkan tanda baca
    text = re.sub(r"[^\w\s]", " ", text)

    # Rapikan whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 4. DETEKSI SEARCH INTENT
# ============================================================

def detect_search_intent(query: str) -> dict:
    """
    Mendeteksi intent berdasarkan frasa dalam query.

    Jika beberapa intent berbeda terdeteksi,
    status dibuat ambiguous agar ekspansi
    tidak menggeser maksud pengguna.
    """

    normalized_query = normalize_intent_text(query)

    if not normalized_query:
        return {
            "intent": "general",
            "intent_label": "General Search",
            "matched_patterns": [],
            "ambiguous": False,
        }

    matches = []

    # Periksa seluruh aturan intent
    for intent_name, config in INTENT_RULES.items():

        matched_patterns = []

        for pattern in config["patterns"]:

            if re.search(pattern, normalized_query):
                matched_patterns.append(pattern)

        if matched_patterns:

            matches.append({
                "intent": intent_name,
                "label": config["label"],
                "patterns": matched_patterns,
            })

    # Tidak ada intent yang cocok
    if not matches:

        return {
            "intent": "general",
            "intent_label": "General Search",
            "matched_patterns": [],
            "ambiguous": False,
        }

    # Lebih dari satu kategori intent
    if len(matches) > 1:

        return {
            "intent": "ambiguous",
            "intent_label": "Multiple Search Intents",
            "matched_patterns": [
                pattern
                for match in matches
                for pattern in match["patterns"]
            ],
            "ambiguous": True,
        }

    # Satu intent ditemukan
    match = matches[0]

    return {
        "intent": match["intent"],
        "intent_label": match["label"],
        "matched_patterns": match["patterns"],
        "ambiguous": False,
    }


# ============================================================
# 5. QUERY UNDERSTANDING
# ============================================================

def understand_query(
    query: str,
    enable_expansion: bool = True,
    max_words: int = MAX_QUERY_WORDS,
) -> dict:
    """
    Memahami maksud query dan menyiapkan ekspansi.

    Parameters
    ----------
    query : str
        Query bahasa Inggris yang sudah diproses
        oleh bilingual.py.

    enable_expansion : bool
        Mengaktifkan atau menonaktifkan ekspansi.

    max_words : int
        Batas panjang query yang diperbolehkan
        menggunakan ekspansi.

    Returns
    -------
    dict
        original_query
        normalized_query
        intent
        intent_label
        expanded_query
        expansion_applied
        expansion_reason
        matched_patterns
    """

    original_query = (
        "" if query is None else str(query).strip()
    )

    normalized_query = normalize_intent_text(
        original_query
    )

    # --------------------------------------------------------
    # A. Query kosong
    # --------------------------------------------------------

    if not normalized_query:

        return {
            "original_query": original_query,
            "normalized_query": "",
            "intent": "general",
            "intent_label": "General Search",
            "expanded_query": None,
            "expansion_applied": False,
            "expansion_reason": "empty_query",
            "matched_patterns": [],
        }

    # --------------------------------------------------------
    # B. Deteksi intent
    # --------------------------------------------------------

    intent_info = detect_search_intent(
        normalized_query
    )

    intent = intent_info["intent"]

    word_count = len(
        normalized_query.split()
    )

    # --------------------------------------------------------
    # C. Ekspansi tidak diaktifkan
    # --------------------------------------------------------

    if not enable_expansion:

        return {
            "original_query": original_query,
            "normalized_query": normalized_query,
            "intent": intent,
            "intent_label": intent_info["intent_label"],
            "expanded_query": None,
            "expansion_applied": False,
            "expansion_reason": "disabled",
            "matched_patterns": intent_info["matched_patterns"],
        }

    # --------------------------------------------------------
    # D. Query terlalu panjang
    # --------------------------------------------------------

    if word_count > max_words:

        return {
            "original_query": original_query,
            "normalized_query": normalized_query,
            "intent": intent,
            "intent_label": intent_info["intent_label"],
            "expanded_query": None,
            "expansion_applied": False,
            "expansion_reason": "query_too_long",
            "matched_patterns": intent_info["matched_patterns"],
        }

    # --------------------------------------------------------
    # E. Intent ambigu atau tidak diketahui
    # --------------------------------------------------------

    if intent not in INTENT_RULES:

        reason = (
            "ambiguous_intent"
            if intent == "ambiguous"
            else "no_matching_intent"
        )

        return {
            "original_query": original_query,
            "normalized_query": normalized_query,
            "intent": intent,
            "intent_label": intent_info["intent_label"],
            "expanded_query": None,
            "expansion_applied": False,
            "expansion_reason": reason,
            "matched_patterns": intent_info["matched_patterns"],
        }

    # --------------------------------------------------------
    # F. Ambil ekspansi berdasarkan intent
    # --------------------------------------------------------

    expanded_query = INTENT_RULES[
        intent
    ]["expansion"]

    expanded_query = normalize_intent_text(
        expanded_query
    )

    return {
        "original_query": original_query,
        "normalized_query": normalized_query,
        "intent": intent,
        "intent_label": intent_info["intent_label"],
        "expanded_query": expanded_query,
        "expansion_applied": True,
        "expansion_reason": "intent_matched",
        "matched_patterns": intent_info["matched_patterns"],
    }


# ============================================================
# 6. PENGUJIAN MODUL
# ============================================================

if __name__ == "__main__":

    test_queries = [
        "self development",
        "personal growth",
        "psychology",
        "murder mystery",
        "world history",
        "criminal law",
        "philosophy and ethics",
        "self help personal growth motivation",
        "machine learning",
    ]

    print("=" * 65)
    print("SMARTLIBRARY AI - QUERY UNDERSTANDING TEST")
    print("=" * 65)

    for query in test_queries:

        result = understand_query(query)

        print(f"\nInput     : {query}")
        print(f"Intent    : {result['intent']}")
        print(f"Label     : {result['intent_label']}")
        print(f"Expansion : {result['expanded_query']}")
        print(f"Applied   : {result['expansion_applied']}")
        print(f"Reason    : {result['expansion_reason']}")

        print("-" * 65)
