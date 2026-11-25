
# =====================================================
# SMARTLIBRARY AI - MODUL PENCARIAN BILINGUAL
# File: src/bilingual.py
# =====================================================

import re
import html
import unicodedata

from functools import lru_cache

from deep_translator import MyMemoryTranslator
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


# Agar hasil deteksi bahasa konsisten
DetectorFactory.seed = 42


# =====================================================
# 1. NORMALISASI DASAR QUERY
# =====================================================

def normalize_query(text):
    """
    Membersihkan teks pencarian agar sesuai
    dengan format teks pada dataset TF-IDF.
    """

    if text is None or not str(text).strip():
        return ""

    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)

    # Menghapus URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Menghapus HTML tags
    text = re.sub(r"<[^>]*>", " ", text)

    # Mengubah menjadi huruf kecil
    text = text.casefold()

    # Menghapus tanda baca
    text = re.sub(r"[^\w\s]", " ", text)

    # Mengganti underscore menjadi spasi
    text = re.sub(r"_+", " ", text)

    # Merapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# 2. NORMALISASI TOPIK PENCARIAN
# =====================================================

def normalize_search_intent(text):
    """
    Menyeragamkan variasi ejaan dan menghapus
    kalimat pembuka umum yang tidak menggambarkan
    topik buku yang dicari.
    """

    text = normalize_query(text)

    if not text:
        return ""

    # Menyamakan variasi ejaan British dan American
    spelling_variants = {
        "behaviour": "behavior",
        "behaviours": "behaviors",
        "colour": "color",
        "colours": "colors",
        "favourite": "favorite",
        "favourites": "favorites",
        "centre": "center",
        "centres": "centers",
        "theatre": "theater",
        "theatres": "theaters",
        "organise": "organize",
        "organised": "organized"
    }

    words = text.split()

    words = [
        spelling_variants.get(word, word)
        for word in words
    ]

    text = " ".join(words)

    # Menghapus kalimat pembuka umum
    text = re.sub(
        r"^(?:i want to read|"
        r"i want to find|"
        r"i would like to read|"
        r"i am looking for|"
        r"im looking for|"
        r"i am searching for|"
        r"i want|"
        r"i need|"
        r"please recommend|"
        r"recommend me)\s+",
        "",
        text
    )

    # Menghapus pola umum permintaan buku
    # Contoh:
    # "a book about psychology" -> "psychology"
    # "books about psychology" -> "psychology"
    text = re.sub(
        r"^(?:(?:a|an|some)\s+)?"
        r"(?:book|books)\s+"
        r"(?:about|on|related to|regarding)\s+",
        "",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# 3. DETEKSI BAHASA INPUT
# =====================================================

def detect_query_language(text):
    """
    Memperkirakan bahasa input pengguna.

    Hasil:
    id      = Bahasa Indonesia
    en      = Bahasa Inggris
    unknown = Tidak teridentifikasi
    """

    if text is None or not str(text).strip():
        return "unknown"

    try:
        return detect(str(text))

    except LangDetectException:
        return "unknown"


# =====================================================
# 4. PENERJEMAHAN INDONESIA KE INGGRIS
# =====================================================

@lru_cache(maxsize=256)
def translate_id_to_en(text):
    """
    Menerjemahkan input Indonesia ke Inggris
    menggunakan MyMemory Translator.

    Cache menghindari pengiriman permintaan
    berulang untuk teks yang sama.
    """

    if text is None or not str(text).strip():
        return ""

    translator = MyMemoryTranslator(
        source="id-ID",
        target="en-GB"
    )

    translated = translator.translate(
        str(text).strip()
    )

    if not translated or not str(translated).strip():
        raise RuntimeError(
            "Layanan penerjemahan mengembalikan "
            "hasil kosong."
        )

    return str(translated).strip()


# =====================================================
# 5. FUNGSI UTAMA PENCARIAN BILINGUAL
# =====================================================

def prepare_bilingual_query(query, language="auto"):
    """
    Menyiapkan input Indonesia/Inggris
    menjadi teks bahasa Inggris untuk TF-IDF.

    Parameters
    ----------
    query : str
        Permintaan buku dari pengguna.

    language : str
        auto = Deteksi otomatis
        id   = Bahasa Indonesia
        en   = Bahasa Inggris

    Returns
    -------
    dict
        Informasi query asli, bahasa,
        terjemahan, dan hasil normalisasi.
    """

    # Validasi pilihan bahasa
    if language not in {"auto", "id", "en"}:
        raise ValueError(
            "language harus 'auto', 'id', atau 'en'."
        )

    # Membersihkan spasi input
    original_query = re.sub(
        r"\s+", " ", str(query or "")
    ).strip()

    # Menangani input kosong
    if not original_query:
        return {
            "original_query": "",
            "detected_language": "unknown",
            "english_query": "",
            "normalized_query": "",
            "translated": False
        }

    # Menentukan bahasa input
    if language == "auto":
        detected_language = detect_query_language(
            original_query
        )
    else:
        detected_language = language

    # Validasi bahasa terdeteksi
    if detected_language not in {"id", "en"}:
        raise RuntimeError(
            "Bahasa input tidak dikenali sebagai "
            "Indonesia atau Inggris. "
            "Pilih bahasa secara manual."
        )

    # Jika bahasa Inggris, tidak diterjemahkan
    if detected_language == "en":
        english_query = original_query
        translated = False

    # Jika bahasa Indonesia, diterjemahkan
    else:
        try:
            english_query = translate_id_to_en(
                original_query
            )

        except Exception as error:
            raise RuntimeError(
                "Penerjemahan MyMemory gagal. "
                "Periksa koneksi atau gunakan "
                "input bahasa Inggris. "
                f"Detail: {type(error).__name__}: {error}"
            ) from error

        translated = True

    # Normalisasi akhir untuk TF-IDF
    normalized_query = normalize_search_intent(
        english_query
    )

    # Mengembalikan informasi hasil pemrosesan
    return {
        "original_query": original_query,
        "detected_language": detected_language,
        "english_query": english_query,
        "normalized_query": normalized_query,
        "translated": translated
    }
