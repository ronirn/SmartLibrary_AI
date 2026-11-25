
"""SmartLibrary AI - Tahap 6.5: Kartu rekomendasi Top 10."""

import html
import logging
import math
import re
import sys
import textwrap
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import streamlit as st
from streamlit_option_menu import option_menu


# ============================================================
# 1. PATH DAN KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Nama file CSS yang benar
CSS_PATH = BASE_DIR / "style.css"

MODEL_PATH = (
    PROJECT_DIR / "models" / "smartlibrary_tfidf.joblib"
)

DATASET_PATH = (
    PROJECT_DIR / "dataset" / "processed" / "books_clean.csv"
)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.recommender import BookRecommender


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartlibrary")

st.set_page_config(
    page_title="SmartLibrary AI",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

MENU_PENCARIAN = "Pencarian Buku"
MENU_TENTANG = "Tentang Sistem"
MENU_EVALUASI = "Hasil Evaluasi"

BAHASA_OPTIONS = ["Otomatis", "Indonesia", "English"]
BAHASA_DEFAULT = "Otomatis"

LANGUAGE_MAP = {
    "Otomatis": "auto",
    "Indonesia": "id",
    "English": "en",
}

CONTOH_TOPIK = [
    "Novel Misteri",
    "Psikologi",
    "Sejarah Dunia",
    "Filsafat dan Etika",
    "Pengembangan Diri",
]


# ============================================================
# 2. HELPER HTML DAN CSS
# ============================================================

def render_html(markup: str) -> None:
    """
    Render HTML tanpa indentasi Markdown
    yang menyebabkan HTML tampil sebagai teks.
    """
    compact = " ".join(
        line.strip()
        for line in textwrap.dedent(markup).splitlines()
        if line.strip()
    )

    st.markdown(
        compact,
        unsafe_allow_html=True,
    )


def load_css() -> None:
    """
    Membaca CSS dari app/style.css.
    """
    try:
        css = CSS_PATH.read_text(encoding="utf-8")

    except OSError:
        logger.exception(
            "Gagal membaca file CSS: %s",
            CSS_PATH,
        )

        st.warning(
            f"File CSS tidak ditemukan atau gagal dibaca: {CSS_PATH}"
        )
        return

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def escaped(value: object, fallback: str = "-") -> str:
    """
    Mengamankan teks metadata atau input pengguna
    sebelum ditampilkan dalam HTML.
    """
    value = "" if value is None else str(value)
    value = " ".join(value.split())

    if value.lower() in {"", "nan", "none"}:
        value = fallback

    return html.escape(value, quote=True)


def synopsis_text(
    value: object,
    max_chars: int = 360,
) -> str:
    """
    Membersihkan sinopsis dari tag HTML
    dan membatasi panjang teks.
    """
    value = "" if value is None else str(value)

    value = html.unescape(value)
    value = re.sub(r"<[^>]*>", " ", value)
    value = " ".join(value.split())

    if not value or value.lower() in {"nan", "none"}:
        value = "Deskripsi buku belum tersedia."

    if len(value) > max_chars:
        value = (
            value[:max_chars]
            .rsplit(" ", 1)[0]
            .rstrip() + "…"
        )

    return html.escape(value, quote=True)


def safe_cover_url(value: object) -> str:
    """
    Hanya menerima URL HTTP/HTTPS untuk sampul.
    URL HTTP dinaikkan ke HTTPS.
    """
    if not isinstance(value, str) or not value.strip():
        return ""

    value = value.strip()

    if any(char in value for char in ("\r", "\n", "\t")):
        return ""

    try:
        parsed = urlsplit(value)

        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
        ):
            return ""

        if parsed.scheme.lower() == "http":
            parsed = parsed._replace(scheme="https")

        return html.escape(
            urlunsplit(parsed),
            quote=True,
        )

    except ValueError:
        return ""


# ============================================================
# 3. SESSION STATE DAN MODEL AI
# ============================================================

DEFAULT_STATE = {
    "search_query": "",
    "search_language": BAHASA_DEFAULT,
    "search_submitted": False,
    "last_query": "",
    "last_language": BAHASA_DEFAULT,
    "search_results": [],
    "search_details": {},
    "search_error": "",
}


def init_session_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Pertahankan nilai input saat berpindah menu
    for key in ("search_query", "search_language"):
        st.session_state[key] = st.session_state[key]


def pilih_contoh(topik: str) -> None:
    """
    Mengisi search bar dari contoh pencarian.
    Tidak langsung menjalankan model.
    """
    st.session_state.search_query = topik
    st.session_state.search_submitted = False
    st.session_state.search_results = []
    st.session_state.search_details = {}
    st.session_state.search_error = ""


def jaga_pilihan_bahasa() -> None:
    if not st.session_state.get("search_language"):
        st.session_state.search_language = BAHASA_DEFAULT


@st.cache_resource(show_spinner=False)
def get_recommender() -> BookRecommender:
    """
    Memuat model TF-IDF dan metadata menggunakan cache.
    Tidak melakukan training ulang.
    """
    return BookRecommender(
        MODEL_PATH,
        DATASET_PATH,
    )


# ============================================================
# 4. SVG ILUSTRASI
# ============================================================

LOGO_ICON = """
<svg width="46" height="46" viewBox="0 0 48 48"
     fill="none" xmlns="http://www.w3.org/2000/svg"
     aria-hidden="true">
  <path d="M24 39C17 34 10 33 4 35V9C12 6 19 8 24 13V39Z"
        stroke="#145CF2" stroke-width="2.6"
        stroke-linejoin="round"/>
  <path d="M24 39C31 34 38 33 44 35V9C36 6 29 8 24 13V39Z"
        stroke="#145CF2" stroke-width="2.6"
        stroke-linejoin="round"/>
</svg>
"""


HERO_ILLUSTRATION = """
<svg viewBox="0 0 350 230"
     xmlns="http://www.w3.org/2000/svg"
     fill="none" aria-hidden="true">

  <path d="M18 153C18 111 48 78 89 78C112 78 124 96
           147 85C164 77 166 52 197 34C242 8
           295 32 312 74C331 119 318 167
           287 190H77C42 190 18 179 18 153Z"
        fill="#EFF6FF"/>

  <rect x="48" y="153" width="170" height="32" rx="11"
        fill="#FFFFFF" stroke="#145CF2" stroke-width="3"/>
  <path d="M75 153C61 162 61 176 75 185"
        stroke="#145CF2" stroke-width="3"/>

  <rect x="63" y="123" width="157" height="30" rx="10"
        fill="#FFFFFF" stroke="#145CF2" stroke-width="3"/>
  <path d="M91 123C80 132 80 144 91 153"
        stroke="#145CF2" stroke-width="3"/>

  <rect x="169" y="80" width="41" height="105" rx="6"
        fill="#FFFFFF" stroke="#145CF2" stroke-width="3"/>
  <path d="M174 96H204M174 173H204"
        stroke="#145CF2" stroke-width="2.5"/>

  <rect x="211" y="64" width="36" height="121" rx="5"
        fill="#F8FBFF" stroke="#145CF2" stroke-width="3"/>
  <path d="M216 79H242M216 174H242"
        stroke="#145CF2" stroke-width="2.5"/>

  <rect x="248" y="91" width="31" height="94" rx="5"
        fill="#FFFFFF" stroke="#145CF2" stroke-width="3"/>
  <path d="M253 105H273M253 113H273"
        stroke="#145CF2" stroke-width="2.5"/>

  <path d="M303 185C304 160 303 136 315 102"
        stroke="#145CF2" stroke-width="3"
        stroke-linecap="round"/>

  <path d="M311 146C286 140 285 127 287 117
           C302 119 310 131 311 146Z"
        fill="#93C5FD"/>

  <path d="M315 119C312 100 323 87 335 82
           C337 101 329 113 315 119Z"
        fill="#60A5FA"/>

  <path d="M307 166C321 149 334 150 343 153
           C335 168 325 172 307 166Z"
        fill="#93C5FD"/>

  <path d="M288 169H337L331 191H296L288 169Z"
        fill="#DBEAFE" stroke="#145CF2" stroke-width="2.5"/>

  <path d="M28 192H344"
        stroke="#145CF2" stroke-width="3"
        stroke-linecap="round"/>
</svg>
"""


WELCOME_ILLUSTRATION = """
<svg width="116" height="116" viewBox="0 0 120 120"
     xmlns="http://www.w3.org/2000/svg"
     fill="none" aria-hidden="true">

  <path d="M60 14V26M35 22L43 32M85 22L77 32"
        stroke="#145CF2" stroke-width="3"
        stroke-linecap="round"/>

  <path d="M60 96C46 85 31 83 16 87V44
           C32 39 48 41 60 51V96Z"
        fill="#FFFFFF" stroke="#145CF2"
        stroke-width="3" stroke-linejoin="round"/>

  <path d="M60 96C74 85 89 83 104 87V44
           C88 39 72 41 60 51V96Z"
        fill="#FFFFFF" stroke="#145CF2"
        stroke-width="3" stroke-linejoin="round"/>

  <path d="M60 51V96"
        stroke="#145CF2" stroke-width="3"/>

  <path d="M22 52C34 49 45 52 54 59
           M98 52C86 49 75 52 66 59"
        stroke="#93C5FD" stroke-width="2"
        stroke-linecap="round"/>

  <path d="M22 65C34 62 45 65 54 72
           M98 65C86 62 75 65 66 72"
        stroke="#DBEAFE" stroke-width="2"
        stroke-linecap="round"/>
</svg>
"""


COVER_PLACEHOLDER = """
<svg viewBox="0 0 64 64" width="53" height="53"
     fill="none" xmlns="http://www.w3.org/2000/svg"
     aria-hidden="true">

  <path d="M32 51C23 45 14 44 5 47V13
           C15 10 25 13 32 19V51Z"
        stroke="#2563EB" stroke-width="2.8"
        stroke-linejoin="round"/>

  <path d="M32 51C41 45 50 44 59 47V13
           C49 10 39 13 32 19V51Z"
        stroke="#2563EB" stroke-width="2.8"
        stroke-linejoin="round"/>
</svg>
"""


# ============================================================
# 5. SIDEBAR
# ============================================================

def render_sidebar() -> str:
    with st.sidebar:

        render_html(f"""
            <div class="sidebar-brand">
                {LOGO_ICON}
                <div class="sidebar-brand-title">
                    SmartLibrary AI
                </div>
            </div>
        """)

        selected = option_menu(
            menu_title=None,

            options=[
                MENU_PENCARIAN,
                MENU_TENTANG,
                MENU_EVALUASI,
            ],

            icons=[
                "search",
                "info-circle",
                "bar-chart-line",
            ],

            default_index=0,
            orientation="vertical",
            key="main_navigation",

            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "#FFFFFF",
                },

                "icon": {
                    "color": "#145CF2",
                    "font-size": "20px",
                },

                "nav-link": {
                    "font-family": "Inter, sans-serif",
                    "font-size": "15px",
                    "font-weight": "500",
                    "color": "#10234B",
                    "margin": "5px 0",
                    "padding": "15px 17px",
                    "border-radius": "13px",
                    "--hover-color": "#EFF6FF",
                },

                "nav-link-selected": {
                    "background-color": "#E8F1FF",
                    "color": "#145CF2",
                    "font-weight": "700",
                },
            },
        )

    return selected


# ============================================================
# 6. HERO SECTION
# ============================================================

def render_hero() -> None:
    render_html(f"""
        <div class="hero-section">

            <div class="hero-content">

                <div class="hero-title">
                    Temukan Buku yang Tepat untuk Anda
                </div>

                <div class="hero-description">
                    Cari buku sesuai minat Anda dengan
                    rekomendasi cerdas berbasis NLP,
                    mendukung Bahasa Indonesia dan Inggris.
                </div>

            </div>

            <div class="hero-illustration">
                {HERO_ILLUSTRATION}
            </div>

        </div>
    """)


# ============================================================
# 7. PILIHAN BAHASA
# ============================================================

def render_language_selector() -> str:
    with st.container(key="language_switch"):

        bahasa = st.segmented_control(
            label="Bahasa pencarian",
            options=BAHASA_OPTIONS,
            selection_mode="single",
            key="search_language",
            on_change=jaga_pilihan_bahasa,
            label_visibility="collapsed",
        )

    return bahasa or BAHASA_DEFAULT


# ============================================================
# 8. SEARCH BAR
# ============================================================

def render_search_form() -> tuple[bool, str]:

    with st.container(key="search_area"):

        with st.form(
            key="book_search_form",
            clear_on_submit=False,
            border=False,
        ):

            col_input, col_button = st.columns(
                [5.1, 1.1],
                gap="small",
                vertical_alignment="center",
            )

            with col_input:
                query = st.text_input(
                    "Topik atau deskripsi buku",
                    placeholder=(
                        "Cari buku berdasarkan topik, minat, "
                        "atau deskripsi yang Anda inginkan..."
                    ),
                    key="search_query",
                    label_visibility="collapsed",
                )

            with col_button:
                submitted = st.form_submit_button(
                    "Cari Buku",
                    icon=":material/search:",
                    type="primary",
                    use_container_width=True,
                )

    return submitted, query


# ============================================================
# 9. PROSES INFERENSI MODEL AI
# ============================================================

def proses_submit(query: str, bahasa: str) -> None:

    query_bersih = (query or "").strip()

    # Reset hasil sebelumnya
    st.session_state.search_results = []
    st.session_state.search_details = {}
    st.session_state.search_error = ""

    if not query_bersih:
        st.session_state.search_submitted = False

        st.warning(
            "Silakan masukkan topik atau "
            "deskripsi buku terlebih dahulu."
        )
        return

    st.session_state.search_submitted = True
    st.session_state.last_query = query_bersih
    st.session_state.last_language = bahasa

    try:
        with st.spinner(
            "Memproses bahasa dan mencari buku "
            "yang relevan..."
        ):

            output = get_recommender().recommend(
                query=query_bersih,
                language=LANGUAGE_MAP[bahasa],
                top_k=10,
            )

        st.session_state.search_results = output["books"]

        st.session_state.search_details = (
            output["query_details"]
        )

        logger.info(
            "Pencarian selesai: %s buku",
            len(output["books"]),
        )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:

        logger.exception("Pencarian gagal")
        st.session_state.search_error = str(error)

    except Exception:
        logger.exception(
            "Kesalahan tak terduga saat inferensi"
        )

        st.session_state.search_error = (
            "Pencarian belum dapat diproses. "
            "Periksa log terminal."
        )


# ============================================================
# 10. KARTU REKOMENDASI BUKU
# ============================================================

def render_book_card(book: dict, rank: int) -> str:
    """
    Membuat satu kartu buku dari data rekomendasi asli.
    """

    title = escaped(
        book.get("title"),
        "Judul tidak tersedia",
    )

    authors = escaped(
        book.get("authors"),
        "Penulis tidak tersedia",
    )

    description = synopsis_text(
        book.get("description")
    )

    isbn = escaped(
        book.get("isbn13")
    )

    cover_url = safe_cover_url(
        book.get("thumbnail")
    )

    # Skor dari model TF-IDF + Cosine Similarity
    try:
        score = float(
            book.get("similarity_score", 0)
        )
    except (TypeError, ValueError):
        score = 0.0

    if not math.isfinite(score):
        score = 0.0

    score_percent = max(
        0,
        min(100, score * 100),
    )

    score_label = (
        f"{score_percent:.2f}".replace(".", ",")
    )

    # Sampul buku dari metadata asli
    if cover_url:
        cover_image = (
            f'<img src="{cover_url}" '
            f'class="sl-book-image" '
            f'alt="" loading="lazy">'
        )
    else:
        cover_image = ""

    # Kartu buku
    return f"""
        <article class="sl-book-card">

            <div class="sl-book-cover">

                <div class="sl-cover-fallback">
                    {COVER_PLACEHOLDER}
                </div>

                {cover_image}

            </div>

            <div class="sl-book-content">

                <div class="sl-book-top">

                    <span class="sl-book-rank">
                        REKOMENDASI #{rank}
                    </span>

                    <span class="sl-book-score">
                        {score_label}% Kemiripan
                    </span>

                </div>

                <h3 class="sl-book-title">
                    {title}
                </h3>

                <div class="sl-book-author">
                    {authors}
                </div>

                <p class="sl-book-synopsis">
                    {description}
                </p>

                <div class="sl-book-footer">
                    <span class="sl-book-isbn">
                        ISBN: {isbn}
                    </span>
                </div>

            </div>

        </article>
    """


# ============================================================
# 11. HASIL REKOMENDASI TOP 10
# ============================================================

def render_hasil_pencarian() -> None:

    # Kondisi error
    if st.session_state.search_error:
        st.error(st.session_state.search_error)
        return

    # Maksimal 10 buku
    books = st.session_state.search_results[:10]

    # Kondisi tanpa hasil
    if not books:
        st.info(
            "Tidak ditemukan buku dengan kecocokan kata "
            "dalam model. Coba istilah lain atau "
            "pilih bahasa secara manual."
        )
        return

    # Teks aman untuk HTML
    query_label = escaped(
        st.session_state.last_query
    )

    language_label = escaped(
        st.session_state.last_language
    )

    # Generate seluruh kartu buku
    cards_html = "".join(
        render_book_card(book, rank)
        for rank, book in enumerate(books, start=1)
    )

    # Header dan daftar rekomendasi
    render_html(f"""
        <section class="sl-results">

            <div class="sl-results-header">

                <div>

                    <div class="sl-results-label">
                        HASIL PENCARIAN
                    </div>

                    <h2 class="sl-results-title">
                        Rekomendasi Buku untuk Anda
                    </h2>

                    <p class="sl-results-description">
                        Menampilkan
                        <strong>{len(books)} buku teratas</strong>
                        untuk pencarian
                        <strong>"{query_label}"</strong>.
                    </p>

                </div>

                <div class="sl-results-count">
                    Top {len(books)} Buku
                </div>

            </div>

            <div class="sl-language-info">
                Bahasa pencarian:
                <strong>{language_label}</strong>
            </div>

            <div class="sl-book-list">
                {cards_html}
            </div>

        </section>
    """)

    # Keterangan skor
    st.caption(
        "Skor kemiripan berasal dari TF-IDF dan "
        "Cosine Similarity, bukan akurasi model "
        "atau probabilitas bahwa buku akan disukai."
    )

    # Informasi bilingual
    with st.expander(
        "Informasi pemrosesan bilingual"
    ):

        details = st.session_state.search_details

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**Bahasa terdeteksi:**",
                details.get(
                    "detected_language",
                    "-",
                ),
            )

            st.write(
                "**Diterjemahkan:**",
                "Ya" if details.get("translated") else "Tidak",
            )

        with col2:

            st.write(
                "**Query bahasa Inggris:**",
                details.get(
                    "english_query",
                    "-",
                ),
            )

            st.write(
                "**Query TF-IDF:**",
                details.get(
                    "normalized_query",
                    "-",
                ),
            )


# ============================================================
# 12. WELCOME STATE
# ============================================================

def tampilkan_welcome_state() -> None:

    render_html(f"""
        <div class="welcome-state">

            <div class="welcome-icon">
                {WELCOME_ILLUSTRATION}
            </div>

            <div class="welcome-title">
                Mulai jelajahi buku sesuai minat Anda
            </div>

            <div class="welcome-description">
                Ketik topik atau deskripsi buku di atas
                untuk menemukan rekomendasi yang sesuai.
            </div>

            <div class="example-label">
                Contoh Pencarian:
            </div>

        </div>
    """)

    with st.container(key="example_chips"):

        cols = st.columns(
            len(CONTOH_TOPIK),
            gap="small",
        )

        for i, topik in enumerate(CONTOH_TOPIK):

            with cols[i]:

                st.button(
                    topik,
                    key=f"example_{i}",
                    icon=":material/search:",
                    on_click=pilih_contoh,
                    args=(topik,),
                )


# ============================================================
# 13. HALAMAN PENCARIAN BUKU
# ============================================================

def halaman_pencarian() -> None:

    render_hero()

    bahasa = render_language_selector()

    submitted, query = render_search_form()

    if submitted:
        proses_submit(query, bahasa)

    if st.session_state.search_submitted:
        render_hasil_pencarian()
    else:
        tampilkan_welcome_state()


# ============================================================
# 14. HALAMAN TENTANG SISTEM
# ============================================================


# ============================================================
# HALAMAN TENTANG SMARTLIBRARY AI
# ============================================================

def halaman_tentang() -> None:

    render_html("""
        <section class="about-page">

            <!-- HEADER -->

            <div class="about-header">

                <div class="about-label">
                    ABOUT SMARTLIBRARY AI
                </div>

                <h1 class="about-title">
                    Tentang SmartLibrary AI
                </h1>

                <p class="about-description">
                    SmartLibrary AI adalah sistem rekomendasi buku
                    berbasis Natural Language Processing (NLP)
                    yang membantu pengguna menemukan buku
                    sesuai minat, kebutuhan, dan topik pencarian
                    secara lebih mudah dan relevan.
                </p>

            </div>


            <!-- FITUR UTAMA -->

            <div class="about-section">

                <div class="about-section-header">

                    <h2 class="about-section-title">
                        Apa yang Ditawarkan?
                    </h2>

                    <p class="about-section-description">
                        Fitur utama yang dirancang untuk memberikan
                        pengalaman pencarian buku yang lebih baik.
                    </p>

                </div>

                <div class="about-feature-grid">

                    <article class="about-feature-card">

                        <div class="about-feature-number">
                            01
                        </div>

                        <h3 class="about-feature-title">
                            Pencarian Bilingual
                        </h3>

                        <p class="about-feature-description">
                            Mendukung pencarian menggunakan
                            Bahasa Indonesia dan Bahasa Inggris
                            melalui deteksi bahasa, penerjemahan,
                            dan normalisasi query.
                        </p>

                    </article>


                    <article class="about-feature-card">

                        <div class="about-feature-number">
                            02
                        </div>

                        <h3 class="about-feature-title">
                            Rekomendasi Top 10
                        </h3>

                        <p class="about-feature-description">
                            Menampilkan hingga 10 buku
                            berdasarkan hasil pencocokan
                            teks dan pemeringkatan relevansi,
                            lengkap dengan informasi buku.
                        </p>

                    </article>


                    <article class="about-feature-card">

                        <div class="about-feature-number">
                            03
                        </div>

                        <h3 class="about-feature-title">
                            Pencarian Berbasis Relevansi
                        </h3>

                        <p class="about-feature-description">
                            Mengombinasikan TF-IDF,
                            Cosine Similarity, Query Expansion,
                            dan Metadata Re-ranking untuk
                            meningkatkan kesesuaian hasil.
                        </p>

                    </article>

                </div>

            </div>


            <!-- CARA KERJA -->

            <div class="about-section about-workflow-section">

                <div class="about-section-header">

                    <h2 class="about-section-title">
                        Bagaimana Sistem Bekerja?
                    </h2>

                    <p class="about-section-description">
                        Proses pencarian dilakukan melalui
                        empat tahapan utama.
                    </p>

                </div>

                <div class="about-workflow-grid">

                    <div class="about-step">

                        <div class="about-step-number">
                            01
                        </div>

                        <h3 class="about-step-title">
                            Input Pencarian
                        </h3>

                        <p class="about-step-description">
                            Pengguna memasukkan topik
                            atau deskripsi buku.
                        </p>

                    </div>


                    <div class="about-step">

                        <div class="about-step-number">
                            02
                        </div>

                        <h3 class="about-step-title">
                            Pemrosesan Query
                        </h3>

                        <p class="about-step-description">
                            Sistem memproses bahasa
                            dan mengenali intent pencarian.
                        </p>

                    </div>


                    <div class="about-step">

                        <div class="about-step-number">
                            03
                        </div>

                        <h3 class="about-step-title">
                            Pencocokan Buku
                        </h3>

                        <p class="about-step-description">
                            TF-IDF dan Cosine Similarity
                            menghitung kemiripan teks buku.
                        </p>

                    </div>


                    <div class="about-step">

                        <div class="about-step-number">
                            04
                        </div>

                        <h3 class="about-step-title">
                            Hasil Rekomendasi
                        </h3>

                        <p class="about-step-description">
                            Buku diurutkan berdasarkan
                            skor relevansi akhir.
                        </p>

                    </div>

                </div>

            </div>


            <!-- TEKNOLOGI -->

            <div class="about-tech-panel">

                <div class="about-tech-content">

                    <div class="about-tech-label">
                        TEKNOLOGI SISTEM
                    </div>

                    <h2 class="about-tech-title">
                        Dibangun dengan Pendekatan NLP
                    </h2>

                    <p class="about-tech-description">
                        SmartLibrary AI menggunakan TF-IDF
                        untuk representasi teks dan
                        Cosine Similarity untuk menghitung
                        kemiripan antara query pengguna
                        dan informasi buku.

                        Sistem juga dilengkapi
                        Intent-Aware Query Expansion
                        dan Metadata Re-ranking
                        untuk membantu meningkatkan
                        relevansi rekomendasi.
                    </p>

                </div>

                <div class="about-tech-tags">

                    <span class="about-tech-tag">
                        Python
                    </span>

                    <span class="about-tech-tag">
                        Streamlit
                    </span>

                    <span class="about-tech-tag">
                        NLP
                    </span>

                    <span class="about-tech-tag">
                        TF-IDF
                    </span>

                    <span class="about-tech-tag">
                        Cosine Similarity
                    </span>

                    <span class="about-tech-tag">
                        Query Expansion
                    </span>

                </div>

            </div>

        </section>
    """)


# ============================================================
# 15. HALAMAN HASIL EVALUASI
# ============================================================


# ============================================================
# HALAMAN HASIL EVALUASI SMARTLIBRARY AI
# ============================================================

def halaman_evaluasi() -> None:

    import pandas as pd
    from pathlib import Path

    # --------------------------------------------------------
    # 1. LOKASI FILE EVALUASI
    # --------------------------------------------------------

    report_dir = (
        PROJECT_DIR / "reports" / "evaluation"
    )

    ranking_path = (
        report_dir
        / "query_expansion_weight_comparison.csv"
    )

    label_path = (
        report_dir
        / "query_expansion_relevance_labels.csv"
    )

    # --------------------------------------------------------
    # 2. MEMBACA FILE CSV
    # --------------------------------------------------------

    def read_report(path):

        if not path.is_file():
            return None

        try:
            return pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )

        except (
            OSError,
            UnicodeError,
            pd.errors.ParserError,
        ) as error:

            st.warning(
                f"Gagal membaca {path.name}: {error}"
            )

            return None

    rankings = read_report(ranking_path)
    labels = read_report(label_path)

    required_ranking = {
        "query",
        "configuration",
        "rank",
        "isbn13",
        "title",
        "similarity_score",
    }

    required_labels = {
        "query",
        "isbn13",
        "relevance",
    }

    if rankings is not None:

        missing = (
            required_ranking - set(rankings.columns)
        )

        if missing:

            st.warning(
                "Kolom hasil ranking tidak lengkap: "
                f"{sorted(missing)}"
            )

            rankings = None

    if labels is not None:

        missing = (
            required_labels - set(labels.columns)
        )

        if missing:

            st.warning(
                "Kolom label relevansi tidak lengkap: "
                f"{sorted(missing)}"
            )

            labels = None

    # --------------------------------------------------------
    # 3. HITUNG STATISTIK DARI DATA ASLI
    # --------------------------------------------------------

    if rankings is not None:

        total_queries = rankings["query"].nunique()

        total_configs = (
            rankings["configuration"].nunique()
        )

        total_records = len(rankings)

    else:

        total_queries = None
        total_configs = None
        total_records = None

    if labels is not None:

        total_labels = len(labels)

        label_values = (
            labels["relevance"]
            .astype(str)
            .str.strip()
        )

        completed_labels = int(
            label_values.isin(["0", "1"]).sum()
        )

    else:

        total_labels = None
        completed_labels = None

    def metric_value(value):
        return "—" if value is None else str(value)

    progress_label = (
        "—"
        if total_labels is None
        else f"{completed_labels}/{total_labels}"
    )

    # --------------------------------------------------------
    # 4. HEADER DAN METRIC CARDS
    # --------------------------------------------------------

    render_html(f"""
        <section class="eval-page">

            <div class="eval-header">

                <div class="eval-label">
                    MODEL EVALUATION
                </div>

                <h1 class="eval-title">
                    Hasil Evaluasi Sistem
                </h1>

                <p class="eval-description">
                    Ringkasan eksperimen dan analisis hasil
                    pemeringkatan buku pada SmartLibrary AI,
                    berdasarkan data pengujian yang tersedia.
                </p>

            </div>

            <div class="eval-metrics-grid">

                <div class="eval-metric-card">

                    <div class="eval-metric-label">
                        Query Pengujian
                    </div>

                    <div class="eval-metric-value">
                        {metric_value(total_queries)}
                    </div>

                    <div class="eval-metric-note">
                        Topik pencarian yang diuji
                    </div>

                </div>

                <div class="eval-metric-card">

                    <div class="eval-metric-label">
                        Konfigurasi Bobot
                    </div>

                    <div class="eval-metric-value">
                        {metric_value(total_configs)}
                    </div>

                    <div class="eval-metric-note">
                        Termasuk baseline
                    </div>

                </div>

                <div class="eval-metric-card">

                    <div class="eval-metric-label">
                        Hasil Pemeringkatan
                    </div>

                    <div class="eval-metric-value">
                        {metric_value(total_records)}
                    </div>

                    <div class="eval-metric-note">
                        Total baris hasil eksperimen
                    </div>

                </div>

                <div class="eval-metric-card">

                    <div class="eval-metric-label">
                        Label Relevansi
                    </div>

                    <div class="eval-metric-value">
                        {progress_label}
                    </div>

                    <div class="eval-metric-note">
                        Buku yang sudah dinilai
                    </div>

                </div>

            </div>

            <div class="eval-context-note">

                Laporan ini menampilkan eksperimen bobot
                Query Expansion. Hasilnya tidak otomatis
                merepresentasikan evaluasi algoritma
                pemeringkatan versi terbaru.

            </div>

        </section>
    """)

    # --------------------------------------------------------
    # 5. DATA BELUM TERSEDIA
    # --------------------------------------------------------

    if rankings is None:

        st.info(
            "File hasil eksperimen belum tersedia. "
            "Halaman akan menampilkan perbandingan ranking "
            "setelah laporan CSV tersedia pada "
            "folder reports/evaluation."
        )

        return

    # --------------------------------------------------------
    # 6. NAVIGASI EVALUASI
    # --------------------------------------------------------

    tab_ranking, tab_relevance = st.tabs([
        "Perbandingan Ranking",
        "Evaluasi Relevansi",
    ])

    # --------------------------------------------------------
    # TAB 1 - PERBANDINGAN RANKING
    # --------------------------------------------------------

    with tab_ranking:

        render_html("""
            <div class="eval-section-header">

                <h2 class="eval-section-title">
                    Perbandingan Rekomendasi
                </h2>

                <p class="eval-section-description">
                    Lihat perubahan urutan buku pada
                    berbagai konfigurasi bobot
                    Query Expansion.
                </p>

            </div>
        """)

        available_queries = sorted(
            rankings["query"]
            .dropna()
            .unique()
            .tolist()
        )

        if not available_queries:

            st.info(
                "Belum ada query pengujian."
            )

        else:

            with st.container(
                key="eval_query_filter"
            ):

                selected_query = st.selectbox(
                    "Pilih topik pencarian",
                    available_queries,
                    key="evaluation_query",
                )

            subset = rankings.loc[
                rankings["query"] == selected_query
            ].copy()

            subset["rank"] = pd.to_numeric(
                subset["rank"],
                errors="coerce",
            )

            subset = subset.dropna(
                subset=["rank"]
            )

            if subset.empty:

                st.info(
                    "Tidak ada hasil ranking untuk "
                    "query yang dipilih."
                )

            else:

                subset["rank"] = (
                    subset["rank"].astype(int)
                )

                comparison = subset.pivot_table(
                    index="rank",
                    columns="configuration",
                    values="title",
                    aggfunc="first",
                )

                # Urutan konfigurasi eksperimen
                config_order = [
                    "Baseline",
                    "80_20",
                    "70_30",
                    "60_40",
                    "50_50",
                ]

                ordered_columns = [
                    col
                    for col in config_order
                    if col in comparison.columns
                ]

                remaining_columns = [
                    col
                    for col in comparison.columns
                    if col not in ordered_columns
                ]

                comparison = comparison[
                    ordered_columns + remaining_columns
                ]

                comparison = (
                    comparison.sort_index()
                    .fillna("-")
                )

                comparison.index.name = "Peringkat"

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    height=430,
                )

                st.caption(
                    "Tabel menunjukkan judul buku "
                    "pada setiap peringkat. "
                    "Perubahan ranking tidak otomatis "
                    "menunjukkan peningkatan relevansi."
                )

    # --------------------------------------------------------
    # TAB 2 - EVALUASI RELEVANSI
    # --------------------------------------------------------

    with tab_relevance:

        render_html("""
            <div class="eval-section-header">

                <h2 class="eval-section-title">
                    Pengukuran Relevansi Buku
                </h2>

                <p class="eval-section-description">
                    Penilaian relevansi digunakan untuk
                    mengukur kualitas hasil rekomendasi,
                    bukan hanya kemiripan teks.
                </p>

            </div>
        """)

        if labels is None:

            st.info(
                "Data label relevansi belum tersedia."
            )

        else:

            total = len(labels)

            valid_labels = (
                labels["relevance"]
                .astype(str)
                .str.strip()
            )

            completed = int(
                valid_labels.isin(["0", "1"]).sum()
            )

            progress = (
                completed / total
                if total > 0
                else 0.0
            )

            st.progress(
                progress,
                text=(
                    f"Penilaian relevansi: "
                    f"{completed} dari {total} buku"
                ),
            )

            # Pemeriksaan label duplikat
            duplicate_labels = labels.duplicated(
                subset=["query", "isbn13"]
            ).any()

            if duplicate_labels:

                st.warning(
                    "Terdapat pasangan query-ISBN "
                    "duplikat dalam file label. "
                    "Periksa file sebelum menghitung metrik."
                )

            elif total == 0:

                st.info(
                    "File penilaian masih kosong."
                )

            elif completed < total:

                st.info(
                    "Precision@10 belum dihitung karena "
                    "masih ada label relevansi kosong. "
                    "Isi kolom relevance dengan 1 "
                    "untuk relevan atau 0 "
                    "untuk tidak relevan."
                )

            else:

                # --------------------------------------------
                # COMPUTE PRECISION@10
                # --------------------------------------------

                label_data = labels[
                    ["query", "isbn13", "relevance"]
                ].copy()

                label_data["relevance"] = (
                    label_data["relevance"]
                    .astype(int)
                )

                merged = rankings.merge(
                    label_data,
                    on=["query", "isbn13"],
                    how="left",
                    validate="many_to_one",
                )

                # Pastikan semua ranking memiliki label
                all_labeled = (
                    merged["relevance"].notna().all()
                )

                # Pastikan setiap konfigurasi memuat 10 hasil
                group_counts = merged.groupby(
                    ["query", "configuration"]
                ).size()

                complete_top10 = (
                    not group_counts.empty
                    and group_counts.eq(10).all()
                )

                if not all_labeled or not complete_top10:

                    st.warning(
                        "Data ranking atau label belum "
                        "lengkap untuk menghitung "
                        "Precision@10 secara valid."
                    )

                else:

                    precision = (
                        merged.groupby(
                            ["query", "configuration"],
                            as_index=False,
                        )
                        .agg(
                            precision_at_10=(
                                "relevance", "mean"
                            )
                        )
                    )

                    summary = (
                        precision.groupby(
                            "configuration",
                            as_index=False,
                        )
                        .agg(
                            mean_precision_at_10=(
                                "precision_at_10",
                                "mean",
                            )
                        )
                    )

                    summary["Mean Precision@10"] = (
                        summary["mean_precision_at_10"]
                        * 100
                    ).round(2)

                    summary = summary.rename(
                        columns={
                            "configuration":
                            "Konfigurasi"
                        }
                    )

                    st.markdown(
                        "#### Rata-rata Precision@10"
                    )

                    st.dataframe(
                        summary[
                            [
                                "Konfigurasi",
                                "Mean Precision@10",
                            ]
                        ],
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Mean Precision@10":
                                st.column_config.NumberColumn(
                                    "Mean Precision@10 (%)",
                                    format="%.2f%%",
                                )
                        },
                    )

                    st.caption(
                        "Precision@10 dihitung dari "
                        "label relevansi 0/1. "
                        "Nilai ini hanya mencerminkan "
                        "query dan konfigurasi yang diuji."
                    )

        # Penjelasan metodologi
        render_html("""
            <div class="eval-method-panel">

                <div class="eval-method-label">
                    METODOLOGI
                </div>

                <h3 class="eval-method-title">
                    Bagaimana relevansi diukur?
                </h3>

                <p class="eval-method-description">
                    Precision@10 mengukur proporsi
                    buku relevan pada sepuluh
                    rekomendasi teratas.

                    Buku dinilai berdasarkan
                    kesesuaian isi dengan kebutuhan
                    pencarian, bukan berdasarkan
                    skor Cosine Similarity saja.
                </p>

                <div class="eval-formula">
                    Precision@10 =
                    Jumlah Buku Relevan / 10
                </div>

            </div>
        """)


# ============================================================
# 16. ROUTING DAN MAIN
# ============================================================

PAGES = {
    MENU_PENCARIAN: halaman_pencarian,
    MENU_TENTANG: halaman_tentang,
    MENU_EVALUASI: halaman_evaluasi,
}


def main() -> None:

    load_css()
    init_session_state()

    selected = render_sidebar()

    PAGES.get(
        selected,
        halaman_pencarian,
    )()


if __name__ == "__main__":
    main()
