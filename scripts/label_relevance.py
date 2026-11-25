
"""
SMARTLIBRARY AI - RELEVANCE ANNOTATION

Halaman penilaian relevansi buku secara manual.

Input:
reports/evaluation/query_expansion_relevance_labels.csv

Output:
File CSV yang sama, dengan kolom relevance berisi:
1 = Relevan
0 = Tidak relevan
Kosong = Belum dinilai

Tidak mengubah model atau hasil ranking.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 1. KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

LABEL_FILE = (
    ROOT
    / "reports"
    / "evaluation"
    / "query_expansion_relevance_labels.csv"
)

st.set_page_config(
    page_title="Evaluasi Relevansi - SmartLibrary AI",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# 2. MEMBACA LABEL
# ============================================================

def load_labels():

    if not LABEL_FILE.is_file():
        st.error(
            "File penilaian tidak ditemukan. "
            "Jalankan compare_expansion_weights.py dahulu."
        )
        st.stop()

    df = pd.read_csv(
        LABEL_FILE,
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "query",
        "isbn13",
        "title",
        "authors",
        "description",
        "relevance",
    }

    missing = required - set(df.columns)

    if missing:
        st.error(
            f"Kolom CSV tidak lengkap: {sorted(missing)}"
        )
        st.stop()

    if df.duplicated(
        ["query", "isbn13"]
    ).any():
        st.error(
            "CSV memiliki pasangan query-ISBN duplikat."
        )
        st.stop()

    return df


# ============================================================
# 3. MENYIMPAN LABEL
# ============================================================

def save_labels(df):

    temporary_file = LABEL_FILE.with_suffix(".tmp")

    try:
        df.to_csv(
            temporary_file,
            index=False,
            encoding="utf-8-sig",
        )

        # Ganti file setelah penulisan selesai
        os.replace(
            temporary_file,
            LABEL_FILE,
        )

        return True

    except OSError as error:

        st.error(
            "Gagal menyimpan CSV. "
            "Pastikan file tidak sedang dibuka "
            f"di Excel. Detail: {error}"
        )

        return False


# ============================================================
# 4. TAMPILAN UTAMA
# ============================================================

def main():

    st.title("📚 Evaluasi Relevansi SmartLibrary AI")

    st.write(
        "Berikan penilaian berdasarkan kesesuaian "
        "isi buku dengan kebutuhan pencarian pengguna."
    )

    df = load_labels()

    # Normalisasi nilai label
    df["relevance"] = (
        df["relevance"].astype(str).str.strip()
    )

    total = len(df)

    completed = int(
        df["relevance"].isin(["0", "1"]).sum()
    )

    percentage = (
        completed / total if total else 0.0
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total buku yang dinilai",
        total,
    )

    col2.metric(
        "Sudah dinilai",
        completed,
    )

    col3.metric(
        "Belum dinilai",
        total - completed,
    )

    st.progress(
        percentage,
        text=f"Progress penilaian: {percentage:.1%}",
    )

    if st.session_state.get("label_saved"):
        st.success(
            st.session_state.pop("label_saved")
        )

    st.divider()

    # --------------------------------------------------------
    # PEDOMAN PENILAIAN
    # --------------------------------------------------------

    with st.expander(
        "Pedoman menentukan relevansi",
        expanded=True,
    ):

        st.markdown(
            """
            **Relevan (1)**

            Buku membahas topik atau kebutuhan yang
            dicari pengguna secara jelas.

            **Tidak Relevan (0)**

            Buku membahas topik berbeda atau hanya
            memiliki kesamaan kata tanpa kesesuaian
            makna yang memadai.

            **Belum dinilai**

            Gunakan jika kamu masih membutuhkan
            informasi untuk memastikan relevansi.

            Penilaian didasarkan pada judul dan
            sinopsis, bukan skor Cosine Similarity.
            """
        )

    # --------------------------------------------------------
    # PILIH QUERY
    # --------------------------------------------------------

    queries = df["query"].drop_duplicates().tolist()

    query = st.selectbox(
        "Pilih query pencarian",
        queries,
    )

    subset = df.loc[
        df["query"] == query
    ]

    query_completed = int(
        subset["relevance"].isin(["0", "1"]).sum()
    )

    st.caption(
        f"{query_completed}/{len(subset)} "
        "buku sudah dinilai untuk query ini."
    )

    st.divider()

    # --------------------------------------------------------
    # FORM PENILAIAN
    # --------------------------------------------------------

    choices = [
        "Belum dinilai",
        "Relevan (1)",
        "Tidak relevan (0)",
    ]

    value_map = {
        "Belum dinilai": "",
        "Relevan (1)": "1",
        "Tidak relevan (0)": "0",
    }

    index_map = {
        "": 0,
        "1": 1,
        "0": 2,
    }

    submitted_values = {}

    with st.form("relevance_form"):

        for number, (index, book) in enumerate(
            subset.iterrows(),
            start=1,
        ):

            st.subheader(
                f"{number}. {book['title']}"
            )

            st.caption(
                f"Penulis: {book['authors']} "
                f"| ISBN: {book['isbn13']}"
            )

            description = (
                book["description"]
                or "Deskripsi belum tersedia."
            )

            with st.expander(
                "Baca sinopsis",
                expanded=True,
            ):
                st.write(description)

            current_value = book["relevance"]

            selected = st.radio(
                "Apakah buku ini relevan?",
                choices,
                index=index_map.get(
                    current_value,
                    0,
                ),
                horizontal=True,
                key=(
                    f"relevance_{query}_"
                    f"{book['isbn13']}"
                ),
            )

            submitted_values[index] = value_map[
                selected
            ]

            st.divider()

        submit = st.form_submit_button(
            "Simpan Penilaian Query Ini",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # SIMPAN PERUBAHAN
    # --------------------------------------------------------

    if submit:

        for index, relevance in submitted_values.items():
            df.at[index, "relevance"] = relevance

        if save_labels(df):

            st.session_state["label_saved"] = (
                f"Penilaian untuk '{query}' "
                "berhasil disimpan."
            )

            st.rerun()

    # --------------------------------------------------------
    # STATUS PENILAIAN
    # --------------------------------------------------------

    if completed == total and total > 0:

        st.success(
            "Seluruh penilaian relevansi telah selesai. "
            "Sekarang jalankan evaluasi Precision@10."
        )


if __name__ == "__main__":
    main()
