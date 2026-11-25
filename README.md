# SmartLibrary AI

**Judul:** Sistem Rekomendasi Buku Berbasis Natural Language Processing Menggunakan TF-IDF dan Cosine Similarity.  
**Mata kuliah:** Prinsip Kecerdasan Artifisial (PKA).  
**Tahap saat ini:** struktur project dan notebook awal (belum melakukan pembersihan, penerjemahan, modeling, atau evaluasi).

## Sumber data

- `dataset/raw/books_with_categories.csv` = file asli dari pengguna (jangan ditimpa).
- Kolom aslinya: `isbn13`, `isbn10`, `title`, `authors`, `categories`, `thumbnail`, `description`, `published_year`, `average_rating`, `num_pages`, `ratings_count`, `title_and_subtitle`, `tagged_description`, `simple_categories`.
- `dataset/processed/` = tujuan hasil pembersihan dan penyesuaian bahasa di tahap berikutnya.

## Urutan notebook

1. `notebooks/01_eksplorasi_dataset.ipynb`: mengenali kolom, data kosong, distribusi dan duplikasi.
2. `notebooks/02_pembersihan_dataset.ipynb`: membersihkan data tanpa mengubah CSV asli, menghasilkan `books_clean.csv`.
3. `notebooks/03_persiapan_bahasa_indonesia.ipynb`: memutuskan strategi bilingual/terjemahan yang menjaga istilah buku.
4. `notebooks/04_tfidf_cosine_similarity.ipynb`: model rekomendasi Top-K, rekomendasi buku serupa.
5. `notebooks/05_evaluasi_rekomendasi.ipynb`: uji kueri, penilaian relevansi, Precision@K dan perbandingan baseline.

Selanjutnya UI aplikasi Streamlit akan dibangun di `app/` setelah rekomendasi dapat diuji melalui notebook.

## Mulai di VS Code

1. Ekstrak ZIP, buka folder `SmartLibrary_AI` pada VS Code.
2. Pastikan CSV berada di `dataset/raw/books_with_categories.csv` (sudah disertakan).
3. Buka notebook `01_eksplorasi_dataset.ipynb`, lalu pilih kernel Python yang sesuai.
4. Kode cleaning dan modeling akan diisi bertahap, per cell, pada langkah berikutnya.

**Penting:** notebook di ZIP ini sengaja merupakan kerangka kerja; belum mencantumkan hasil eksperimen yang tidak dijalankan.
