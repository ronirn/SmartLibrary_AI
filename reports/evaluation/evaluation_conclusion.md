
# Kesimpulan Evaluasi SmartLibrary AI

## 1. Hasil Evaluasi Relevansi

Sistem rekomendasi buku SmartLibrary AI dikembangkan
menggunakan metode TF-IDF dan Cosine Similarity.

Evaluasi dilakukan terhadap lima skenario pencarian
dengan total 50 rekomendasi buku yang dinilai
berdasarkan kesesuaian judul, kategori, dan sinopsis.

Hasil evaluasi menunjukkan:

- Mean Precision@5: **72.00%**
- Mean Precision@10: **72.00%**

Skenario dengan Precision@10 tertinggi adalah
**Filsafat dan Etika**
(90.00%).

Skenario dengan Precision@10 terendah adalah
**Pengembangan Diri**
(50.00%).

## 2. Evaluasi Konsistensi Bilingual

Dari 5 pasangan query, 5 berhasil diuji. Rata-rata Top-10 overlap pada pengujian yang berhasil sebesar 78.00%.

Evaluasi menggunakan pasangan query bahasa Indonesia
dan bahasa Inggris dengan pemilihan bahasa manual.

## 3. Analisis Keterbatasan

Sistem masih memiliki beberapa keterbatasan:

1. TF-IDF berfokus pada kemiripan kata sehingga
   belum sepenuhnya memahami makna atau konteks.
2. Buku dengan kata yang cocok belum tentu relevan.
3. Penerjemahan Indonesia-Inggris bergantung pada
   layanan eksternal MyMemory.
4. Evaluasi relevansi didasarkan pada lima skenario
   dan penilaian sinopsis, sehingga belum mewakili
   seluruh kemungkinan permintaan pengguna.
5. Pengujian konsistensi bilingual tidak secara
   langsung mengukur ketepatan relevansi buku.

## 4. Kesimpulan

SmartLibrary AI berhasil mengimplementasikan
rekomendasi buku berbasis NLP menggunakan TF-IDF
dan Cosine Similarity.

Berdasarkan evaluasi awal, sistem memperoleh
rata-rata Precision@5 sebesar 72.00%
dan Precision@10 sebesar 72.00%.

Hasil tersebut menunjukkan bahwa sistem dapat
menghasilkan rekomendasi relevan pada sebagian
besar skenario yang diuji, tetapi kualitas
pemeringkatan masih dapat ditingkatkan.

Pengembangan selanjutnya dapat mempertimbangkan
pembobotan judul dan kategori serta metode
pemahaman semantik yang lebih baik.
