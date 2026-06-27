# Implementasi Case-Based Reasoning dengan TF-IDF dan Support Vector Machine untuk Retrieval dan Prediksi Putusan Perdata Wanprestasi di Pengadilan Negeri Surabaya"

## Penjelasan Judul

| Frasa | Penjelasan |
|---|---|
| **Sistem Case-Based Reasoning** | Sistem yang memecahkan masalah baru berdasarkan pengalaman kasus lama, mengikuti siklus Retrieve → Reuse → Revise → Retain secara lengkap |
| **Berbasis TF-IDF** | Representasi vektor dokumen menggunakan Term Frequency-Inverse Document Frequency (unigram+bigram, max 5.000 fitur) untuk mengukur kemiripan antar teks putusan |
| **dan SVM** | Support Vector Machine dengan kernel linear digunakan sebagai model klasifikasi untuk memprediksi label putusan (Dikabulkan / Ditolak / Tidak Dapat Diterima) |
| **untuk Analisis Putusan Perdata Wanprestasi** | Domain hukum yang dianalisis: gugatan ingkar janji (wanprestasi) dalam ranah hukum perdata |
| **di Pengadilan Negeri Surabaya** | Sumber data: 54 dokumen putusan yang diambil dari Direktori Putusan Mahkamah Agung RI, khusus dari PN Surabaya |

---

## Deskripsi

Sistem **Case-Based Reasoning (CBR)** sederhana berbasis Python untuk memprediksi kemungkinan amar
putusan (Dikabulkan / Ditolak-NO) pada perkara perdata gugatan **wanprestasi** di **Pengadilan Negeri
Surabaya**, menggunakan data putusan yang dipublikasikan di [Direktori Putusan Mahkamah Agung RI](https://putusan3.mahkamahagung.go.id).

Disusun untuk memenuhi tugas **SubCPMK-3 — Mata Kuliah Penalaran Komputer** (siklus CBR: Membangun Case
Base → Case Representation → Case Retrieval → Case/Solution Reuse → Revisi & Retain (opsional) →
Evaluasi Model).

---

## Anggota Tim

| Nama | NIM |
|---|---|
| Marcela Setiawan | 202310370311393 |
| Faradita Bilbiana Eka Saputeri | 202310370311074 |

---

## Statistik Dataset

| Keterangan | Jumlah |
|---|---|
| PDF diunduh | 80 dokumen |
| Dokumen valid (Putusan) | 54 dokumen |
| Dokumen dibuang di Tahap 1 (Penetapan/Akta/Pencabutan) | 22 dokumen |
| Dokumen dibuang di Tahap 2 (bukan Putusan substantif) | 4 dokumen |
| Case base final | 54 kasus |
| Split train | ±43 kasus (80%) |
| Split test | ±11 kasus (20%) |

---

## Siklus CBR yang Diimplementasikan

```
┌─────────────────┐   ┌──────────────────┐   ┌────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│ 1. Case Base    │ → │ 2. Representation│ → │ 3. Retrieval   │ → │ 4. Solution Reuse  │ → │ 5. Evaluation    │
│ scraping + clean│   │ metadata + fitur │   │ TF-IDF + SVM   │   │ majority/weighted  │   │ Accuracy/Prec/   │
│                 │   │                  │   │ cosine retrieve│   │ vote -> prediksi   │   │ Recall/F1        │
└─────────────────┘   └──────────────────┘   └────────────────┘   └────────────────────┘   └──────────────────┘
                                                                                                       │
                                                                          ┌────────────────────────────┘
                                                                          ▼
                                                              ┌────────────────────────┐
                                                              │ 6. Revisi & Retain     │  (opsional, iteratif)
                                                              │ kasus baru tervalidasi │
                                                              │ -> masuk case base     │
                                                              └────────────────────────┘
```

---

## Struktur Repository

```
.
├── README.md                          <- dokumen ini
├── requirements.txt                   <- daftar dependensi Python
├── notebooks/                         <- satu notebook per tahap siklus CBR
│   ├── 01_case_base_scraping.ipynb        Tahap 1 — scraping & cleaning
│   ├── 02_case_representation.ipynb       Tahap 2 — metadata & feature engineering
│   ├── 03_case_retrieval.ipynb            Tahap 3 — TF-IDF, SVM, retrieve()
│   ├── 04_case_solution_reuse.ipynb       Tahap 4 — predict_outcome()
│   ├── 05_model_evaluation.ipynb          Tahap 5 — Accuracy/Precision/Recall/F1
│   └── Revisi & Retain.ipynb              Tahap 6 (opsional) — revisi & retain
├── data/
│   ├── raw/
│   │   ├── pdf/                       <- PDF asli hasil download dari MA RI
│   │   └── case_001.txt ... case_NNN.txt  <- teks bersih hasil cleaning (Tahap 1)
│   ├── processed/
|   |   ├── tokens/
|   |   |   └── case_001_tokens.json...case_054_tokens.json
│   │   ├── cases.csv                  <- case base terstruktur (Tahap 2)
|   |   ├── case_base_structured.json
|   |   ├── cases.json
|   |   └── metadata_raw.csv
│   ├── eval/
│   │   ├── queries.json               <- query uji + ground-truth (Tahap 3)
│   │   ├── retrieval_metrics.csv      <- hasil evaluasi retrieval (Tahap 5)
│   │   └── prediction_metrics.csv     <- hasil evaluasi klasifikasi (Tahap 5)
│   └── results/
│       └── predictions.csv            <- hasil demo predict_outcome() (Tahap 4)
└── logs/
    ├── cleaning.log                   <- riwayat pembersihan teks (Tahap 1, opsional)
    └── retain_log.json                <- riwayat retain & asal-usul pasal (Tahap 6)
```

> **Catatan:** folder `data/raw/pdf/` berisi file besar (puluhan PDF). Jika ukuran repository menjadi
> masalah saat upload ke GitHub, folder ini boleh diabaikan (`.gitignore`) - hasil scraping yang penting
> untuk dinilai adalah `data/raw/*.txt` dan `data/processed/cases.csv`, yang jauh lebih kecil.


---

## Instalasi

### Prasyarat
- Python **3.10+**
- Google Chrome (untuk scraping Tahap 1)
- ChromeDriver yang sesuai versi Chrome

### 1. Clone repository

```bash
git clone https://github.com/Marcelasetiawan/Tugas-Akhir-Penalaran-Komputer.git
cd Tugas-Akhir-Penalaran-Komputer
```

### 2. Buat virtual environment (disarankan)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan Jupyter

```bash
jupyter notebook
# atau
jupyter lab
```

## Cara Menjalankan Pipeline End-to-End

Jalankan notebook **berurutan sesuai nomor**, karena setiap tahap membaca output tahap sebelumnya dari
folder `data/`.

> **Penting — soal working directory:** semua path data di notebook (mis. `'data/processed/cases.csv'`)
> ditulis relatif terhadap **root repository**, bukan terhadap folder `notebooks/`. Karena Jupyter secara
> default menjalankan kernel dengan working directory = folder tempat file `.ipynb` berada, setiap
> notebook di sini sudah disisipi **cell "Setup — Jangkar Working Directory"** sebagai cell pertama yang
> otomatis pindah ke root repository sebelum cell lain dijalankan. Cukup pastikan Anda menjalankan cell
> ini (cell ke-1) **sebelum** cell lainnya setiap kali membuka ulang notebook — atau gunakan
> *Run All Cells*, karena urutannya sudah benar secara otomatis.

| # | Notebook | Input | Output | Catatan |
|---|---|---|---|---|
| 1 | `01_case_base_scraping.ipynb` | — (scraping live dari MA RI) | `data/raw/pdf/*.pdf`, `data/raw/case_*.txt` | Lihat **langkah khusus** di bawah (mode debug Chrome) |
| 2 | `02_case_representation.ipynb` | `data/raw/case_*.txt` | `data/processed/cases.csv`, `cases.json` | Ekstraksi metadata (no. perkara, pasal, pihak, dll.) + feature engineering |
| 3 | `03_case_retrieval.ipynb` | `data/processed/cases.csv` | `data/eval/queries.json` | TF-IDF + cosine similarity, training SVM, fungsi `retrieve()` |
| 4 | `04_case_solution_reuse.ipynb` | `data/processed/cases.csv` | `data/results/predictions.csv` | Fungsi `predict_outcome()`, demo 5 kasus baru |
| 5 | `05_model_evaluation.ipynb` | `cases.csv`, `queries.json`, `predictions.csv` | `data/eval/retrieval_metrics.csv`, `prediction_metrics.csv` | Accuracy, Precision, Recall, F1 + bar chart + error analysis |
| 6 | `Revisi & Retain.ipynb` *(opsional)* | `cases.csv`, `predictions.csv` | `cases.csv` (diperbarui), `logs/retain_log.json` | Menambah kasus baru yang prediksinya terbukti tepat ke case base |

Setelah Tahap 6 dijalankan, **Tahap 3–5 boleh dijalankan ulang** agar kasus hasil retain ikut masuk ke
model TF-IDF/SVM dan ke evaluasi — ini mencerminkan sifat CBR yang iteratif (*Retrieve → Reuse → Revise
→ Retain → kembali ke Retrieve*).

### Langkah Khusus untuk Tahap 1 (Scraping)

Website Direktori Putusan MA RI menggunakan proteksi Cloudflare, sehingga scraping dilakukan lewat sesi
Chrome yang sudah login manual (bukan headless browser otomatis). Sebelum menjalankan
`01_case_base_scraping.ipynb`:

1. Tutup semua jendela Chrome yang sedang berjalan.
2. Buka **Command Prompt** (bukan PowerShell) dan jalankan:
   ```
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome_debug"
   ```
   (Sesuaikan path `chrome.exe` jika menggunakan macOS/Linux, mis. `google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome_debug`)
3. Di jendela Chrome yang terbuka, kunjungi `https://putusan3.mahkamahagung.go.id` dan selesaikan
   verifikasi Cloudflare secara manual.
4. **Jangan tutup** jendela Chrome tersebut, lalu jalankan notebook `01_case_base_scraping.ipynb` dari
   awal sampai akhir secara berurutan.

Jika Anda hanya ingin mereproduksi Tahap 2 dan seterusnya **tanpa scraping ulang**, cukup pastikan folder
`data/raw/` sudah berisi `case_*.txt` (sudah disertakan di repository ini) dan langsung lanjut ke
notebook nomor 2.

### Contoh Perintah (mode non-interaktif / headless)

Untuk menjalankan seluruh pipeline tanpa membuka antarmuka Jupyter secara manual (mis. di CI atau
terminal), gunakan `jupyter nbconvert` per notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_case_representation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_case_retrieval.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_case_solution_reuse.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_model_evaluation.ipynb
```

(Notebook 01 disarankan tetap dijalankan interaktif karena membutuhkan verifikasi Cloudflare manual di
Chrome; notebook 06 bersifat opsional dan dijalankan sesuai kebutuhan retain.)


## Penjelasan Pertahap

### Tahap 1 — Membangun Case Base
**File:** `notebooks/tahap_01_case_base_scraping_wanprestasi_pn_surabaya.ipynb`

**Yang dilakukan:**
- Scraping link putusan dari Direktori MA RI (domain Wanprestasi, PN Surabaya)
- Download 80 PDF putusan
- Konversi PDF → TXT menggunakan `pdfminer`
- Preprocessing: hapus header/footer/watermark, normalisasi lowercase, tokenisasi
- Filter otomatis: buang 22 dokumen yang bukan Putusan (Penetapan/Akta/Pencabutan)
- Rename `case_id` agar berurutan (`case_001` – `case_058`)

**Output:**
- `data/raw/pdf/` — 80 file PDF
- `data/raw/case_001.txt` – `case_058.txt` — teks bersih
- `data/processed/metadata_raw.csv`
- `logs/cleaning.log`, `logs/links_putusan.json`

>  **Catatan:** Tahap ini memerlukan **interaksi manual** karena situs MA RI menggunakan
> proteksi Cloudflare. Jalankan Chrome dalam mode remote-debugging terlebih dahulu:
> ```bash
> # Windows
> "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome_debug"
> ```
> Buka `https://putusan3.mahkamahagung.go.id`, selesaikan verifikasi Cloudflare secara manual,
> baru jalankan cell-cell notebook. **Data hasil scraping (`data/raw/*.txt`) sudah tersedia
> di repository ini**, sehingga Tahap 2–5 dapat langsung dijalankan tanpa mengulang Tahap 1.

---

### Tahap 2 — Case Representation
**File:** `notebooks/tahap_02_case_representation.ipynb`

**Yang dilakukan:**
- Load semua file `.txt` dari `data/raw/`
- Ekstraksi metadata: nomor perkara, tanggal, jenis perkara, pasal, nama pihak (penggugat & tergugat)
- Ekstraksi konten kunci: ringkasan fakta (duduk perkara), argumen hukum (amar putusan lengkap)
- Feature engineering: `word_count`, `bag-of-words`, label putusan (`0`=Ditolak/NO, `1`=Dikabulkan)
- Filter tambahan: buang dokumen yang teridentifikasi bukan Putusan substantif (total 58 → 54 kasus)
- Simpan ke format terstruktur

**Output:**
- `data/processed/cases.csv` — case base final (kolom: `case_id`, `no_perkara`, `tanggal`, `jenis_perkara`, `pihak`, `pasal`, `ringkasan_fakta`, `argumen_hukum`, `amar_lengkap`, `label_putusan`, `word_count`, `text_full`)
- `data/processed/cases.json`

---

### Tahap 3 — Case Retrieval
**File:** `notebooks/tahap_03_case_retrieval.ipynb`

**Yang dilakukan:**
- Load `cases.csv` hasil Tahap 2
- Vektorisasi seluruh corpus menggunakan `TfidfVectorizer` (unigram+bigram, max 5.000 fitur, `sublinear_tf=True`)
- Split data: **80:20 stratified** berdasarkan `label_putusan`
- Training model SVM (kernel linear, C=1.0) pada representasi TF-IDF
- Implementasi fungsi `retrieve(query, k=5)`:
  1. Pre-process query (lowercase, hapus stopword)
  2. Hitung vektor query (TF-IDF)
  3. Hitung cosine-similarity dengan seluruh case vector
  4. Kembalikan top-k `case_id`
- Evaluasi retrieval: Precision@k, Recall@k, F1@k untuk k=1,3,5,10
- Siapkan 8 query uji + ground-truth

**Output:**
- `data/eval/queries.json` — 8 query uji beserta ground-truth case_id
- `data/eval/retrieval_metrics.csv`

---

### Tahap 4 — Case Solution Reuse
**File:** `notebooks/tahap_04_case_solution_reuse.ipynb`

**Yang dilakukan:**
- Bangun ulang TF-IDF + fungsi `retrieve()` (identik Tahap 3, agar dapat dijalankan mandiri)
- Ekstrak solusi dari setiap kasus: `{case_id: amar_putusan_ringkas}`
- Implementasi dua algoritma prediksi:
  - **Majority Vote**: pilih label yang paling banyak muncul di top-5
  - **Weighted Similarity**: bobot = skor cosine-similarity; label dengan total bobot terbesar menang
- Implementasi fungsi `predict_outcome(query, k=5, metode='weighted')`
- Demo manual 5 contoh kasus baru → bandingkan prediksi vs putusan sebenarnya
- Perbandingan Majority Vote vs Weighted Similarity pada 5 kasus demo

**Output:**
- `data/results/predictions.csv` — kolom: `query_id`, `predicted_solution`, `predicted_label`, `top_5_case_ids`, `similarity_scores`

---

### Tahap 5 — Model Evaluation
**File:** `notebooks/tahap_05_model_evaluation.ipynb`

**Yang dilakukan:**
- Evaluasi retrieval: Accuracy, Precision, Recall, F1 pada k=1,3,5,10 menggunakan `sklearn.metrics`
- Evaluasi klasifikasi SVM pada data test (20%)
- Evaluasi prediksi CBR (Weighted Similarity) pada 5 kasus demo
- **Tabel perbandingan 3 model**: TF-IDF Retrieval (cosine sim) vs SVM vs CBR (Weighted Sim)
- Bar chart visualisasi perbandingan performa
- Error analysis: analisis kasus kegagalan retrieval (MISS) & prediksi salah
- Rekomendasi perbaikan berbasis pola kegagalan

**Output:**
- `data/eval/retrieval_metrics.csv` — metrik retrieval per k
- `data/eval/prediction_metrics.csv` — metrik klasifikasi per model
- Bar chart visualisasi (ditampilkan inline di notebook)

---

### Tahap 6 — Revisi & Retain *(Opsional)*
**File:** `notebooks/Revisi___Retain.ipynb`

**Yang dilakukan:**
- Load `predictions.csv` hasil Tahap 4
- Filter otomatis: tandai kasus yang prediksinya **tepat** (`predicted_label == label_sebenarnya`)
- Override manual: opsi untuk menambah/mengecualikan kasus tertentu dari daftar retain
- Ambil metadata kasus sumber dari `top_5_case_ids`
- Bangun baris kasus baru dengan `case_id` melanjutkan urutan terakhir di case base
- Gabungkan ke `cases.csv` dan `cases.json` untuk dipakai pada iterasi CBR berikutnya
- Verifikasi akhir ukuran case base setelah retain

**Output:**
- `data/processed/cases.csv` — diperbarui (case base bertambah)
- `data/processed/cases.json` — diperbarui

---

## Contoh Penggunaan Fungsi Retrieve & Predict

```python
# Setelah menjalankan Tahap 3 (kernel aktif), panggil langsung:

hasil = retrieve("tergugat tidak menyelesaikan renovasi rumah sesuai perjanjian kerja", k=5)
print(hasil)
# Output: ['case_012', 'case_027', 'case_003', 'case_041', 'case_019']

# Setelah menjalankan Tahap 4:
prediksi = predict_outcome("kontraktor meninggalkan pekerjaan sebelum selesai", k=5, metode='weighted')
print(prediksi)
# Output: 'Dikabulkan'
```

---

## Dependensi

Lihat `requirements.txt`. Install dengan:

```bash
pip install -r requirements.txt
```

| Library | Kegunaan |
|---|---|
| `selenium` | Scraping halaman MA RI (Tahap 1) |
| `beautifulsoup4` | Parsing HTML indeks putusan |
| `pdfminer.six` | Ekstraksi teks dari PDF |
| `pandas` | Manipulasi data tabular |
| `scikit-learn` | TF-IDF, SVM, metrik evaluasi |
| `tqdm` | Progress bar |
| `matplotlib` | Visualisasi bar chart evaluasi |
| `nltk` | Tokenisasi (fallback) |
| `lxml` | Parser HTML alternatif |
| `requests` | HTTP request tambahan |

---

## Ringkasan Metode

- **Domain perkara:** Perdata Gugatan Wanprestasi, PN Surabaya
- **Volume data:** 80 dokumen diunduh, ~54 dokumen putusan substantif lolos validasi (di atas minimum 30 dokumen)
- **Representasi vektor:** TF-IDF (unigram + bigram), `sklearn.feature_extraction.text.TfidfVectorizer`
- **Model retrieval/klasifikasi:** Cosine similarity (CBR `retrieve()`) dan SVM kernel linear (pembanding)
- **Split data:** 80:20 (stratified)
- **Algoritma reuse:** Majority Vote dan Weighted Similarity (bobot = skor cosine similarity)
- **Evaluasi:** Accuracy, Precision, Recall, F1-score (`sklearn.metrics`), dievaluasi pada Precision@k/Recall@k untuk retrieval dan classification metrics untuk prediksi label

## Keterbatasan

- Ekstraksi metadata (`no_perkara`, `pasal`, `pihak`, label putusan) menggunakan pendekatan **regex
  berbasis pola teks**, bukan NLP/NER terlatih — sehingga sebagian kecil dokumen dengan format penulisan
  putusan yang tidak baku dapat menghasilkan nilai `"Tidak Tertera"`.
- Ground-truth pada `queries.json` (Tahap 3) disusun berbasis kata kunci tematik, bukan anotasi pakar
  hukum independen.
- Label putusan dibatasi pada dua kelas (`Dikabulkan` / `Ditolak-NO`); putusan berupa Akta Perdamaian /
  Penetapan dikeluarkan dari case base karena bukan putusan substantif berkontradiktif.

## Tim & Mata Kuliah

- **Nama Anggota:** Marcela Setiawan (202310370311393) dan Faradita Bilbiana Eka Saputeri (202310370311074)
- **Mata Kuliah:** Penalaran Komputer A — SubCPMK-3, Semester Genap 2025/2026
- **Domain Hukum:** Perdata Gugatan Wanprestasi — PN Surabaya
