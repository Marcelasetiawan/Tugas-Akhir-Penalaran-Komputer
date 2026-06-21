# CBR Prediksi Putusan Wanprestasi — Pengadilan Negeri Surabaya

Sistem **Case-Based Reasoning (CBR)** sederhana berbasis Python untuk memprediksi kemungkinan amar
putusan (Dikabulkan / Ditolak-NO) pada perkara perdata gugatan **wanprestasi** di **Pengadilan Negeri
Surabaya**, menggunakan data putusan yang dipublikasikan di [Direktori Putusan Mahkamah Agung RI](https://putusan3.mahkamahagung.go.id).

Disusun untuk memenuhi tugas **SubCPMK-3 — Mata Kuliah Penalaran Komputer** (siklus CBR: Membangun Case
Base → Case Representation → Case Retrieval → Case/Solution Reuse → Revisi & Retain (opsional) →
Evaluasi Model).

## Siklus CBR yang Diimplementasikan

```
┌─────────────────┐   ┌──────────────────┐   ┌────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│ 1. Case Base     │ → │ 2. Representation│ → │ 3. Retrieval    │ → │ 4. Solution Reuse   │ → │ 5. Evaluation     │
│ scraping + clean │   │ metadata + fitur │   │ TF-IDF + SVM    │   │ majority/weighted   │   │ Accuracy/Prec/    │
│                  │   │                  │   │ cosine retrieve │   │ vote -> prediksi    │   │ Recall/F1         │
└─────────────────┘   └──────────────────┘   └────────────────┘   └────────────────────┘   └──────────────────┘
                                                                                                       │
                                                                          ┌────────────────────────────┘
                                                                          ▼
                                                              ┌──────────────────────┐
                                                              │ 6. Revisi & Retain    │  (opsional, iteratif)
                                                              │ kasus baru tervalidasi│
                                                              │ -> masuk case base    │
                                                              └──────────────────────┘
```

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
│   └── 06_retain_optional.ipynb           Tahap 6 (opsional) — revisi & retain
├── data/
│   ├── raw/
│   │   ├── pdf/                       <- PDF asli hasil download dari MA RI
│   │   └── case_001.txt ... case_NNN.txt  <- teks bersih hasil cleaning (Tahap 1)
│   ├── processed/
│   │   ├── cases.csv                  <- case base terstruktur (Tahap 2)
│   │   └── cases.json
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
> masalah saat upload ke GitHub, folder ini boleh diabaikan (`.gitignore`) — hasil scraping yang penting
> untuk dinilai adalah `data/raw/*.txt` dan `data/processed/cases.csv`, yang jauh lebih kecil.

## Instalasi

### 1. Clone repository

```bash
git clone <URL_REPOSITORY_ANDA>
cd <NAMA_REPOSITORY>
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
| 6 | `06_retain_optional.ipynb` *(opsional)* | `cases.csv`, `predictions.csv` | `cases.csv` (diperbarui), `logs/retain_log.json` | Menambah kasus baru yang prediksinya terbukti tepat ke case base |

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
