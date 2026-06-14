# Final Project KSSC: Analisis Saturasi & Halusinasi pada RAG

Project ini merupakan final assignment mata kuliah Kapita Selekta SC. Project ini membangun sistem **Retrieval-Augmented Generation (RAG)** menggunakan dataset **IDK-MRC** untuk menganalisis pengaruh jumlah dokumen konteks terhadap kualitas jawaban dan akurasi sitasi.

Eksperimen dilakukan dengan variasi jumlah dokumen hasil retrieval atau **Top-K**, yaitu:

* K = 1
* K = 3
* K = 5
* K = 10

Tujuan utama project ini adalah melihat apakah performa RAG mengalami **saturasi** ketika jumlah konteks bertambah, serta menganalisis kemungkinan **halusinasi sitasi** dari model generator.

---

## Dataset

Dataset yang digunakan adalah:

* **IDK-MRC** dari HuggingFace
* Dataset berisi pasangan konteks, pertanyaan, dan jawaban dalam bahasa Indonesia
* Struktur data utama yang digunakan:

  * `context`
  * `qas`
  * `question`
  * `answers`
  * `is_impossible`

Pada tahap preprocessing, data `qas` di-flatten menjadi dataframe berisi pasangan question-answer. Pertanyaan juga dipisahkan menjadi:

* **Answerable questions**
* **Unanswerable questions**

Eksperimen generator dilakukan pada 50 sampel acak dari data answerable dengan `random_state=42` agar hasil sampling dapat direproduksi.

---

## Arsitektur Sistem

Sistem RAG pada project ini terdiri dari empat komponen utama:

1. **Dataset & Preprocessing**

   * Load dataset IDK-MRC
   * Flatten struktur `qas`
   * Membuat dataframe question-answer
   * Membuat corpus dokumen dari kolom `context`

2. **Embedding**

   * Model embedding yang digunakan:

     * `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   * Setiap dokumen diubah menjadi representasi vektor

3. **Retriever**

   * Retriever menggunakan **FAISS**
   * Similarity search menggunakan inner product pada normalized embedding
   * Dokumen diambil berdasarkan variasi Top-K: 1, 3, 5, dan 10

4. **Generator**

   * Model generator yang digunakan:

     * `Qwen/Qwen2.5-1.5B-Instruct`
   * Prompt dirancang agar model menjawab berdasarkan konteks yang diberikan
   * Model diwajibkan memberikan sitasi dalam format `[1]`, `[2]`, dan seterusnya

---

## Evaluasi

Evaluasi dilakukan pada dua aspek utama:

### 1. Kualitas Jawaban

Kualitas jawaban model dibandingkan dengan ground truth answer menggunakan metrik:

* **ROUGE-1**
* **ROUGE-2**
* **ROUGE-L**
* **BLEU**
* **BERTScore F1**

### 2. Akurasi dan Halusinasi Sitasi

Evaluasi sitasi dilakukan dengan beberapa metrik:

* **Retrieval Hit Rate**
  Mengukur apakah dokumen ground truth muncul dalam hasil retrieval Top-K.

* **Citation Hit Rate**
  Mengukur apakah sitasi yang diberikan model mengarah ke dokumen yang benar.

* **Invalid Citation Count**
  Menghitung jumlah sitasi yang tidak valid atau tidak tersedia dalam konteks.

* **No Citation Rate**
  Mengukur proporsi jawaban yang tidak memiliki sitasi.

* **Citation Hallucination Rate**
  Dihitung sebagai:

  ```text
  citation_hallucination_rate = 1 - citation_hit_rate
  ```

---

## Hasil Eksperimen

Ringkasan hasil evaluasi akhir:

| Top-K | ROUGE-1 | ROUGE-2 | ROUGE-L |    BLEU | BERTScore F1 | Retrieval Hit Rate | Citation Hit Rate | Citation Hallucination Rate |
| ----- | ------: | ------: | ------: | ------: | -----------: | -----------------: | ----------------: | --------------------------: |
| 1     |  0.2901 |  0.1843 |  0.2807 | 16.9753 |       0.6966 |               0.56 |              0.28 |                        0.72 |
| 3     |  0.3416 |  0.2229 |  0.3401 | 17.5177 |       0.7130 |               0.76 |              0.36 |                        0.64 |
| 5     |  0.3390 |  0.2215 |  0.3403 | 17.2756 |       0.7078 |               0.76 |              0.44 |                        0.56 |
| 10    |  0.3349 |  0.2086 |  0.3344 | 23.0064 |       0.7141 |               0.88 |              0.46 |                        0.54 |

Berdasarkan hasil tersebut, retrieval hit rate meningkat ketika Top-K bertambah. Namun, peningkatan jumlah konteks tidak selalu menghasilkan peningkatan kualitas jawaban secara konsisten. Hal ini menunjukkan adanya kemungkinan saturasi performa pada sistem RAG. Citation hit rate juga meningkat dari K=1 ke K=10, tetapi masih terdapat kasus model tidak memberi sitasi atau memberi sitasi yang tidak sesuai.

---

## Struktur Repository

```text
Final-Project-KSSC/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── Final_Assignment_KSSC.ipynb
│
├── outputs/
│   ├── documents_corpus.csv
│   ├── sample_retrieval_results.csv
│   ├── retrieval_hit_summary.csv
│   ├── rag_generation_results.csv
│   ├── final_evaluation_metrics.csv
│   └── figures/
│       ├── metric_vs_k.png
│       ├── retrieval_citation_hit_vs_k.png
│       └── invalid_citation_vs_k.png
│
└── report/
    └── laporan_final_project.pdf
```

---

## Cara Menjalankan Project

### 1. Clone repository

```bash
git clone <repository-url>
cd Final-Project-KSSC
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan notebook

Buka notebook berikut:

```text
notebooks/Final_Assignment_KSSC.ipynb
```

Notebook dapat dijalankan melalui:

* Google Colab
* Jupyter Notebook
* Kaggle Notebook

Disarankan menggunakan GPU untuk menjalankan bagian generator LLM.

---

## Requirements

Library utama yang digunakan:

```text
datasets
sentence-transformers
faiss-cpu
pandas
numpy
tqdm
transformers
accelerate
bitsandbytes
torch
evaluate
rouge-score
bert-score
nltk
matplotlib
```

---

## Output Project

Output utama dari project ini meliputi:

* Notebook eksperimen RAG
* Corpus dokumen hasil preprocessing
* Hasil retrieval Top-K
* Hasil generation dengan variasi K
* Tabel evaluasi akhir
* Grafik performa metrik terhadap Top-K
* Grafik retrieval hit rate dan citation hit rate
* Grafik invalid citation
* Laporan final dalam format PDF

---

## Catatan

Eksperimen generator dilakukan pada subset 50 sampel acak dari data answerable karena proses inference LLM membutuhkan waktu komputasi yang cukup lama. Hasil eksperimen tetap dapat digunakan untuk menganalisis tren pengaruh Top-K terhadap kualitas jawaban dan akurasi sitasi.