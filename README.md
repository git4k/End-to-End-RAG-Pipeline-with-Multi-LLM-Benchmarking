# End-to-End-RAG-Pipeline-with-Multi-LLM-Benchmarking
# End-to-End RAG Pipeline with Multi-LLM Benchmarking

> A complete **Retrieval-Augmented Generation (RAG)** system for English
> extractive Question Answering — benchmarking **LLaMA-3.8B, Qwen2.5-7B &
> Mistral-7B** with FAISS dense retrieval vs. No-RAG baselines across
> 500 queries.

---

## Overview

This project processes **50 English documents** into **449 overlapping chunks**,
builds a **FAISS vector index** using `sentence-transformers/all-MiniLM-L6-v2`
embeddings, and generates short extractive answers (≤5 words) for **500 queries**
using locally hosted LLMs via **Ollama**. Both RAG and No-RAG modes are rigorously
evaluated using exact match, substring match, and semantic similarity metrics.

---

## Results

### Answerable Queries (350 questions)

| Metric | LLaMA-3.8B | Qwen2.5-7B | Mistral-7B |
|---|---|---|---|
| Exact Match — RAG | 0.540 | **0.609** | 0.506 |
| Exact Match — No-RAG | 0.089 | 0.211 | 0.151 |
| Substring Match — RAG | 0.771 | **0.774** | 0.700 |
| Semantic Match ≥0.75 — RAG | 0.731 | **0.786** | 0.720 |
| Semantic Mean Score — RAG | 0.815 | **0.850** | 0.823 |

### Unanswerable Queries — NA Detection (150 questions)

| Model | RAG | No-RAG |
|---|---|---|
| LLaMA-3.8B | 0.820 | 0.553 |
| Qwen2.5-7B | **0.873** | **0.940** |
| Mistral-7B | 0.080 | 0.000 |

> ✅ **Qwen2.5-7B + RAG** is the top-performing model across all metrics.

---

## 🛠 Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector Store | FAISS `IndexFlatIP` (cosine similarity) |
| LLMs | LLaMA-3.8B · Qwen2.5-7B · Mistral-7B via Ollama |
| Language | Python 3.11 |
| Libraries | Pandas · NumPy · scikit-learn · tqdm · transformers |

---

## Pipeline Architecture

```
Raw Documents (50 English docs)
        │
        ▼
  Unicode Cleaning + Evidence Validation
        │
        ▼
  Sliding Window Chunking (150w, 30w overlap)
        │  → 449 total chunks
        ▼
  MiniLM Embeddings → FAISS Index
        │
        ▼
  Query → Top-3 Chunk Retrieval (cosine sim)
        │
        ▼
  Strict RAG Prompt → Ollama LLM → ≤5 word answer / NA
        │
        ▼
  Evaluation: Exact / Substring / Semantic Match
        │
        ▼
  EVALUATION_ALL_MODELS.csv
```

---

## Project Structure

```
├── phase1_english_pipeline.ipynb    # Main pipeline notebook
├── E2/
│   ├── documents.csv                # 50 source documents
│   ├── queries.csv                  # 500 questions
│   ├── answers.csv                  # Gold answers + evidence spans
│   ├── results.csv                  # LLaMA RAG results
│   ├── resultsfull.csv              # All model outputs
│   └── EVALUATION_ALL_MODELS.csv    # Master evaluation table
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install sentence-transformers faiss-cpu pandas numpy scikit-learn tqdm transformers

# 2. Pull Ollama models
ollama pull llama3:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b

# 3. Run the notebook
jupyter notebook phase1_english_pipeline.ipynb
```

---

## 📈 Key Findings

- RAG improves exact match by **3–6×** over No-RAG across all three models
- **Qwen2.5-7B** is the best model: 60.9% exact match + 87.3% correct NA detection
- **Mistral-7B** nearly fails on unanswerable queries (only 8% NA detection with RAG)
- All RAG models reach semantic mean scores of **0.82–0.85**, confirming
  high answer quality even when exact match falls short

---

## 📄 License

MIT
