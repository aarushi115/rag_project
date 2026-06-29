# ⚡ NeuralRAG — Hybrid Document Q&A Engine

A production-grade RAG (Retrieval-Augmented Generation) system that answers questions grounded strictly in your own documents. No hallucinations, no guessing.

Built with a hybrid retrieval pipeline — combining dense vector search and sparse keyword search, fused via Reciprocal Rank Fusion and re-ranked with FlashRank — before hitting the LLM.

![Python](https://img.shields.io/badge/Python-3.10+-8b5cf6?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-6d28d9?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-7c3aed?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-4f46e5?style=flat-square&logo=streamlit&logoColor=white)

---

## How It Works

```
Your PDFs → BM25 (sparse) ──┐
                              ├── RRF Fusion → FlashRank Rerank → Llama 3.3 → Answer
           FAISS (dense) ────┘
```

1. **Ingest** — PDFs are loaded, chunked, and indexed into both a FAISS vector store (dense semantic search via `BAAI/bge-small-en-v1.5` embeddings) and a BM25 index (sparse keyword search)
2. **Retrieve** — both retrievers run in parallel on your query
3. **Fuse** — Reciprocal Rank Fusion merges and deduplicates results
4. **Rerank** — FlashRank (`ms-marco-MiniLM-L-12-v2`) picks the top 3 most relevant chunks
5. **Generate** — Llama 3.3 70B on Groq answers strictly from retrieved context

---

## Features

- Hybrid BM25 + FAISS retrieval with RRF fusion
- FlashRank neural re-ranking
- Grounded answers — refuses to answer outside provided context
- Conversation history with follow-up questions
- Latency + source chunk inspection per answer
- Fully free-tier stack (Groq, HuggingFace, local FAISS)

---

## Stack

| Component | Tool |
|---|---|
| LLM | Llama 3.3 70B via Groq |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace) |
| Vector Store | FAISS (local) |
| Sparse Retrieval | BM25 |
| Reranker | FlashRank `ms-marco-MiniLM-L-12-v2` |
| Framework | LangChain Core + LangChain Groq |
| UI | Streamlit |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/neuralrag.git
cd neuralrag
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Add your PDFs
Drop any PDF files into the `data_source/` folder.

### 6. Run ingestion
```bash
python ingest.py
```
This builds the FAISS and BM25 indexes in `vector_store/`.

### 7. Launch the app
```bash
streamlit run app.py
```

---

## Project Structure

```
neuralrag/
├── app.py              # Streamlit UI + RAG pipeline
├── ingest.py           # PDF ingestion + index building
├── data_source/        # Put your PDFs here
├── vector_store/       # Auto-generated indexes (gitignored)
│   ├── faiss_index/
│   └── bm25_retriever.pkl
├── .env                # Your API keys (gitignored)
├── requirements.txt
└── README.md
```

---

## Requirements

```
streamlit
langchain-core
langchain-groq
langchain-huggingface
langchain-community
langchain-text-splitters
faiss-cpu
flashrank
python-dotenv
sentence-transformers
pypdf
rank-bm25
```

---

## Notes

- `vector_store/` and `.env` are gitignored — never commit your API keys or indexes
- First run will download the embedding model (~130MB) and FlashRank model (~22MB)
- Works entirely on free-tier services — no OpenAI, no paid vector DB
