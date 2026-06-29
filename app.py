import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv
from typing import List
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

load_dotenv()

FAISS_INDEX_PATH = "vector_store/faiss_index"
BM25_INDEX_PATH  = "vector_store/bm25_retriever.pkl"

st.set_page_config(page_title="NeuralRAG", page_icon="⚡", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #1a0533 0%, #0a0a0f 50%, #000d1a 100%) !important;
}
[data-testid="stHeader"]  { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

.block-container {
    padding: 3rem 2rem 10rem 2rem !important;
    max-width: 780px !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem 0;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed22, #2563eb22);
    border: 1px solid #7c3aed55;
    color: #a78bfa;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
}
.hero h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #e2d9f3 0%, #a78bfa 40%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2 !important;
    margin: 0 0 0.7rem 0 !important;
    letter-spacing: -0.02em;
}
.hero p {
    color: #475569;
    font-size: 0.92rem;
    margin: 0;
}

/* ── Search input ── */
[data-testid="stTextInput"] label { display: none !important; }
[data-testid="stTextInput"] input {
    background: #ffffff0a !important;
    border: 1.5px solid #ffffff18 !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 0.9rem 1.2rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7c3aed99 !important;
    box-shadow: 0 0 0 3px #7c3aed22 !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #334155 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #ffffff06;
    border: 1px solid #ffffff10;
    border-radius: 12px;
    padding: 0.9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Query bubble ── */
.query-bubble {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e2e8f0;
    line-height: 1.35;
    margin: 2rem 0 1.4rem 0;
    letter-spacing: -0.02em;
}

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, #ffffff08 0%, #7c3aed08 100%);
    border: 1px solid #7c3aed30;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #2563eb, #7c3aed);
}
.answer-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c3aed;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.8rem;
}
.answer-text {
    color: #cbd5e1;
    font-size: 0.97rem;
    line-height: 1.8;
}

/* ── Divider ── */
.turn-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #ffffff10, transparent);
    margin: 2rem 0;
}

/* ── Expander / sources ── */
[data-testid="stExpander"] {
    background: #ffffff04 !important;
    border: 1px solid #ffffff0e !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem;
}
[data-testid="stExpander"] summary {
    color: #475569 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover { color: #64748b !important; }

/* ── Source chunks ── */
.chunk-card {
    background: #ffffff04;
    border: 1px solid #ffffff08;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
.chunk-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.rank-pill {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
}
.source-tag {
    color: #334155;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
}
.chunk-text {
    color: #64748b;
    font-size: 0.85rem;
    line-height: 1.65;
}

/* ── Fixed follow-up bar ── */
.followup-outer {
    position: fixed;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    max-width: 780px;
    padding: 0 2rem 1.8rem 2rem;
    background: linear-gradient(to top, #0a0a0f 65%, transparent);
    z-index: 999;
}
.followup-label {
    font-size: 0.68rem;
    color: #1e293b;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
    padding-left: 0.2rem;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #475569 !important;
    font-size: 0.85rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #ffffff12; border-radius: 999px; }
</style>
""", unsafe_allow_html=True)


# ── Logic ─────────────────────────────────────────────────────────────────────
def reciprocal_rank_fusion(results_lists: List[List[Document]], k: int = 60) -> List[Document]:
    scores: dict = defaultdict(float)
    docs_map: dict = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            scores[key] += 1.0 / (k + rank + 1)
            docs_map[key] = doc
    return [docs_map[k] for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def flashrank_rerank(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    try:
        from flashrank import Ranker, RerankRequest
        ranker   = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        results  = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[r["id"]] for r in results[:top_n]]
    except Exception:
        return docs[:top_n]


@st.cache_resource
def load_retrievers():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(BM25_INDEX_PATH):
        return None, None
    embeddings      = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    faiss_db        = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 5})
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 5
    return faiss_retriever, bm25_retriever


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("🔑 Missing GROQ_API_KEY in .env")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.1)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer using ONLY the context below.
If the context doesn't contain the answer, say so clearly. Be concise and precise.

<context>{context}</context>

Question: {question}
Answer:"""
)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

def run_rag(query: str, faiss_retriever, bm25_retriever) -> dict:
    dense_docs  = faiss_retriever.invoke(query)
    sparse_docs = bm25_retriever.invoke(query)
    fused       = reciprocal_rank_fusion([sparse_docs, dense_docs])
    final_docs  = flashrank_rerank(query, fused, top_n=3)
    context     = format_docs(final_docs)
    t0          = time.perf_counter()
    answer      = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": query})
    latency     = time.perf_counter() - t0
    return {"answer": answer, "context": final_docs, "latency": latency}


# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

faiss_retriever, bm25_retriever = load_retrievers()

if faiss_retriever is None:
    st.markdown("""
    <div class='hero'>
        <div class='hero-badge'>⚡ NeuralRAG</div>
        <h1>Index not found</h1>
        <p>Add PDFs to <code>./data_source/</code> and run <code>python ingest.py</code></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── EMPTY STATE ────────────────────────────────────────────────────────────────
if not st.session_state.history:
    st.markdown("""
    <div class='hero'>
        <div class='hero-badge'>⚡ NeuralRAG</div>
        <h1>Ask your documents</h1>
        <p>Hybrid retrieval · Re-ranking · Grounded answers</p>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input("q", placeholder="Ask anything about your documents...", key=f"input_{st.session_state.input_key}", label_visibility="collapsed")

    if query:
        with st.spinner("Searching..."):
            result = run_rag(query, faiss_retriever, bm25_retriever)
        st.session_state.history.append({**result, "query": query})
        st.session_state.input_key += 1
        st.rerun()


# ── CONVERSATION STATE ─────────────────────────────────────────────────────────
else:
    for i, turn in enumerate(st.session_state.history):
        if i > 0:
            st.markdown("<div class='turn-divider'></div>", unsafe_allow_html=True)

        # Query heading
        st.markdown(f"<div class='query-bubble'>{turn['query']}</div>", unsafe_allow_html=True)

        # Metrics row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("⏱ Latency", f"{turn['latency']:.2f}s")
        with c2:
            st.metric("📄 Chunks", len(turn["context"]))
        with c3:
            st.metric("🧠 Model", "Llama 3.3 70B")

        # Answer
        st.markdown(f"""
        <div class='answer-card'>
            <div class='answer-label'>✦ Answer</div>
            <div class='answer-text'>{turn['answer']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Sources
        with st.expander(f"🔍  {len(turn['context'])} sources"):
            for rank, doc in enumerate(turn["context"], start=1):
                source = os.path.basename(doc.metadata.get("source", "Document"))
                st.markdown(f"""
                <div class='chunk-card'>
                    <div class='chunk-header'>
                        <span class='rank-pill'>#{rank}</span>
                        <span class='source-tag'>📄 {source}</span>
                    </div>
                    <div class='chunk-text'>{doc.page_content[:300]}{'…' if len(doc.page_content) > 300 else ''}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Fixed follow-up bar ────────────────────────────────────────────────────
    st.markdown("<div class='followup-outer'>", unsafe_allow_html=True)
    st.markdown("<div class='followup-label'>Follow-up</div>", unsafe_allow_html=True)

    followup = st.text_input(
        "followup",
        placeholder="Ask a follow-up question...",
        key=f"input_{st.session_state.input_key}",
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if followup:
        with st.spinner("Searching..."):
            result = run_rag(followup, faiss_retriever, bm25_retriever)
        st.session_state.history.append({**result, "query": followup})
        st.session_state.input_key += 1
        st.rerun()