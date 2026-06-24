"""
DocuRAG — Streamlit frontend.

Connects to the FastAPI backend running at API_BASE_URL (default: http://localhost:8000).
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API = f"{API_BASE}/api/v1"

st.set_page_config(
    page_title="DocuRAG",
    page_icon="📄",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    top_k = st.slider("Retrieval top-k", min_value=1, max_value=15, value=5)
    filter_doc = st.text_input(
        "Filter by document (optional)",
        placeholder="e.g. Attention",
        help="Leave empty to search all uploaded documents.",
    )

    st.divider()

    st.subheader("🔍 Backend status")
    try:
        resp = requests.get(f"{API}/health", timeout=3)
        if resp.ok:
            st.success("API online ✅")
        else:
            st.error(f"API returned {resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API — is the server running?")

    st.divider()
    st.caption("DocuRAG v1.0.0 · [GitHub](https://github.com)")

# ── Main area ──────────────────────────────────────────────────────────────────
st.title("📄 DocuRAG")
st.markdown(
    "Upload a PDF and ask questions. Answers include citations to the exact "
    "document and page that support each response."
)

tab_upload, tab_query, tab_eval = st.tabs(["📤 Upload", "💬 Query", "📊 Evaluate"])

# ── Upload tab ─────────────────────────────────────────────────────────────────
with tab_upload:
    st.header("Upload a PDF document")
    uploaded = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        label = "Ingest document" if len(uploaded) == 1 else f"Ingest {len(uploaded)} documents"
        if st.button(label, type="primary"):
            with st.spinner("Uploading and processing…"):
                try:
                    if len(uploaded) == 1:
                        f = uploaded[0]
                        resp = requests.post(
                            f"{API}/upload",
                            files={"file": (f.name, f.getvalue(), "application/pdf")},
                            timeout=120,
                        )
                        if resp.ok:
                            d = resp.json()
                            st.success(f"✅ **{d['doc_name']}** — {d['pages_loaded']} pages, {d['chunks_created']} chunks.")
                        else:
                            st.error(f"Upload failed ({resp.status_code}): {resp.text}")
                    else:
                        files = [
                            ("files", (f.name, f.getvalue(), "application/pdf"))
                            for f in uploaded
                        ]
                        resp = requests.post(f"{API}/upload/batch", files=files, timeout=300)
                        if resp.ok:
                            data = resp.json()
                            st.success(data["message"])
                            for r in data["results"]:
                                st.markdown(f"- ✅ **{r['doc_name']}** — {r['pages_loaded']} pages, {r['chunks_created']} chunks")
                            for err in data["errors"]:
                                st.warning(f"- ⚠️ {err}")
                        else:
                            st.error(f"Batch upload failed ({resp.status_code}): {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API server. Make sure it is running.")

# ── Query tab ──────────────────────────────────────────────────────────────────
with tab_query:
    st.header("Ask a question")
    query = st.text_area(
        "Your question",
        height=100,
        placeholder="e.g. What is the attention mechanism in Transformers?",
    )

    if st.button("Ask", type="primary", disabled=not query.strip()):
        with st.spinner("Retrieving and generating answer…"):
            payload = {
                "query": query,
                "k": top_k,
                "filter_doc_name": filter_doc.strip() or None,
            }
            try:
                resp = requests.post(f"{API}/query", json=payload, timeout=60)
                if resp.ok:
                    data = resp.json()

                    st.subheader("Answer")
                    st.markdown(data["answer"])

                    st.divider()
                    col1, col2 = st.columns(2)
                    col1.metric("Response time", f"{data['latency_seconds']:.2f}s")
                    col2.metric("Chunks retrieved", data["retrieved_chunk_count"])

                    if data["citations"]:
                        st.subheader("📚 Sources")
                        seen = set()
                        for c in data["citations"]:
                            key = (c["doc_name"], c["page_label"])
                            if key not in seen:
                                st.markdown(
                                    f"- **{c['doc_name']}** — page {c['page_label']}"
                                )
                                seen.add(key)
                elif resp.status_code == 400:
                    st.warning(resp.json().get("detail", "No documents ingested yet."))
                else:
                    st.error(f"Query failed ({resp.status_code}): {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API server.")

# ── Evaluation tab ─────────────────────────────────────────────────────────────
with tab_eval:
    st.header("Evaluate the pipeline")
    st.markdown(
        "Enter one question per line. Optionally provide reference answers "
        "(one per line, same order) for relevance scoring."
    )

    eval_queries_raw = st.text_area(
        "Test queries (one per line)",
        height=150,
        placeholder="What is attention?\nWhat are embeddings?",
    )
    eval_refs_raw = st.text_area(
        "Reference answers — optional (one per line, matching order above)",
        height=150,
        placeholder="Attention maps query-key-value pairs…\nEmbeddings are dense vector representations…",
    )

    if st.button("Run evaluation", type="primary", disabled=not eval_queries_raw.strip()):
        queries = [q.strip() for q in eval_queries_raw.strip().splitlines() if q.strip()]
        refs = [r.strip() for r in eval_refs_raw.strip().splitlines() if r.strip()]
        refs = refs if len(refs) == len(queries) else None

        with st.spinner(f"Running {len(queries)} queries…"):
            payload = {"queries": queries, "k": top_k}
            if refs:
                payload["reference_answers"] = refs

            try:
                resp = requests.post(f"{API}/evaluate", json=payload, timeout=300)
                if resp.ok:
                    s = resp.json()
                    st.subheader("📊 Results")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Queries", s["num_queries"])
                    col2.metric("Avg latency", f"{s['avg_latency_seconds']:.2f}s")
                    col3.metric("Retrieval relevance", f"{s['avg_retrieval_relevance']:.2%}")
                    col4.metric("Faithfulness", f"{s['avg_faithfulness']:.2%}")
                    col5.metric("Hallucination risk", f"{s['avg_hallucination_risk']:.2%}")

                    st.info(
                        "Metrics use lexical overlap as a proxy. For production use, "
                        "consider LLM-judge or embedding-based evaluations."
                    )
                else:
                    st.error(f"Evaluation failed ({resp.status_code}): {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API server.")
