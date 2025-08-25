import io
import os
import json
import hashlib
import numpy as np
import requests
import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"   # small & fast
EMBED_BATCH = 64
TOP_K = 3

# ---------- Caching ----------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    import torch, platform
    st.write(f"🔧 Torch {torch.__version__} • Python {platform.python_version()}")
    # Force CPU to avoid Metal/MPS path
    os.environ["PYTORCH_MPS_DISABLE"] = "1"
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    st.write("✅ Using CPU for embeddings")
    return model

@st.cache_data(show_spinner=False)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Robust text extraction with page markers; skips None pages."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        if txt.strip():
            out.append(f"[Page {i}] {txt}")
    return "\n".join(out)

def chunk_text(text: str, chunk_size: int = 500):
    """Simple word-based chunking by approx char length."""
    words, chunks, cur, sz = text.split(), [], [], 0
    for w in words:
        cur.append(w); sz += len(w) + 1
        if sz >= chunk_size:
            chunks.append(" ".join(cur)); cur, sz = [], 0
    if cur: chunks.append(" ".join(cur))
    return chunks

@st.cache_data(show_spinner=False)
def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Embed and return normalized float32 numpy array."""
    model = load_embedding_model()
    embs = model.encode(
        chunks,
        batch_size=EMBED_BATCH,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")
    return embs

@st.cache_data(show_spinner=False)
def build_index(embs: np.ndarray):
    """Build cosine-similarity FAISS index (dot product on normalized vectors)."""
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    # embeddings are already normalized above
    index.add(embs)
    return index

def query_ollama(prompt: str) -> str:
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("response", "")
        return "Error: Could not connect to Ollama. Make sure Ollama is running with 'ollama serve'."
    except requests.exceptions.RequestException:
        return "Error: Ollama is not running. Please start Ollama with 'ollama serve'."

def retrieve(query: str, index, chunks: list[str], k: int = TOP_K, embs_model=None):
    if index is None or not chunks: return []
    if embs_model is None:
        embs_model = load_embedding_model()
    q = embs_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q, k)
    results = []
    for rank, idx in enumerate(I[0]):
        if 0 <= idx < len(chunks):
            results.append({"text": chunks[idx], "score": float(D[0][rank])})
    return results

# ---------- UI ----------
st.title("🤖 Simple RAG Chatbot")
st.write("Upload a PDF and ask questions about its content!")

# Session state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("index", None)
st.session_state.setdefault("chunks", [])

with st.sidebar:
    st.header("📄 Upload Document")
    up = st.file_uploader("Choose a PDF file", type="pdf")

    if up is not None:
        # Use bytes as the cache key so re-uploads are instant
        file_bytes = up.getvalue()
        with st.status("Processing document…", expanded=True) as status:
            st.write("📥 Extracting text…")
            text = extract_text_from_pdf_bytes(file_bytes)

            if not text.strip():
                status.update(label="❗ No extractable text found (is this a scanned PDF?).", state="error")
            else:
                st.write("✂️ Chunking…")
                chunks = chunk_text(text)

                st.write("🔡 Loading embedding model (cached)…")
                _ = load_embedding_model()  # warm cache

                st.write("🧮 Embedding chunks…")
                embs = embed_chunks(chunks)

                st.write("📚 Building FAISS index…")
                index = build_index(embs)

                st.session_state.index = index
                st.session_state.chunks = chunks
                status.update(label=f"✅ Ready! {len(chunks)} chunks indexed.", state="complete")

# Chat area
if st.session_state.index is not None:
    # history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if "sources" in m:
                with st.expander("📚 Sources"):
                    for i, s in enumerate(m["sources"]):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(s["text"][:200] + "…")
                        st.write(f"*Relevance: {s['score']:.3f}*")

    if prompt := st.chat_input("Ask a question about your document…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        rel = retrieve(prompt, st.session_state.index, st.session_state.chunks)

        context = "\n\n".join(r["text"] for r in rel)
        rag_prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say "I don't have information about that in the provided document."

Context:
{context}

Question: {prompt}

Answer:"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = query_ollama(rag_prompt)
                st.write(resp)
                if rel:
                    with st.expander("📚 Sources"):
                        for i, s in enumerate(rel):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(s["text"][:200] + "…")
                            st.write(f"*Relevance: {s['score']:.3f}*")

        st.session_state.messages.append({"role": "assistant", "content": resp, "sources": rel})
else:
    st.info("👆 Upload a PDF to get started! (Text-only PDFs work best; scanned PDFs may be empty.)")
