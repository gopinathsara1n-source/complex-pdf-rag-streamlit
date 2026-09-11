import os, sys, json, tempfile, hashlib
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from pipeline import pdf_hash, process_pdf, load_index, retrieve, answer_question

st.set_page_config(page_title="Complex PDF RAG", page_icon="📚", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 1.5rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.22); border-radius: 18px;
       background: linear-gradient(135deg, rgba(90,90,180,.10), rgba(60,160,160,.08));}
.hero h1 {margin-bottom: .2rem;}
.source-card {padding: .75rem 1rem; border: 1px solid rgba(128,128,128,.22);
              border-radius: 12px; margin: .5rem 0;}
.small {opacity:.72; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

ROOT = APP_DIR.parent
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
EXAMPLE = ROOT / "data" / "example"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_dir" not in st.session_state:
    st.session_state.active_dir = None
if "active_name" not in st.session_state:
    st.session_state.active_name = None

def cache_dir(digest):
    p = CACHE / digest
    p.mkdir(parents=True, exist_ok=True)
    return p

def ready(d):
    return (d/"chunks.json").exists() and (d/"faiss.index").exists()

# ---------- sidebar ----------
with st.sidebar:
    st.header("📚 Complex PDF RAG")
    st.caption("Docling + Gemini Vision + Gemini Embedding 2 + FAISS")

    st.divider()
    st.subheader("Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    example_files = list(EXAMPLE.glob("*.pdf"))
    if example_files:
        st.caption("Example PDF")
        if st.button("Load example PDF", use_container_width=True):
            data = example_files[0].read_bytes()
            digest = pdf_hash(data)
            d = cache_dir(digest)
            pdf_path = d / example_files[0].name
            pdf_path.write_bytes(data)
            st.session_state.active_dir = str(d)
            st.session_state.active_name = example_files[0].name
            st.session_state.messages = []
            st.rerun()

    if uploaded is not None:
        data = uploaded.getvalue()
        digest = pdf_hash(data)
        d = cache_dir(digest)
        pdf_path = d / uploaded.name
        pdf_path.write_bytes(data)
        st.session_state.active_dir = str(d)
        st.session_state.active_name = uploaded.name

        if not ready(d):
            if st.button("⚙️ Process PDF", type="primary", use_container_width=True):
                bar = st.progress(0)
                status = st.empty()
                def cb(msg, frac):
                    status.info(msg)
                    bar.progress(min(1.0, max(0.0, frac)))
                try:
                    process_pdf(pdf_path, d, cb)
                    st.success("PDF processed successfully.")
                    st.session_state.messages = []
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        else:
            st.success("Processed PDF found in cache.")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Embeddings are checkpointed so interrupted processing can resume.")

# ---------- main ----------
st.markdown("""
<div class="hero">
<h1>📖 Complex PDF Assistant</h1>
<p>Ask questions about paragraphs, tables, charts and visual information in your PDF.</p>
</div>
""", unsafe_allow_html=True)

active_dir = Path(st.session_state.active_dir) if st.session_state.active_dir else None

if not active_dir or not ready(active_dir):
    st.info("Upload a PDF from the sidebar, then choose **Process PDF**. An example PDF can also be placed in `data/example/`.")
    st.markdown("### Pipeline")
    st.markdown("**PDF → Docling → Text/Tables/Pictures → Gemini Vision → Structure-aware chunks → Gemini Embedding 2 → FAISS → Chatbot**")
    st.stop()

try:
    chunks, index, client = load_index(active_dir)
except Exception as e:
    st.error(f"Could not load the processed document: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Document", st.session_state.active_name or "PDF")
c2.metric("RAG chunks", len(chunks))
c3.metric("Embedding", "Gemini 2 · 768D")

st.divider()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(
                        f'<div class="source-card"><b>Page {s.get("page")}</b> · '
                        f'{s.get("type")} · {s.get("chunk_id")} · score {s.get("_score",0):.3f}<br>'
                        f'<span class="small">{s.get("text","")[:700]}</span></div>',
                        unsafe_allow_html=True
                    )

question = st.chat_input("Ask anything about the document…")
if question:
    st.session_state.messages.append({"role":"user", "content":question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and generating an answer…"):
            try:
                hits = retrieve(question, chunks, index, client, top_k=6)
                answer = answer_question(client, question, hits, st.session_state.messages[:-1])
                st.markdown(answer)
                st.session_state.messages.append({
                    "role":"assistant", "content":answer, "sources":hits
                })
                with st.expander("Sources"):
                    for s in hits:
                        st.markdown(
                            f'<div class="source-card"><b>Page {s.get("page")}</b> · '
                            f'{s.get("type")} · {s.get("chunk_id")} · score {s.get("_score",0):.3f}<br>'
                            f'<span class="small">{s.get("text","")[:700]}</span></div>',
                            unsafe_allow_html=True
                        )
            except Exception as e:
                st.error(str(e))
