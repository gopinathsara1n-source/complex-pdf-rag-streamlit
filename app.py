import streamlit as st
from pathlib import Path
import hashlib
import json
import gc

from rag_pipeline import (
    load_manifest,
    build_or_load_document,
    answer_question,
)

st.set_page_config(
    page_title="Complex PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25);
       border-radius: 18px; margin-bottom: 1.2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>📚 Complex PDF RAG Assistant</h1>
<p>Structure-aware RAG for paragraphs, long tables, charts and figures.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    section = st.radio("Section", ["Example PDF", "Upload & Process"], index=1)
    st.divider()

    api_key = st.text_input(
        "Gemini API key",
        value=st.session_state.get("gemini_api_key", ""),
        type="password",
    )
    if api_key:
        st.session_state.gemini_api_key = api_key

    visual_model = st.text_input(
        "Visual model",
        value=st.session_state.get("visual_model", "gemini-2.5-flash"),
        help="Use a Gemini model available to your API account.",
    )
    answer_model = st.text_input(
        "Answer model",
        value=st.session_state.get("answer_model", visual_model),
    )
    top_k = st.slider("Retrieved chunks", 3, 10, 5)

def render_chat(doc_id: str):
    manifest = load_manifest(doc_id)
    if not manifest:
        st.info("No processed document is loaded.")
        return

    st.success(
        f"Ready: **{manifest.get('source_name', 'document')}** · "
        f"{manifest.get('num_chunks', 0)} chunks · "
        f"{manifest.get('pages', '?')} pages"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"- **Page {s.get('page', '?')}** · "
                            f"score `{s.get('score', 0):.4f}` · "
                            f"{s.get('kind', 'content')}"
                        )

    question = st.chat_input("Ask a question about this PDF…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating answer…"):
                result = answer_question(
                    doc_id,
                    question,
                    st.session_state.get("gemini_api_key", ""),
                    answer_model,
                    top_k,
                )
            st.markdown(result["answer"])
            if result.get("sources"):
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.markdown(
                            f"- **Page {s.get('page', '?')}** · "
                            f"score `{s.get('score', 0):.4f}` · "
                            f"{s.get('kind', 'content')}"
                        )
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })

if section == "Example PDF":
    st.subheader("Example PDF")
    manifest_path = Path("example") / "manifest.json"
    if manifest_path.exists():
        example_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        render_chat(example_manifest["document_id"])
    else:
        st.info(
            "Example slot is ready. Add a disk-backed example index under "
            "`example/` to make it immediately queryable."
        )
else:
    st.subheader("Upload & Process")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded:
        pdf_bytes = uploaded.getvalue()
        document_id = hashlib.sha256(pdf_bytes).hexdigest()
        st.caption(f"Document ID: `{document_id[:16]}`")

        if st.button("🚀 Process PDF", type="primary", use_container_width=True):
            if not st.session_state.get("gemini_api_key"):
                st.error("Enter your Gemini API key in the sidebar first.")
                st.stop()

            progress = st.progress(0)
            status = st.empty()

            def cb(frac, message):
                progress.progress(max(0, min(100, int(frac * 100))))
                status.info(message)

            try:
                manifest = build_or_load_document(
                    pdf_bytes=pdf_bytes,
                    source_name=uploaded.name,
                    document_id=document_id,
                    api_key=st.session_state["gemini_api_key"],
                    visual_model=visual_model,
                    progress_callback=cb,
                )
                st.session_state.doc_id = manifest["document_id"]
                st.session_state.messages = []
                st.success(
                    f"Processed successfully: {manifest['num_chunks']} chunks."
                )
                st.rerun()
            except Exception as exc:
                st.exception(exc)
            finally:
                gc.collect()

    if st.session_state.get("doc_id"):
        render_chat(st.session_state["doc_id"])
