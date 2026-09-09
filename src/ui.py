import streamlit as st

from .rag import answer_question


def render_processing_status(store):
    stats = store.get("statistics", {})
    if not stats:
        return
    with st.expander("Document processing summary", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Text elements", stats.get("text_elements", "—"))
        cols[1].metric("Tables", stats.get("table_elements", "—"))
        cols[2].metric("Visuals", stats.get("visual_elements", "—"))
        cols[3].metric("Chunks", len(store.get("chunks", [])))


def render_sources(results):
    if not results:
        return

    st.subheader("Retrieved Sources")
    for result in results:
        chunk = result["chunk"]
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        page_text = (
            f"Page {page_start}" if page_start == page_end
            else f"Pages {page_start}–{page_end}"
        )
        with st.expander(
            f"Source {result['rank']} • {page_text} • relevance {result['score']:.4f}"
        ):
            st.caption(
                f"Chunk: {chunk.get('chunk_id')} • "
                f"Section: {chunk.get('section') or '—'}"
            )
            st.write(chunk.get("content", ""))


def render_chat(key_prefix, history_key, store, title):
    st.subheader(title)

    for message in st.session_state[history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input(
        "Ask a question about the document…",
        key=f"{key_prefix}_chat_input",
    )

    if question:
        question = question.strip()
        if not question:
            st.warning("Please enter a question.")
            return

        st.session_state[history_key].append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document and generating an answer…"):
                try:
                    answer, model_used, results = answer_question(store, question)
                    st.markdown(answer)
                    st.caption(f"Answer model: {model_used}")
                    render_sources(results)
                    st.session_state[history_key].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": results,
                        }
                    )
                except Exception as exc:
                    st.error(f"Could not answer the question: {exc}")
                    st.session_state[history_key].append(
                        {"role": "assistant", "content": f"Error: {exc}"}
                    )
