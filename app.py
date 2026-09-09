from pathlib import Path

import streamlit as st

from src.config import APP_TITLE
from src.rag import answer_question, load_example_store, build_uploaded_store
from src.ui import render_chat, render_sources, render_processing_status


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# EXACTLY TWO APPLICATION SECTIONS
# ============================================================

page = st.sidebar.radio(
    "Sections",
    ["Example PDF", "Upload Your Own PDF"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Complex PDF RAG • Docling + BGE-M3 + FAISS + Gemini"
)


# ============================================================
# SESSION STATE
# ============================================================

def init_state():
    defaults = {
        "example_store": None,
        "uploaded_store": None,
        "uploaded_name": None,
        "uploaded_size": None,
        "chat_example": [],
        "chat_upload": [],
        "processing_log": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# EXAMPLE PDF
# ============================================================

if page == "Example PDF":

    st.title("📄 Example PDF — Ready for Q&A")

    st.caption(
        "Pre-processed demonstration using the validated project pipeline. "
        "No document processing is required on this page."
    )

    with st.expander(
        "About the example document",
        expanded=True,
    ):

        st.write(
            "The bundled example is the processed ANN Projects prospectus. "
            "Its validated retrieval assets are loaded directly from the repository."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Pages",
            "462",
        )

        col2.metric(
            "Retrieval chunks",
            "1,536",
        )

        col3.metric(
            "Embedding size",
            "1,024",
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Only load the existing pre-processed assets.
    #
    # This does NOT run Docling.
    # This does NOT process the PDF.
    # This does NOT generate embeddings.
    # --------------------------------------------------------

    if st.session_state.example_store is None:

        try:

            with st.spinner(
                "Loading the pre-processed example index..."
            ):

                st.session_state.example_store = (
                    load_example_store()
                )

        except Exception as exc:

            st.error(
                f"Could not load the example index: {exc}"
            )

            st.exception(exc)

            st.stop()

    render_chat(
        key_prefix="example",
        history_key="chat_example",
        store=st.session_state.example_store,
        title="Ask the example document",
    )


# ============================================================
# UPLOAD YOUR OWN PDF
# ============================================================

else:

    st.title("📤 Upload Your Own PDF")

    st.caption(
        "Upload a PDF and process it with the same "
        "structure-aware RAG approach used for the validated example."
    )

    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help=(
            "PDF documents containing text, tables, figures, "
            "charts, or mixed layouts are supported."
        ),
    )

    if uploaded is None:

        st.info(
            "Choose a PDF to start processing."
        )

        st.stop()

    st.write(
        f"**Document:** {uploaded.name}  \n"
        f"**File size:** "
        f"{uploaded.size / (1024 * 1024):.2f} MB"
    )

    # --------------------------------------------------------
    # Detect a newly uploaded file
    # --------------------------------------------------------

    file_signature = (
        f"{uploaded.name}:{uploaded.size}"
    )

    if (
        st.session_state.uploaded_name
        != file_signature
    ):

        st.session_state.uploaded_store = None
        st.session_state.chat_upload = []
        st.session_state.processing_log = []

        st.session_state.uploaded_name = (
            file_signature
        )

        st.session_state.uploaded_size = (
            uploaded.size
        )

    # --------------------------------------------------------
    # Process PDF only after explicit button click
    # --------------------------------------------------------

    if st.session_state.uploaded_store is None:

        if st.button(
            "Process PDF",
            type="primary",
            use_container_width=True,
        ):

            try:

                progress = st.progress(0)

                status = st.empty()

                def update(stage, value):

                    status.info(stage)

                    progress.progress(
                        min(
                            max(
                                float(value),
                                0.0,
                            ),
                            1.0,
                        )
                    )

                with st.spinner(
                    "Processing document..."
                ):

                    store = build_uploaded_store(
                        uploaded,
                        progress_callback=update,
                    )

                st.session_state.uploaded_store = (
                    store
                )

                progress.progress(1.0)

                status.success(
                    "Document ready for Q&A."
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    f"Processing failed: {exc}"
                )

                st.exception(exc)

                st.stop()

        else:

            st.stop()

    # --------------------------------------------------------
    # Processing statistics
    # --------------------------------------------------------

    render_processing_status(
        st.session_state.uploaded_store
    )

    # --------------------------------------------------------
    # Q&A
    # --------------------------------------------------------

    render_chat(
        key_prefix="upload",
        history_key="chat_upload",
        store=st.session_state.uploaded_store,
        title="Ask your uploaded document",
    )
