import os
import tempfile
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

from .config import (
    ROOT_DIR,
    EXAMPLE_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    RETRIEVAL_TOP_K,
    GEMINI_API_KEY_ENV,
    GEMINI_MODELS,
    MAX_RETRIES,
    INITIAL_RETRY_DELAY,
)


# ============================================================
# SECRET / ENVIRONMENT
# ============================================================

def _secret_or_env(name):
    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st

        return st.secrets.get(name)

    except Exception:
        return None


# ============================================================
# JSON
# ============================================================

def _load_json(path):
    import json

    path = Path(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# BGE-M3 EMBEDDING MODEL
# ============================================================

def _load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )


# ------------------------------------------------------------
# Streamlit resource cache.
#
# IMPORTANT:
# BGE-M3 is loaded only when it is actually required.
# It is NOT loaded while the Example PDF store is created.
# ------------------------------------------------------------

try:

    import streamlit as st

    _load_embedding_model = st.cache_resource(
        show_spinner="Loading BGE-M3 embedding model..."
    )(_load_embedding_model)

except Exception:

    pass


# ============================================================
# EXAMPLE STORE
# ============================================================

def load_example_store():

    chunks_path = (
        EXAMPLE_DIR
        / "chunks"
        / "chunks_final_v2.json"
    )

    metadata_path = (
        EXAMPLE_DIR
        / "vectors"
        / "embedding_metadata.json"
    )

    index_path = (
        EXAMPLE_DIR
        / "vectors"
        / "faiss.index"
    )

    if not chunks_path.exists():

        raise FileNotFoundError(
            f"Example chunks file not found: {chunks_path}"
        )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Example embedding metadata not found: {metadata_path}"
        )

    if not index_path.exists():

        raise FileNotFoundError(
            f"Example FAISS index not found: {index_path}"
        )

    chunks = _load_json(
        chunks_path
    )

    metadata = _load_json(
        metadata_path
    )

    index = faiss.read_index(
        str(index_path)
    )

    if not chunks:

        raise ValueError(
            "Example chunks file is empty."
        )

    if not metadata:

        raise ValueError(
            "Example embedding metadata is empty."
        )

    if index.ntotal == 0:

        raise ValueError(
            "Example FAISS index is empty."
        )

    if len(chunks) != index.ntotal:

        raise ValueError(
            "Example data mismatch: "
            f"{len(chunks)} chunks but "
            f"{index.ntotal} FAISS vectors."
        )

    if len(metadata) != index.ntotal:

        raise ValueError(
            "Example data mismatch: "
            f"{len(metadata)} metadata records but "
            f"{index.ntotal} FAISS vectors."
        )

    return {
        "name": (
            "ANN Projects Prospectus — "
            "pre-processed example"
        ),
        "chunks": chunks,
        "metadata": metadata,
        "index": index,
        "embedding_model": EMBEDDING_MODEL,
        "document_path": (
            EXAMPLE_DIR
            / "1788150561832.pdf"
        ),
        "source_type": "example",
    }


# ============================================================
# EMBEDDING
# ============================================================

def _embed_chunks(chunks):

    model = _load_embedding_model()

    texts = [
        c["content"]
        for c in chunks
    ]

    vectors = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        "float32"
    )

    return vectors


# ============================================================
# FAISS
# ============================================================

def _build_index(vectors):

    if vectors is None:

        raise ValueError(
            "Embedding vectors are None."
        )

    if len(vectors) == 0:

        raise ValueError(
            "No embedding vectors were generated."
        )

    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

    index.add(vectors)

    return index


# ============================================================
# METADATA
# ============================================================

def _metadata_from_chunks(chunks):

    metadata = []

    for i, chunk in enumerate(chunks):

        metadata.append(
            {
                "faiss_index": i,

                "chunk_id":
                    chunk["chunk_id"],

                "page_start":
                    chunk["page_start"],

                "page_end":
                    chunk["page_end"],

                "section":
                    chunk.get("section"),

                "element_ids":
                    chunk["element_ids"],

                "element_types":
                    chunk["element_types"],

                "has_table":
                    chunk["has_table"],

                "has_visual":
                    chunk["has_visual"],
            }
        )

    return metadata


# ============================================================
# UPLOADED PDF STORE
# ============================================================

def build_uploaded_store(
    uploaded_file,
    progress_callback=None,
):

    # --------------------------------------------------------
    # IMPORTANT:
    # Docling is imported ONLY when an uploaded PDF is
    # explicitly processed.
    #
    # Therefore Example PDF startup does not import Docling.
    # --------------------------------------------------------

    from .document_processor import process_pdf

    api_key = _secret_or_env(
        GEMINI_API_KEY_ENV
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Add it to the environment or "
            "Streamlit secrets before processing a PDF."
        )

    # --------------------------------------------------------
    # Temporary working directory
    # --------------------------------------------------------

    work_root = Path(
        tempfile.mkdtemp(
            prefix="complex_pdf_rag_"
        )
    )

    pdf_path = (
        work_root
        / uploaded_file.name
    )

    pdf_path.write_bytes(
        uploaded_file.getvalue()
    )

    # --------------------------------------------------------
    # Docling processing
    # --------------------------------------------------------

    canonical, chunks, processed_dir = (
        process_pdf(
            pdf_path,
            work_root / "processed",
            api_key,
            progress_callback=progress_callback,
        )
    )

    if not chunks:

        raise ValueError(
            "No retrieval chunks were produced from the PDF."
        )

    # --------------------------------------------------------
    # BGE-M3 embeddings
    # --------------------------------------------------------

    vectors = _embed_chunks(
        chunks
    )

    if progress_callback:

        progress_callback(
            "✓ Embeddings generated with BGE-M3",
            0.90,
        )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    index = _build_index(
        vectors
    )

    metadata = _metadata_from_chunks(
        chunks
    )

    if progress_callback:

        progress_callback(
            "✓ FAISS index created",
            0.97,
        )

    return {
        "name":
            uploaded_file.name,

        "chunks":
            chunks,

        "metadata":
            metadata,

        "index":
            index,

        "embedding_model":
            EMBEDDING_MODEL,

        "document_path":
            pdf_path,

        "processed_dir":
            processed_dir,

        "source_type":
            "upload",

        "statistics":
            canonical["statistics"],
    }


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    store,
    question,
    top_k=RETRIEVAL_TOP_K,
):

    model = _load_embedding_model()

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        "float32"
    )

    actual_top_k = min(
        top_k,
        store["index"].ntotal,
    )

    scores, indices = (
        store["index"].search(
            query_embedding,
            actual_top_k,
        )
    )

    results = []

    for rank, (
        score,
        idx,
    ) in enumerate(
        zip(
            scores[0],
            indices[0],
        ),
        start=1,
    ):

        if (
            idx < 0
            or idx >= len(
                store["chunks"]
            )
        ):
            continue

        chunk = store["chunks"][
            int(idx)
        ]

        meta = store["metadata"][
            int(idx)
        ]

        results.append(
            {
                "rank":
                    rank,

                "score":
                    float(score),

                "chunk":
                    chunk,

                "metadata":
                    meta,
            }
        )

    return results


# ============================================================
# CONTEXT
# ============================================================

def build_context(results):

    parts = []

    for result in results:

        chunk = result["chunk"]

        parts.append(
            f"""SOURCE {result['rank']}
Page: {chunk.get('page_start')} - {chunk.get('page_end')}
Section: {chunk.get('section')}
Chunk ID: {chunk.get('chunk_id')}
Content:
{chunk.get('content')}
"""
        )

    return "\n\n".join(
        parts
    )


# ============================================================
# GEMINI RETRY
# ============================================================

def _retryable(text):

    text = str(
        text
    ).upper()

    return any(
        x in text
        for x in [
            "429",
            "500",
            "502",
            "503",
            "504",
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "INTERNAL",
            "DEADLINE",
        ]
    )


def call_gemini_with_fallback(
    client,
    prompt,
):

    last_error = None

    for model_name in GEMINI_MODELS:

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            try:

                response = (
                    client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                )

                if not response.text:

                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                return (
                    response.text.strip(),
                    model_name,
                )

            except Exception as exc:

                last_error = exc

                if not _retryable(
                    exc
                ):

                    break

                if attempt < MAX_RETRIES:

                    import time
                    import random

                    time.sleep(
                        INITIAL_RETRY_DELAY
                        * (
                            2
                            ** (
                                attempt - 1
                            )
                        )
                        + random.uniform(
                            0,
                            2,
                        )
                    )

    raise RuntimeError(
        "All configured Gemini models failed."
    ) from last_error


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    results,
):

    api_key = _secret_or_env(
        GEMINI_API_KEY_ENV
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY is not configured."
        )

    context = build_context(
        results
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the retrieved document context below.

IMPORTANT RULES:
1. Do not use outside knowledge.
2. Do not invent information.
3. Carefully compare all retrieved sources before answering.
4. Prefer the source that directly answers the exact question.
5. Do not select a source merely because it has the highest similarity score.
6. Pay close attention to dates, categories, units, metrics, geographic scope,
   and time periods.
7. Preserve numerical values from the document.
8. If calculation is required, calculate it using the numbers provided.
9. If the retrieved context does not contain enough information, clearly say
   that the information is not available.
10. For multi-part questions, answer every part.
11. Mention the relevant page number when useful.
12. Do not mention FAISS, embeddings, chunks, or retrieval unless specifically asked.

FORMAT RULES:
- Return clean plain text or simple Markdown.
- Never use LaTeX.
- Use % for percentages.
- Keep the answer concise but complete.

RETRIEVED DOCUMENT CONTEXT
==========================
{context}

USER QUESTION
=============
{question}

FINAL ANSWER
============
Give a concise, accurate answer based strictly on the retrieved document context.
"""

    client = genai.Client(
        api_key=api_key
    )

    return call_gemini_with_fallback(
        client,
        prompt,
    )


# ============================================================
# COMPLETE QUESTION ANSWERING
# ============================================================

def answer_question(
    store,
    question,
):

    results = retrieve_chunks(
        store,
        question,
    )

    answer, model_used = (
        generate_answer(
            question,
            results,
        )
    )

    return (
        answer,
        model_used,
        results,
    )
