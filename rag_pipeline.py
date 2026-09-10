from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import Callable, Any

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_ROOT = Path(os.getenv("RAG_INDEX_ROOT", "storage"))
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "4")
)

DEFAULT_TOP_K = int(
    os.getenv("TOP_K", "5")
)

CHUNK_TARGET = int(
    os.getenv("CHUNK_TARGET", "1800")
)

CHUNK_MAXIMUM = int(
    os.getenv("CHUNK_MAXIMUM", "3200")
)


Progress = Callable[[float, str], None] | None


# ============================================================
# HELPERS
# ============================================================

def _progress(
    cb: Progress,
    frac: float,
    message: str,
) -> None:
    if cb:
        cb(float(frac), message)


def _doc_dir(document_id: str) -> Path:
    return INDEX_ROOT / document_id


def load_manifest(
    document_id: str,
) -> dict[str, Any] | None:

    path = _doc_dir(document_id) / "manifest.json"

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def _atomic_json(
    path: Path,
    data: Any,
) -> None:

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def _iter_collection(
    doc: Any,
    names: list[str],
) -> list[Any]:

    for name in names:

        value = getattr(
            doc,
            name,
            None,
        )

        if value is not None:

            try:
                return list(value)
            except TypeError:
                pass

    return []


def _page(obj: Any) -> int | None:

    for name in (
        "page_no",
        "page",
        "page_number",
    ):

        value = getattr(
            obj,
            name,
            None,
        )

        if value is not None:

            try:
                return int(value)
            except Exception:
                return None

    return None


def _text(obj: Any) -> str:

    for name in (
        "text",
        "caption",
        "label",
    ):

        value = getattr(
            obj,
            name,
            None,
        )

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


# ============================================================
# DOCLING PDF CONVERSION
# ============================================================

def _convert_pdf(
    pdf_path: Path,
):

    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )

    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
    )

    from docling.datamodel.base_models import (
        InputFormat,
    )

    options = PdfPipelineOptions()

    # --------------------------------------------------------
    # IMPORTANT:
    # OCR is intentionally disabled.
    #
    # This prevents RapidOCR model downloads and permission
    # problems on Streamlit Cloud.
    # --------------------------------------------------------

    options.do_ocr = False

    # --------------------------------------------------------
    # Table structure extraction stays enabled.
    # This is important for complex PDFs.
    # --------------------------------------------------------

    options.do_table_structure = True

    # --------------------------------------------------------
    # Extract pictures so charts/figures can be passed
    # through the Gemini visual-description stage.
    # --------------------------------------------------------

    options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )

    result = converter.convert(
        str(pdf_path)
    )

    del converter
    gc.collect()

    return result


# ============================================================
# VISUAL DETECTION
# ============================================================

def _find_visual_labels(
    doc: Any,
) -> list[dict[str, Any]]:

    labels: list[dict[str, Any]] = []

    items = _iter_collection(
        doc,
        [
            "texts",
            "text_items",
            "texts_and_tables",
        ],
    )

    for item in items:

        text = _text(item)

        if re.search(
            r"\b(chart|figure|fig\.?)\b",
            text,
            re.IGNORECASE,
        ):

            labels.append(
                {
                    "page": _page(item),
                    "label": text,
                }
            )

    return labels


def _selected_pictures(
    doc: Any,
    labels: list[dict[str, Any]],
) -> list[tuple[Any, int | None]]:

    pictures = _iter_collection(
        doc,
        [
            "pictures",
            "picture_items",
        ],
    )

    pages = {
        item["page"]
        for item in labels
        if item.get("page") is not None
    }

    if not pages:
        return []

    selected = []

    for picture in pictures:

        page = _page(picture)

        if page in pages:

            selected.append(
                (
                    picture,
                    page,
                )
            )

    return selected


def _picture_image(
    picture: Any,
    doc: Any,
):

    if hasattr(
        picture,
        "get_image",
    ):

        return picture.get_image(doc)

    return getattr(
        picture,
        "image",
        None,
    )


# ============================================================
# GEMINI VISUAL DESCRIPTION
# ============================================================

def _gemini_describe(
    image: Any,
    api_key: str,
    model_name: str,
    page: int | None,
) -> str:

    from google import genai

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
Describe ONLY the visible information in this
PDF chart or figure for retrieval-augmented generation.

Page: {page}

Include:

- title
- chart/figure type
- labels
- legend
- units
- exact readable numbers
- dates
- comparisons
- trends
- forecasts
- annotations
- visible source

Rules:

1. Use ONLY information visibly present.
2. Do NOT invent values.
3. Do NOT estimate unreadable numbers.
4. Do NOT describe unrelated page content.
5. Preserve exact numbers and units.
6. Keep the description factual and concise.
7. If a value is not readable, do not create one.

Return concise factual prose.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=[
            prompt,
            image,
        ],
    )

    return (
        response.text or ""
    ).strip()


def _visual_pass(
    doc: Any,
    api_key: str,
    model_name: str,
    cb: Progress,
) -> list[dict[str, Any]]:

    if not api_key:
        return []

    labels = _find_visual_labels(
        doc
    )

    selected = _selected_pictures(
        doc,
        labels,
    )

    descriptions = []

    total = max(
        1,
        len(selected),
    )

    for i, (
        picture,
        page,
    ) in enumerate(
        selected,
        start=1,
    ):

        _progress(
            cb,
            0.35
            + 0.17
            * ((i - 1) / total),
            f"Describing visual {i}/{total}...",
        )

        image = None

        try:

            image = _picture_image(
                picture,
                doc,
            )

            if image is not None:

                text = _gemini_describe(
                    image,
                    api_key,
                    model_name,
                    page,
                )

                if text:

                    descriptions.append(
                        {
                            "kind": "visual",
                            "page": page,
                            "text": text,
                        }
                    )

        except Exception as exc:

            # A single visual should not destroy
            # the complete PDF processing pipeline.

            print(
                f"Visual description failed "
                f"on page {page}: {exc}"
            )

        finally:

            try:

                if image is not None:
                    image.close()

            except Exception:
                pass

            del image
            gc.collect()

    _progress(
        cb,
        0.52,
        f"Visual pass complete: "
        f"{len(descriptions)} descriptions.",
    )

    return descriptions


# ============================================================
# TEXT CLEANING
# ============================================================

def _clean_text(
    text: str,
) -> str:

    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    if len(lines) < 6:
        return "\n".join(lines)

    counts: dict[str, int] = {}

    for line in lines:

        if len(line) <= 180:

            counts[line] = (
                counts.get(line, 0)
                + 1
            )

    repeated = {
        line
        for line, count
        in counts.items()
        if count >= 3
    }

    return "\n".join(
        line
        for line in lines
        if line not in repeated
    )


# ============================================================
# TABLE EXTRACTION
# ============================================================

def _table_text(
    table: Any,
    doc: Any,
) -> str:

    # Current Docling API
    fn = getattr(
        table,
        "export_to_markdown",
        None,
    )

    if callable(fn):

        try:

            return fn(
                doc=doc
            )

        except TypeError:

            try:

                return fn(doc)

            except Exception:
                pass

        except Exception:
            pass

    # Compatibility fallback
    fn = getattr(
        table,
        "to_markdown",
        None,
    )

    if callable(fn):

        try:
            return fn()
        except Exception:
            pass

    return str(table)


# ============================================================
# CANONICAL DOCUMENT
# ============================================================

def _canonical_elements(
    doc: Any,
) -> list[dict[str, Any]]:

    elements = []

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    for item in _iter_collection(
        doc,
        [
            "texts",
            "text_items",
        ],
    ):

        text = _clean_text(
            _text(item)
        )

        if text:

            elements.append(
                {
                    "kind": "text",
                    "page": _page(item),
                    "text": text,
                }
            )

    # --------------------------------------------------------
    # TABLES
    # --------------------------------------------------------

    for table in _iter_collection(
        doc,
        [
            "tables",
            "table_items",
        ],
    ):

        text = _table_text(
            table,
            doc,
        ).strip()

        if text:

            elements.append(
                {
                    "kind": "table",
                    "page": _page(table),
                    "text": text,
                }
            )

    return elements


# ============================================================
# CHUNKING
# ============================================================

def _chunk(
    elements: list[dict[str, Any]],
    target: int = CHUNK_TARGET,
    maximum: int = CHUNK_MAXIMUM,
) -> list[dict[str, Any]]:

    result = []

    buffer = []
    length = 0

    def flush():

        nonlocal buffer
        nonlocal length

        if buffer:

            result.append(
                {
                    "kind": "text",
                    "page": buffer[0].get(
                        "page"
                    ),
                    "text": "\n\n".join(
                        x["text"]
                        for x in buffer
                    ).strip(),
                }
            )

        buffer = []
        length = 0

    for element in elements:

        text = element.get(
            "text",
            "",
        ).strip()

        kind = element.get(
            "kind"
        )

        if not text:
            continue

        # ----------------------------------------------------
        # TABLES AND VISUALS ARE ATOMIC
        # ----------------------------------------------------

        if kind in {
            "table",
            "visual",
        }:

            flush()

            result.append(
                {
                    "kind": kind,
                    "page": element.get(
                        "page"
                    ),
                    "text": text,
                }
            )

            continue

        # ----------------------------------------------------
        # NORMAL TEXT
        # ----------------------------------------------------

        if (
            buffer
            and length
            + len(text)
            + 2
            > maximum
        ):

            flush()

        buffer.append(
            element
        )

        length += (
            len(text)
            + 2
        )

        if length >= target:
            flush()

    flush()

    return result


# ============================================================
# EMBEDDING MODEL
# ============================================================

def _load_embedding_model():

    from sentence_transformers import (
        SentenceTransformer,
    )

    # Explicit CPU prevents accidental GPU/CUDA usage.
    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )

    return model


# ============================================================
# FAISS INDEX CREATION
# ============================================================

def _embed_index(
    chunks: list[dict[str, Any]],
    index_path: Path,
    cb: Progress,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> None:

    import faiss

    if not chunks:
        raise RuntimeError(
            "No chunks available for embedding."
        )

    _progress(
        cb,
        0.61,
        "Loading embedding model...",
    )

    model = _load_embedding_model()

    _progress(
        cb,
        0.64,
        "Embedding model loaded.",
    )

    index = None

    total = len(chunks)

    try:

        for start in range(
            0,
            total,
            batch_size,
        ):

            batch = chunks[
                start:
                start + batch_size
            ]

            texts = [
                item["text"]
                for item in batch
            ]

            vectors = model.encode(
                texts,
                batch_size=min(
                    batch_size,
                    len(texts),
                ),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(
                "float32"
            )

            if index is None:

                index = faiss.IndexFlatIP(
                    vectors.shape[1]
                )

            index.add(
                vectors
            )

            del vectors
            del texts
            gc.collect()

            completed = (
                start
                + len(batch)
            )

            _progress(
                cb,
                0.64
                + 0.16
                * (
                    completed
                    / total
                ),
                f"Embedding chunks "
                f"{completed}/{total}...",
            )

        if index is None:

            raise RuntimeError(
                "FAISS index was not created."
            )

        faiss.write_index(
            index,
            str(index_path),
        )

    finally:

        del index
        del model
        gc.collect()


# ============================================================
# BUILD / LOAD DOCUMENT
# ============================================================

def build_or_load_document(
    pdf_bytes: bytes,
    source_name: str,
    document_id: str,
    api_key: str,
    visual_model: str,
    progress_callback: Progress = None,
):

    existing = load_manifest(
        document_id
    )

    index_path = (
        _doc_dir(document_id)
        / "index.faiss"
    )

    chunks_path = (
        _doc_dir(document_id)
        / "chunks.json"
    )

    # --------------------------------------------------------
    # EXISTING INDEX
    # --------------------------------------------------------

    if (
        existing
        and index_path.exists()
        and chunks_path.exists()
    ):

        _progress(
            progress_callback,
            1.0,
            "Existing processed index loaded.",
        )

        return existing

    # --------------------------------------------------------
    # CREATE DOCUMENT DIRECTORY
    # --------------------------------------------------------

    doc_dir = _doc_dir(
        document_id
    )

    doc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        doc_dir
        / "source.pdf"
    )

    pdf_path.write_bytes(
        pdf_bytes
    )

    # --------------------------------------------------------
    # DOCLING
    # --------------------------------------------------------

    _progress(
        progress_callback,
        0.03,
        "Converting PDF with Docling...",
    )

    result = _convert_pdf(
        pdf_path
    )

    doc = result.document

    try:

        pages = len(
            getattr(
                doc,
                "pages",
                [],
            )
        )

    except Exception:

        pages = None

    _progress(
        progress_callback,
        0.30,
        "Docling extraction complete.",
    )

    # --------------------------------------------------------
    # VISUAL DESCRIPTION
    # --------------------------------------------------------

    visual_descriptions = _visual_pass(
        doc,
        api_key,
        visual_model,
        progress_callback,
    )

    # --------------------------------------------------------
    # CANONICAL STRUCTURE
    # --------------------------------------------------------

    _progress(
        progress_callback,
        0.54,
        "Building canonical structure...",
    )

    elements = _canonical_elements(
        doc
    )

    elements.extend(
        visual_descriptions
    )

    # --------------------------------------------------------
    # RELEASE DOCLING OBJECTS
    # --------------------------------------------------------

    del result
    del doc
    gc.collect()

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    chunks = _chunk(
        elements
    )

    del elements
    gc.collect()

    if not chunks:

        raise RuntimeError(
            "No usable text, table, "
            "or visual content was extracted."
        )

    _atomic_json(
        chunks_path,
        chunks,
    )

    _progress(
        progress_callback,
        0.60,
        f"Prepared {len(chunks)} chunks.",
    )

    # --------------------------------------------------------
    # EMBEDDING + FAISS
    # --------------------------------------------------------

    _embed_index(
        chunks,
        index_path,
        progress_callback,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest = {
        "document_id": document_id,
        "source_name": source_name,
        "pages": pages,
        "num_chunks": len(chunks),

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "index_type":
            "FAISS IndexFlatIP + normalized embeddings",

        "chunking":
            f"target {CHUNK_TARGET} / "
            f"max {CHUNK_MAXIMUM}; "
            f"tables and visuals atomic",

        "visual_descriptions":
            len(visual_descriptions),

        "ocr": False,

        "table_structure": True,

        "picture_images": True,

        "notebook_aligned": True,
    }

    _atomic_json(
        doc_dir / "manifest.json",
        manifest,
    )

    del chunks
    del visual_descriptions

    gc.collect()

    _progress(
        progress_callback,
        1.0,
        "Index saved. Ready for Q&A.",
    )

    return manifest


# ============================================================
# RETRIEVAL
# ============================================================

def _retrieve(
    document_id: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
):

    import faiss

    doc_dir = _doc_dir(
        document_id
    )

    chunks_path = (
        doc_dir
        / "chunks.json"
    )

    index_path = (
        doc_dir
        / "index.faiss"
    )

    if not chunks_path.exists():
        raise FileNotFoundError(
            "Chunk file not found."
        )

    if not index_path.exists():
        raise FileNotFoundError(
            "FAISS index not found."
        )

    chunks = json.loads(
        chunks_path.read_text(
            encoding="utf-8"
        )
    )

    index = faiss.read_index(
        str(index_path)
    )

    if index.ntotal == 0:
        return []

    model = _load_embedding_model()

    try:

        query_vector = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(
            "float32"
        )

        k = min(
            top_k,
            index.ntotal,
        )

        scores, ids = index.search(
            query_vector,
            k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            ids[0],
        ):

            if (
                0 <= idx
                < len(chunks)
            ):

                item = dict(
                    chunks[int(idx)]
                )

                item["score"] = float(
                    score
                )

                results.append(
                    item
                )

        return results

    finally:

        del model
        del index

        if "query_vector" in locals():
            del query_vector

        gc.collect()


# ============================================================
# ANSWER GENERATION
# ============================================================

def _answer(
    question: str,
    context: str,
    api_key: str,
    model_name: str,
) -> str:

    from google import genai

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the QUESTION using ONLY the
RETRIEVED PDF CONTEXT.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the retrieved context is insufficient,
   clearly say that the information is not
   available in the retrieved context.
4. Preserve exact numbers.
5. Preserve dates.
6. Preserve units.
7. Preserve names.
8. Preserve table relationships.
9. Do not change numerical values.
10. Use visual descriptions only for information
    explicitly visible in the visual.
11. Mention page numbers when useful.
12. Give a concise direct answer.

QUESTION:

{question}

RETRIEVED PDF CONTEXT:

{context}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return (
        response.text or ""
    ).strip()


# ============================================================
# PUBLIC QA FUNCTION
# ============================================================

def answer_question(
    document_id: str,
    question: str,
    api_key: str,
    answer_model: str,
    top_k: int = DEFAULT_TOP_K,
):

    if not api_key:

        raise ValueError(
            "Gemini API key is required."
        )

    question = (
        question or ""
    ).strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    results = _retrieve(
        document_id,
        question,
        top_k,
    )

    if not results:

        return {
            "answer":
                "No relevant information "
                "was retrieved from the document.",
            "sources": [],
        }

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
[Page {result.get('page', '?')}
 | {result.get('kind')}
 | similarity {result['score']:.4f}]

{result['text']}
""".strip()
        )

    context = (
        "\n\n---\n\n"
        .join(context_parts)
    )

    answer = _answer(
        question,
        context,
        api_key,
        answer_model,
    )

    sources = [
        {
            "page":
                result.get("page"),

            "kind":
                result.get("kind"),

            "score":
                result["score"],
        }

        for result in results
    ]

    del results
    gc.collect()

    return {
        "answer": answer,
        "sources": sources,
    }
