from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_ROOT = Path(
    os.getenv("RAG_INDEX_ROOT", "storage")
)

INDEX_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)

EMBEDDING_BATCH_SIZE = max(
    1,
    int(
        os.getenv(
            "EMBEDDING_BATCH_SIZE",
            "4",
        )
    ),
)

DEFAULT_TOP_K = max(
    1,
    int(
        os.getenv(
            "TOP_K",
            "5",
        )
    ),
)

CHUNK_TARGET = int(
    os.getenv(
        "CHUNK_TARGET",
        "1800",
    )
)

CHUNK_MAXIMUM = int(
    os.getenv(
        "CHUNK_MAXIMUM",
        "3200",
    )
)


Progress = Callable[
    [float, str],
    None,
] | None


# ============================================================
# LOGGING / PROGRESS
# ============================================================

def _log(message: str) -> None:
    """
    Flush immediately so Streamlit Cloud logs show the
    exact stage before a possible native crash / OOM kill.
    """

    print(
        message,
        flush=True,
    )


def _progress(
    cb: Progress,
    frac: float,
    message: str,
) -> None:

    if cb:
        cb(
            float(frac),
            message,
        )

    _log(
        f"[RAG] {message}"
    )


# ============================================================
# STORAGE HELPERS
# ============================================================

def _doc_dir(
    document_id: str,
) -> Path:

    return INDEX_ROOT / document_id


def load_manifest(
    document_id: str,
) -> dict[str, Any] | None:

    path = (
        _doc_dir(document_id)
        / "manifest.json"
    )

    if not path.exists():
        return None

    try:

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        _log(
            f"[RAG] Manifest read failed: {exc}"
        )

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


def _cleanup_memory() -> None:

    gc.collect()


# ============================================================
# DOCLING OBJECT HELPERS
# ============================================================

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

        if value is None:
            continue

        try:
            return list(value)

        except TypeError:
            continue

        except Exception as exc:

            _log(
                f"[RAG] Could not read "
                f"Docling collection '{name}': {exc}"
            )

    return []


def _page(
    obj: Any,
) -> int | None:

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

        if value is None:
            continue

        try:
            return int(value)

        except Exception:
            continue

    return None


def _text(
    obj: Any,
) -> str:

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

        if (
            isinstance(value, str)
            and value.strip()
        ):

            return value.strip()

    return ""


# ============================================================
# DOCLING PDF CONVERSION
# ============================================================

def _convert_pdf(
    pdf_path: Path,
):

    _log(
        ">>> STAGE 1: Importing Docling"
    )

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

    _log(
        ">>> STAGE 2: Creating PdfPipelineOptions"
    )

    options = PdfPipelineOptions()

    # --------------------------------------------------------
    # OCR MUST REMAIN DISABLED
    # --------------------------------------------------------

    options.do_ocr = False

    # --------------------------------------------------------
    # TABLE STRUCTURE IS REQUIRED
    # --------------------------------------------------------

    options.do_table_structure = True

    # --------------------------------------------------------
    # PICTURE EXTRACTION
    # --------------------------------------------------------

    options.generate_picture_images = True

    _log(
        ">>> STAGE 3: Creating DocumentConverter"
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF:
                PdfFormatOption(
                    pipeline_options=options
                )
        }
    )

    _log(
        ">>> STAGE 4: Starting Docling conversion"
    )

    result = converter.convert(
        str(pdf_path)
    )

    _log(
        ">>> STAGE 5: Docling conversion COMPLETE"
    )

    # Release converter immediately.
    del converter
    del options

    _cleanup_memory()

    _log(
        ">>> STAGE 6: Docling converter released"
    )

    return result


# ============================================================
# VISUAL DETECTION
# ============================================================

def _find_visual_labels(
    doc: Any,
) -> list[dict[str, Any]]:

    _log(
        ">>> VISUAL: Searching for chart/figure labels"
    )

    labels: list[
        dict[str, Any]
    ] = []

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

        if not text:
            continue

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

    _log(
        f">>> VISUAL: Found "
        f"{len(labels)} visual labels"
    )

    return labels


def _selected_pictures(
    doc: Any,
    labels: list[dict[str, Any]],
) -> list[
    tuple[Any, int | None]
]:

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

        _log(
            ">>> VISUAL: No visual pages detected"
        )

        return []

    selected = []

    for picture in pictures:

        page = _page(
            picture
        )

        if page in pages:

            selected.append(
                (
                    picture,
                    page,
                )
            )

    _log(
        f">>> VISUAL: Selected "
        f"{len(selected)} pictures"
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

        return picture.get_image(
            doc
        )

    return getattr(
        picture,
        "image",
        None,
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

def _create_gemini_client(
    api_key: str,
):

    from google import genai

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# GEMINI VISUAL DESCRIPTION
# ============================================================

def _gemini_describe(
    client: Any,
    image: Any,
    model_name: str,
    page: int | None,
) -> str:

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

        _log(
            ">>> VISUAL: Gemini key unavailable; "
            "visual descriptions skipped"
        )

        return []

    _log(
        ">>> VISUAL: Starting visual pass"
    )

    labels = _find_visual_labels(
        doc
    )

    selected = _selected_pictures(
        doc,
        labels,
    )

    if not selected:

        _progress(
            cb,
            0.52,
            "No chart/figure visuals selected.",
        )

        return []

    client = _create_gemini_client(
        api_key
    )

    descriptions = []

    total = len(
        selected
    )

    for i, (
        picture,
        page,
    ) in enumerate(
        selected,
        start=1,
    ):

        _log(
            f">>> VISUAL: Processing visual "
            f"{i}/{total}, page={page}"
        )

        _progress(
            cb,
            0.35
            + 0.17
            * (
                (i - 1)
                / max(1, total)
            ),
            f"Describing visual "
            f"{i}/{total}...",
        )

        image = None

        try:

            image = _picture_image(
                picture,
                doc,
            )

            if image is None:

                _log(
                    f">>> VISUAL: No image "
                    f"available for page {page}"
                )

                continue

            text = _gemini_describe(
                client,
                image,
                model_name,
                page,
            )

            _log(
                f">>> VISUAL: Gemini description "
                f"complete for page {page}"
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

            _log(
                f">>> VISUAL ERROR: page "
                f"{page}: {exc}"
            )

        finally:

            try:

                if image is not None:
                    image.close()

            except Exception:
                pass

            del image

            _cleanup_memory()

    del client

    _progress(
        cb,
        0.52,
        f"Visual pass complete: "
        f"{len(descriptions)} descriptions.",
    )

    _log(
        ">>> VISUAL: Visual pass finished"
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

        return "\n".join(
            lines
        )

    counts: dict[
        str,
        int,
    ] = {}

    for line in lines:

        if len(line) <= 180:

            counts[line] = (
                counts.get(
                    line,
                    0,
                )
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

    fn = getattr(
        table,
        "export_to_markdown",
        None,
    )

    if callable(fn):

        try:

            value = fn(
                doc=doc
            )

            if value:
                return str(value)

        except TypeError:

            try:

                value = fn(
                    doc
                )

                if value:
                    return str(
                        value
                    )

            except Exception as exc:

                _log(
                    f"[RAG] Table markdown "
                    f"fallback failed: {exc}"
                )

        except Exception as exc:

            _log(
                f"[RAG] Table markdown "
                f"export failed: {exc}"
            )

    fn = getattr(
        table,
        "to_markdown",
        None,
    )

    if callable(fn):

        try:

            value = fn()

            if value:
                return str(value)

        except Exception as exc:

            _log(
                f"[RAG] to_markdown failed: {exc}"
            )

    return str(
        table
    )


# ============================================================
# CANONICAL DOCUMENT
# ============================================================

def _canonical_elements(
    doc: Any,
) -> list[
    dict[str, Any]
]:

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

    # --------------------------------------------------------
    # SORT BY PAGE
    #
    # This is important because we later merge visual
    # descriptions into the same page context.
    # --------------------------------------------------------

    elements.sort(
        key=lambda item: (
            item.get("page")
            if item.get("page") is not None
            else 10**9
        )
    )

    _log(
        f">>> CANONICAL: "
        f"{len(elements)} text/table elements"
    )

    return elements


# ============================================================
# MERGE VISUALS INTO PAGE ORDER
# ============================================================

def _merge_visuals(
    elements: list[
        dict[str, Any]
    ],
    visual_descriptions: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:

    combined = (
        list(elements)
        + list(visual_descriptions)
    )

    combined.sort(
        key=lambda item: (
            item.get("page")
            if item.get("page") is not None
            else 10**9
        )
    )

    return combined


# ============================================================
# CHUNKING
# ============================================================

def _chunk(
    elements: list[
        dict[str, Any]
    ],
    target: int = CHUNK_TARGET,
    maximum: int = CHUNK_MAXIMUM,
) -> list[
    dict[str, Any]
]:

    result = []

    buffer = []

    length = 0

    chunk_id = 0

    def flush():

        nonlocal buffer
        nonlocal length
        nonlocal chunk_id

        if not buffer:
            return

        text = "\n\n".join(
            item["text"]
            for item in buffer
        ).strip()

        if not text:
            buffer = []
            length = 0
            return

        pages = [
            item.get("page")
            for item in buffer
            if item.get("page")
            is not None
        ]

        page_start = (
            min(pages)
            if pages
            else None
        )

        page_end = (
            max(pages)
            if pages
            else None
        )

        element_types = [
            item.get("kind")
            for item in buffer
        ]

        result.append(
            {
                "chunk_id": chunk_id,
                "page_start": page_start,
                "page_end": page_end,
                "kind": "text",
                "text": text,
                "element_types":
                    element_types,
                "has_table":
                    "table"
                    in element_types,
                "has_visual":
                    "visual"
                    in element_types,
            }
        )

        chunk_id += 1

        buffer = []
        length = 0

    for element in elements:

        text = (
            element.get(
                "text",
                "",
            )
            .strip()
        )

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
                    "chunk_id": chunk_id,
                    "page_start":
                        element.get(
                            "page"
                        ),
                    "page_end":
                        element.get(
                            "page"
                        ),
                    "kind": kind,
                    "text": text,
                    "element_types":
                        [kind],
                    "has_table":
                        kind == "table",
                    "has_visual":
                        kind == "visual",
                }
            )

            chunk_id += 1

            continue

        # ----------------------------------------------------
        # NORMAL TEXT
        # ----------------------------------------------------

        projected = (
            length
            + len(text)
            + 2
        )

        if (
            buffer
            and projected > maximum
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

    _log(
        f">>> CHUNKING: Created "
        f"{len(result)} chunks"
    )

    return result


# ============================================================
# EMBEDDING MODEL
# ============================================================

def _load_embedding_model():

    _log(
        ">>> EMBEDDING: Loading BGE-M3 on CPU"
    )

    from sentence_transformers import (
        SentenceTransformer,
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )

    _log(
        ">>> EMBEDDING: BGE-M3 loaded"
    )

    return model


# ============================================================
# FAISS INDEX CREATION
# ============================================================

def _embed_index(
    chunks: list[
        dict[str, Any]
    ],
    index_path: Path,
    cb: Progress,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> None:

    if not chunks:

        raise RuntimeError(
            "No chunks available for embedding."
        )

    _log(
        f">>> EMBEDDING: About to load BGE-M3"
        f" for {len(chunks)} chunks"
    )

    import faiss

    model = None
    index = None

    try:

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

        total = len(
            chunks
        )

        _log(
            f">>> EMBEDDING: Starting batches "
            f"batch_size={batch_size}"
        )

        for start in range(
            0,
            total,
            batch_size,
        ):

            end = min(
                start + batch_size,
                total,
            )

            batch = chunks[
                start:end
            ]

            texts = [
                item["text"]
                for item in batch
            ]

            _log(
                f">>> EMBEDDING: Encoding "
                f"{start + 1}-{end}/{total}"
            )

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

                _log(
                    f">>> FAISS: Created "
                    f"IndexFlatIP dimension="
                    f"{vectors.shape[1]}"
                )

            index.add(
                vectors
            )

            del vectors
            del texts
            del batch

            _cleanup_memory()

            completed = end

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

        _log(
            f">>> FAISS: Writing index "
            f"with {index.ntotal} vectors"
        )

        faiss.write_index(
            index,
            str(index_path),
        )

        _log(
            ">>> FAISS: Index successfully saved"
        )

    finally:

        if index is not None:
            del index

        if model is not None:
            del model

        _cleanup_memory()

        _log(
            ">>> EMBEDDING: Model/index released"
        )


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

    _log(
        "================================================"
    )

    _log(
        f">>> PIPELINE: Starting document "
        f"{source_name}"
    )

    _log(
        f">>> PIPELINE: Document ID = "
        f"{document_id}"
    )

    existing = load_manifest(
        document_id
    )

    doc_dir = _doc_dir(
        document_id
    )

    index_path = (
        doc_dir
        / "index.faiss"
    )

    chunks_path = (
        doc_dir
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

        _log(
            ">>> PIPELINE: Existing index reused"
        )

        return existing

    # --------------------------------------------------------
    # DIRECTORY
    # --------------------------------------------------------

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

    _log(
        f">>> PIPELINE: PDF written to "
        f"{pdf_path}"
    )

    # --------------------------------------------------------
    # DOCLING
    # --------------------------------------------------------

    _progress(
        progress_callback,
        0.03,
        "Converting PDF with Docling...",
    )

    _log(
        ">>> PIPELINE: Calling Docling"
    )

    result = _convert_pdf(
        pdf_path
    )

    _log(
        ">>> PIPELINE: Docling returned result"
    )

    if result is None:

        raise RuntimeError(
            "Docling returned no conversion result."
        )

    doc = result.document

    _log(
        ">>> PIPELINE: Document object obtained"
    )

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

    _log(
        f">>> PIPELINE: Docling extraction "
        f"complete; pages={pages}"
    )

    # --------------------------------------------------------
    # VISUALS
    # --------------------------------------------------------

    _log(
        ">>> PIPELINE: Starting visual stage"
    )

    visual_descriptions = _visual_pass(
        doc,
        api_key,
        visual_model,
        progress_callback,
    )

    _log(
        f">>> PIPELINE: Visual stage complete; "
        f"descriptions="
        f"{len(visual_descriptions)}"
    )

    # --------------------------------------------------------
    # CANONICAL
    # --------------------------------------------------------

    _progress(
        progress_callback,
        0.54,
        "Building canonical structure...",
    )

    elements = _canonical_elements(
        doc
    )

    elements = _merge_visuals(
        elements,
        visual_descriptions,
    )

    _log(
        f">>> CANONICAL: Combined elements = "
        f"{len(elements)}"
    )

    # --------------------------------------------------------
    # RELEASE DOCLING OBJECTS
    # --------------------------------------------------------

    del result
    del doc

    _cleanup_memory()

    _log(
        ">>> PIPELINE: Docling objects released"
    )

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    chunks = _chunk(
        elements
    )

    del elements

    _cleanup_memory()

    if not chunks:

        raise RuntimeError(
            "No usable text, table, "
            "or visual content was extracted."
        )

    _log(
        f">>> PIPELINE: Chunking complete; "
        f"chunks={len(chunks)}"
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
    # EMBEDDINGS
    # --------------------------------------------------------

    _log(
        ">>> PIPELINE: About to start embedding"
    )

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
        "document_id":
            document_id,

        "source_name":
            source_name,

        "pages":
            pages,

        "num_chunks":
            len(chunks),

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "index_type":
            "FAISS IndexFlatIP + normalized embeddings",

        "chunking":
            f"target {CHUNK_TARGET} / "
            f"max {CHUNK_MAXIMUM}; "
            f"tables and visuals atomic",

        "visual_descriptions":
            len(
                visual_descriptions
            ),

        "ocr":
            False,

        "table_structure":
            True,

        "picture_images":
            True,

        "notebook_aligned":
            True,

        "embedding_batch_size":
            EMBEDDING_BATCH_SIZE,
    }

    _atomic_json(
        doc_dir
        / "manifest.json",
        manifest,
    )

    del chunks
    del visual_descriptions

    _cleanup_memory()

    _progress(
        progress_callback,
        1.0,
        "Index saved. Ready for Q&A.",
    )

    _log(
        ">>> PIPELINE: COMPLETE"
    )

    _log(
        "================================================"
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

        del index
        del chunks

        _cleanup_memory()

        return []

    model = None
    query_vector = None

    try:

        _log(
            ">>> RETRIEVAL: Loading BGE-M3"
        )

        model = _load_embedding_model()

        _log(
            ">>> RETRIEVAL: Encoding query"
        )

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
                    chunks[
                        int(idx)
                    ]
                )

                item["score"] = float(
                    score
                )

                results.append(
                    item
                )

        _log(
            f">>> RETRIEVAL: "
            f"{len(results)} results"
        )

        return results

    finally:

        if model is not None:
            del model

        del index

        if query_vector is not None:
            del query_vector

        del chunks

        _cleanup_memory()


# ============================================================
# ANSWER GENERATION
# ============================================================

def _answer(
    question: str,
    context: str,
    api_key: str,
    model_name: str,
) -> str:

    if not api_key:

        raise ValueError(
            "Gemini API key is required."
        )

    client = _create_gemini_client(
        api_key
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

    del client

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

    _log(
        f">>> QA: Question = {question}"
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
[Pages {result.get('page_start', result.get('page', '?'))}
-
{result.get('page_end', result.get('page', '?'))}
 | {result.get('kind')}
 | similarity {result['score']:.4f}]

{result['text']}
""".strip()
        )

    context = (
        "\n\n---\n\n"
        .join(
            context_parts
        )
    )

    answer = _answer(
        question,
        context,
        api_key,
        answer_model,
    )

    sources = [
        {
            "page_start":
                result.get(
                    "page_start",
                    result.get("page"),
                ),

            "page_end":
                result.get(
                    "page_end",
                    result.get("page"),
                ),

            "kind":
                result.get("kind"),

            "score":
                result["score"],
        }

        for result in results
    ]

    del results

    _cleanup_memory()

    _log(
        ">>> QA: Answer generated"
    )

    return {
        "answer":
            answer,

        "sources":
            sources,
    }
