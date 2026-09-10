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
    os.getenv(
        "RAG_INDEX_ROOT",
        "storage",
    )
)

INDEX_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# EMBEDDING MODEL
#
# BGE-M3 remains the default because this is the model used
# in the validated project pipeline.
#
# It can be overridden through:
#
# EMBEDDING_MODEL=BAAI/bge-m3
#
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
).strip()


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


# ============================================================
# GEMINI CONFIGURATION
# ============================================================
#
# IMPORTANT:
#
# The same fallback sequence is used for:
#
# 1. Visual / chart description
# 2. Final RAG answer generation
#
# If a model is unavailable for the current API key/project,
# the next model is attempted automatically.
#
# ============================================================

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]


# Timeout for EACH individual Gemini model attempt.
#
# 30 seconds means:
#
# Model 1 -> maximum approximately 30 sec
# Model 2 -> maximum approximately 30 sec
# Model 3 -> maximum approximately 30 sec
# Model 4 -> maximum approximately 30 sec
#
# The fallback does NOT wait indefinitely.
#
GEMINI_TIMEOUT_SECONDS = max(
    5,
    int(
        os.getenv(
            "GEMINI_TIMEOUT_SECONDS",
            "30",
        )
    ),
)


# ------------------------------------------------------------
# CPU MEMORY / THREAD CONTROL
# ------------------------------------------------------------

# These do not change the RAG architecture.
# They reduce unnecessary CPU-side resource pressure.

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1",
)


Progress = Callable[
    [float, str],
    None,
] | None


# ============================================================
# LOGGING / PROGRESS
# ============================================================

def _log(
    message: str,
) -> None:
    """
    Flush immediately so Streamlit Cloud logs show the exact
    stage before a possible native crash / OOM termination.
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
# MEMORY CLEANUP
# ============================================================

def _cleanup_memory() -> None:

    gc.collect()


def _log_torch_environment() -> None:
    """
    Diagnostic information for deployment debugging.

    This is deliberately called immediately before loading
    the embedding model.
    """

    try:

        import torch

        _log(
            ">>> TORCH: "
            f"version={torch.__version__}"
        )

        _log(
            ">>> TORCH: "
            f"cuda_available={torch.cuda.is_available()}"
        )

        _log(
            ">>> TORCH: "
            f"cuda_version={torch.version.cuda}"
        )

        try:

            _log(
                ">>> TORCH: "
                f"threads={torch.get_num_threads()}"
            )

        except Exception:
            pass

    except Exception as exc:

        _log(
            ">>> TORCH: environment check failed: "
            f"{exc}"
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
                encoding="utf-8",
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

    tmp.replace(
        path
    )


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
                f"[RAG] Could not read Docling "
                f"collection '{name}': {exc}"
            )

    return []


def _page(
    obj: Any,
) -> int | None:
    """
    Extract page number from different Docling object forms.

    Docling versions may expose page information differently,
    so this helper intentionally checks several possibilities.
    """

    # --------------------------------------------------------
    # Direct attributes
    # --------------------------------------------------------

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

            if hasattr(
                value,
                "page_no",
            ):

                value = value.page_no

            return int(
                value
            )

        except Exception:
            continue

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    provenance = getattr(
        obj,
        "prov",
        None,
    )

    if provenance is None:

        provenance = getattr(
            obj,
            "provenance",
            None,
        )

    if provenance is not None:

        try:

            prov_list = list(
                provenance
            )

            for prov in prov_list:

                for name in (
                    "page_no",
                    "page",
                    "page_number",
                ):

                    value = getattr(
                        prov,
                        name,
                        None,
                    )

                    if value is None:
                        continue

                    try:

                        return int(
                            value
                        )

                    except Exception:
                        continue

        except Exception:
            pass

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
            isinstance(
                value,
                str,
            )
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
    # TABLE STRUCTURE REQUIRED
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

    # --------------------------------------------------------
    # Release converter immediately.
    # --------------------------------------------------------

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

        text = _text(
            item
        )

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

    for label in labels:

        _log(
            ">>> VISUAL LABEL: "
            f"page={label.get('page')} "
            f"text={label.get('label')[:150]}"
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

    _log(
        f">>> VISUAL: Docling pictures found = "
        f"{len(pictures)}"
    )

    pages = {
        item["page"]
        for item in labels
        if item.get("page") is not None
    }

    if not pages:

        _log(
            ">>> VISUAL: No visual pages detected "
            "from label provenance"
        )

        # ----------------------------------------------------
        # We deliberately do NOT send every picture to Gemini.
        # That could become extremely expensive on a large PDF.
        # ----------------------------------------------------

        return []

    _log(
        f">>> VISUAL: Candidate visual pages = "
        f"{sorted(pages)}"
    )

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

    getter = getattr(
        picture,
        "get_image",
        None,
    )

    if callable(getter):

        return getter(
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
    from google.genai import types

    # --------------------------------------------------------
    # google-genai HttpOptions.timeout is specified in
    # milliseconds.
    # --------------------------------------------------------

    timeout_ms = (
        GEMINI_TIMEOUT_SECONDS
        * 1000
    )

    _log(
        ">>> GEMINI: Creating client "
        f"timeout={GEMINI_TIMEOUT_SECONDS}s"
    )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=timeout_ms,
        ),
    )


# ============================================================
# GEMINI FALLBACK GENERATOR
# ============================================================

def _generate_with_fallback(
    client: Any,
    contents: Any,
    stage: str,
) -> tuple[str, str]:

    """
    Try the configured Gemini models in order.

    Model order:

        1. gemini-3.5-flash-lite
        2. gemini-3.1-flash-lite
        3. gemini-3.5-flash
        4. gemini-3.7-flash

    Each model receives its own timeout.

    Returns:

        (response_text, model_name)

    Raises RuntimeError if every model fails.
    """

    last_error: Exception | None = None

    total_models = len(
        GEMINI_MODELS
    )

    for attempt, model_name in enumerate(
        GEMINI_MODELS,
        start=1,
    ):

        _log(
            f">>> {stage}: Trying Gemini "
            f"{attempt}/{total_models}: "
            f"{model_name}"
        )

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )

            text = (
                getattr(
                    response,
                    "text",
                    None,
                )
                or ""
            ).strip()

            if not text:

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            _log(
                f">>> {stage}: SUCCESS using "
                f"{model_name}"
            )

            return (
                text,
                model_name,
            )

        except Exception as exc:

            last_error = exc

            error_text = str(
                exc
            )

            _log(
                f">>> {stage}: FAILED "
                f"model={model_name}"
            )

            _log(
                f">>> {stage}: Error = "
                f"{error_text}"
            )

            # ------------------------------------------------
            # Continue automatically to the next model.
            # ------------------------------------------------

            if attempt < total_models:

                _log(
                    f">>> {stage}: Falling back "
                    f"to next Gemini model"
                )

                continue

            _log(
                f">>> {stage}: ALL Gemini models failed"
            )

    raise RuntimeError(
        "All configured Gemini models failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# GEMINI VISUAL DESCRIPTION
# ============================================================

def _gemini_describe(
    client: Any,
    image: Any,
    model_name: str,
    page: int | None,
) -> tuple[str, str]:

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

    # --------------------------------------------------------
    # model_name is retained in the function signature so the
    # existing application remains compatible.
    #
    # The actual fallback sequence is controlled centrally by
    # GEMINI_MODELS.
    # --------------------------------------------------------

    return _generate_with_fallback(
        client,
        [
            prompt,
            image,
        ],
        stage=f"VISUAL page={page}",
    )


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
                / max(
                    1,
                    total,
                )
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
                    f">>> VISUAL: No image available "
                    f"for page {page}"
                )

                continue

            text, used_model = _gemini_describe(
                client,
                image,
                model_name,
                page,
            )

            _log(
                f">>> VISUAL: Gemini description "
                f"complete for page {page} "
                f"using {used_model}"
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

            # ------------------------------------------------
            # IMPORTANT:
            #
            # A failed visual must NOT stop the entire PDF
            # processing pipeline.
            #
            # All Gemini fallback models have already been
            # attempted inside _generate_with_fallback().
            # ------------------------------------------------

            _log(
                f">>> VISUAL ERROR: page "
                f"{page}: {exc}"
            )

        finally:

            try:

                if image is not None:

                    close_fn = getattr(
                        image,
                        "close",
                        None,
                    )

                    if callable(close_fn):
                        close_fn()

            except Exception:
                pass

            image = None

            _cleanup_memory()

    del client

    _cleanup_memory()

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

                return str(
                    value
                )

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
                    "[RAG] Table markdown "
                    "fallback failed: "
                    f"{exc}"
                )

        except Exception as exc:

            _log(
                "[RAG] Table markdown "
                "export failed: "
                f"{exc}"
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

                return str(
                    value
                )

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
                "chunk_id":
                    chunk_id,

                "page_start":
                    page_start,

                "page_end":
                    page_end,

                "kind":
                    "text",

                "text":
                    text,

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
                    "chunk_id":
                        chunk_id,

                    "page_start":
                        element.get(
                            "page"
                        ),

                    "page_end":
                        element.get(
                            "page"
                        ),

                    "kind":
                        kind,

                    "text":
                        text,

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
        "================================================"
    )

    _log(
        ">>> EMBEDDING: Starting model load"
    )

    _log(
        f">>> EMBEDDING: Model = "
        f"{EMBEDDING_MODEL_NAME}"
    )

    _log(
        ">>> EMBEDDING: Device = CPU"
    )

    # --------------------------------------------------------
    # Torch diagnostic
    # --------------------------------------------------------

    _log_torch_environment()

    # --------------------------------------------------------
    # Import
    # --------------------------------------------------------

    _log(
        ">>> EMBEDDING: Importing "
        "SentenceTransformer"
    )

    from sentence_transformers import (
        SentenceTransformer,
    )

    # --------------------------------------------------------
    # Configure torch threads
    # --------------------------------------------------------

    try:

        import torch

        torch.set_num_threads(
            max(
                1,
                int(
                    os.getenv(
                        "TORCH_NUM_THREADS",
                        "1",
                    )
                ),
            )
        )

        try:

            torch.set_num_interop_threads(
                1
            )

        except Exception:
            pass

    except Exception as exc:

        _log(
            ">>> EMBEDDING: Torch thread "
            f"configuration skipped: {exc}"
        )

    _cleanup_memory()

    # --------------------------------------------------------
    # IMPORTANT
    #
    # low_cpu_mem_usage reduces the peak memory required
    # while constructing the model.
    #
    # This does NOT magically make BGE-M3 a small model.
    # It simply prevents avoidable duplicate model copies
    # during loading.
    # --------------------------------------------------------

    _log(
        ">>> EMBEDDING: Constructing "
        "SentenceTransformer"
    )

    model_kwargs = {
        "low_cpu_mem_usage": True,
    }

    try:

        model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu",
            model_kwargs=model_kwargs,
        )

    except TypeError as exc:

        # ----------------------------------------------------
        # Compatibility fallback for older versions of
        # sentence-transformers that do not accept the
        # low_cpu_mem_usage argument.
        # ----------------------------------------------------

        _log(
            ">>> EMBEDDING: "
            "low_cpu_mem_usage unsupported; "
            "retrying standard CPU load"
        )

        _log(
            f">>> EMBEDDING: Initial error = {exc}"
        )

        _cleanup_memory()

        model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu",
        )

    _log(
        ">>> EMBEDDING: Model successfully loaded"
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
        f">>> EMBEDDING: About to load "
        f"{EMBEDDING_MODEL_NAME} "
        f"for {len(chunks)} chunks"
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

        # ----------------------------------------------------
        # MODEL LOAD
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SAVE INDEX
        # ----------------------------------------------------

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

        "gemini_models":
            GEMINI_MODELS,

        "gemini_timeout_seconds":
            GEMINI_TIMEOUT_SECONDS,
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

    manifest = load_manifest(
        document_id
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

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Retrieval MUST use the same embedding model that
        # created the FAISS index.
        # ----------------------------------------------------

        retrieval_model = EMBEDDING_MODEL_NAME

        if manifest:

            stored_model = manifest.get(
                "embedding_model"
            )

            if stored_model:

                retrieval_model = stored_model

        if retrieval_model != EMBEDDING_MODEL_NAME:

            _log(
                ">>> RETRIEVAL: Manifest model differs "
                f"from configured model. "
                f"Using manifest model: {retrieval_model}"
            )

            # Temporarily load the exact model recorded
            # in the manifest.
            from sentence_transformers import (
                SentenceTransformer,
            )

            model = SentenceTransformer(
                retrieval_model,
                device="cpu",
                model_kwargs={
                    "low_cpu_mem_usage": True,
                },
            )

        else:

            _log(
                ">>> RETRIEVAL: Loading "
                f"{EMBEDDING_MODEL_NAME}"
            )

            model = _load_embedding_model()

        _log(
            ">>> RETRIEVAL: Encoding query"
        )

        query_vector = model.encode(
            [question],
            batch_size=1,
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
) -> tuple[str, str]:

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

    try:

        # ----------------------------------------------------
        # Same Gemini fallback logic as visual descriptions.
        # ----------------------------------------------------

        answer, used_model = _generate_with_fallback(
            client,
            prompt,
            stage="ANSWER",
        )

        _log(
            f">>> ANSWER: Generated using "
            f"{used_model}"
        )

        return (
            answer,
            used_model,
        )

    finally:

        del client

        _cleanup_memory()


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

            "sources":
                [],
        }

    context_parts = []

    for result in results:

        page_start = result.get(
            "page_start",
            result.get(
                "page",
                "?",
            ),
        )

        page_end = result.get(
            "page_end",
            result.get(
                "page",
                "?",
            ),
        )

        context_parts.append(
            f"""
[Pages {page_start}
-
{page_end}
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

    try:

        answer, used_model = _answer(
            question,
            context,
            api_key,
            answer_model,
        )

    except Exception as exc:

        # ----------------------------------------------------
        # Controlled error for Streamlit.
        # The actual fallback attempts and their errors have
        # already been logged by _generate_with_fallback().
        # ----------------------------------------------------

        _log(
            f">>> QA ERROR: {exc}"
        )

        answer = (
            "I could not generate an answer because "
            "all configured Gemini models failed. "
            "Please try again."
        )

        used_model = None

    sources = [
        {
            "page_start":
                result.get(
                    "page_start",
                    result.get(
                        "page"
                    ),
                ),

            "page_end":
                result.get(
                    "page_end",
                    result.get(
                        "page"
                    ),
                ),

            "kind":
                result.get(
                    "kind"
                ),

            "score":
                result["score"],
        }

        for result in results
    ]

    del results
    del context_parts
    del context

    _cleanup_memory()

    _log(
        ">>> QA: Answer generated"
    )

    if used_model:

        _log(
            f">>> QA: Gemini model used = "
            f"{used_model}"
        )

    return {
        "answer":
            answer,

        "sources":
            sources,
    }
