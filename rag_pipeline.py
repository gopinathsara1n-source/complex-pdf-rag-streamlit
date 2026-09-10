from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from functools import lru_cache
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


# ============================================================
# EMBEDDING MODEL
# ============================================================
#
# BGE-M3 remains the default model because this is the
# validated project embedding model.
#
# It can be overridden through:
#
# EMBEDDING_MODEL=BAAI/bge-m3
#
# ============================================================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
).strip()


# ------------------------------------------------------------
# Embedding batch size
#
# Keep this small on Streamlit Cloud.
#
# BGE-M3 is a large model and CPU memory is limited.
# ------------------------------------------------------------

EMBEDDING_BATCH_SIZE = max(
    1,
    int(
        os.getenv(
            "EMBEDDING_BATCH_SIZE",
            "2",
        )
    ),
)


# ------------------------------------------------------------
# Maximum sequence length used by SentenceTransformer.
#
# This does NOT change the RAG architecture.
#
# It limits unnecessary memory usage during inference.
#
# BGE-M3 supports long context, but most retrieved/document
# chunks in this project do not need an extremely large
# sequence length.
# ------------------------------------------------------------

EMBEDDING_MAX_SEQ_LENGTH = max(
    128,
    int(
        os.getenv(
            "EMBEDDING_MAX_SEQ_LENGTH",
            "512",
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
# Same fallback sequence is used for:
#
# 1. Visual / chart description
# 2. Final RAG answer
#
# ============================================================

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]


GEMINI_TIMEOUT_SECONDS = max(
    5,
    int(
        os.getenv(
            "GEMINI_TIMEOUT_SECONDS",
            "30",
        )
    ),
)


# ============================================================
# CPU / MEMORY CONTROL
# ============================================================

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

# Prevent unnecessary Hugging Face telemetry.
os.environ.setdefault(
    "HF_HUB_DISABLE_TELEMETRY",
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


# ============================================================
# TORCH DIAGNOSTICS
# ============================================================

def _log_torch_environment() -> None:
    """
    Diagnostic information for deployment debugging.
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

        try:

            _log(
                ">>> TORCH: "
                f"interop_threads="
                f"{torch.get_num_interop_threads()}"
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

        label_text = str(
            label.get(
                "label",
                "",
            )
        )

        _log(
            ">>> VISUAL LABEL: "
            f"page={label.get('page')} "
            f"text={label_text[:150]}"
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

            _log(
                f">>> {stage}: FAILED "
                f"model={model_name}"
            )

            _log(
                f">>> {stage}: Error = "
                f"{str(exc)}"
            )

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
#
# IMPORTANT DEPLOYMENT CHANGE
#
# The model is cached for the lifetime of the Python process.
#
# Why?
#
# Previous behavior:
#
#     Upload PDF
#       ↓
#     Load BGE-M3
#       ↓
#     Create index
#       ↓
#     Delete BGE-M3
#
# Then every question:
#
#     Question
#       ↓
#     Load BGE-M3 AGAIN
#       ↓
#     Encode query
#       ↓
#     Delete BGE-M3
#
# That causes unnecessary model loading and memory pressure.
#
# New behavior:
#
#     First request
#         ↓
#     Load BGE-M3
#         ↓
#     Keep one model instance
#
#     Later requests
#         ↓
#     Reuse same model
#
# This does NOT change embeddings, FAISS, chunking or retrieval.
# ============================================================


@lru_cache(
    maxsize=1
)
def _load_embedding_model_cached(
    model_name: str,
):

    _log(
        "================================================"
    )

    _log(
        ">>> EMBEDDING: Starting model load"
    )

    _log(
        f">>> EMBEDDING: Model = "
        f"{model_name}"
    )

    _log(
        ">>> EMBEDDING: Device = CPU"
    )

    _log_torch_environment()

    # --------------------------------------------------------
    # Import SentenceTransformer
    # --------------------------------------------------------

    _log(
        ">>> EMBEDDING: Importing "
        "SentenceTransformer"
    )

    from sentence_transformers import (
        SentenceTransformer,
    )

    # --------------------------------------------------------
    # Torch configuration
    # --------------------------------------------------------

    try:

        import torch

        num_threads = max(
            1,
            int(
                os.getenv(
                    "TORCH_NUM_THREADS",
                    "1",
                )
            ),
        )

        torch.set_num_threads(
            num_threads
        )

        try:

            torch.set_num_interop_threads(
                1
            )

        except Exception:
            pass

        _log(
            f">>> EMBEDDING: Torch threads "
            f"configured = {num_threads}"
        )

    except Exception as exc:

        _log(
            ">>> EMBEDDING: Torch thread "
            f"configuration skipped: {exc}"
        )

    _cleanup_memory()

    # --------------------------------------------------------
    # Construct model
    # --------------------------------------------------------

    _log(
        ">>> EMBEDDING: Constructing "
        "SentenceTransformer"
    )

    _log(
        f">>> EMBEDDING: "
        f"max_seq_length={EMBEDDING_MAX_SEQ_LENGTH}"
    )

    # --------------------------------------------------------
    # low_cpu_mem_usage=True is retained because BGE-M3 is
    # large and model construction can otherwise create a
    # higher temporary CPU memory peak.
    #
    # We explicitly use CPU.
    # --------------------------------------------------------

    model_kwargs = {
        "low_cpu_mem_usage": True,
    }

    try:

        model = SentenceTransformer(
            model_name,
            device="cpu",
            model_kwargs=model_kwargs,
        )

    except TypeError as exc:

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
            model_name,
            device="cpu",
        )

    # --------------------------------------------------------
    # Limit sequence length after model construction.
    #
    # This reduces inference memory consumption.
    # --------------------------------------------------------

    try:

        model.max_seq_length = (
            EMBEDDING_MAX_SEQ_LENGTH
        )

        _log(
            ">>> EMBEDDING: max_seq_length set to "
            f"{model.max_seq_length}"
        )

    except Exception as exc:

        _log(
            ">>> EMBEDDING: Could not set "
            f"max_seq_length: {exc}"
        )

    # --------------------------------------------------------
    # Evaluation mode.
    # --------------------------------------------------------

    try:

        model.eval()

    except Exception as exc:

        _log(
            ">>> EMBEDDING: model.eval() "
            f"failed: {exc}"
        )

    # --------------------------------------------------------
    # Disable gradient tracking globally for inference.
    # --------------------------------------------------------

    try:

        import torch

        torch.set_grad_enabled(
            False
        )

    except Exception:
        pass

    _cleanup_memory()

    _log(
        ">>> EMBEDDING: Model successfully loaded"
    )

    return model


def _load_embedding_model():

    return _load_embedding_model_cached(
        EMBEDDING_MODEL_NAME
    )


# ============================================================
# EMBEDDING ENCODE HELPER
# ============================================================

def _encode_texts(
    model: Any,
    texts: list[str],
    batch_size: int,
):

    """
    Encode text safely for CPU inference.

    The important difference from the previous implementation
    is that inference is explicitly performed without gradient
    tracking.
    """

    if not texts:

        raise ValueError(
            "No text supplied for embedding."
        )

    try:

        import torch

        with torch.inference_mode():

            vectors = model.encode(
                texts,
                batch_size=max(
                    1,
                    min(
                        batch_size,
                        len(texts),
                    ),
                ),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    except Exception as exc:

        _log(
            ">>> EMBEDDING: encode failed: "
            f"{exc}"
        )

        raise

    return vectors.astype(
        "float32"
    )


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

            vectors = _encode_texts(
                model,
                texts,
                batch_size,
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

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT delete the cached model here.
        #
        # The model is intentionally retained for Q&A.
        # ----------------------------------------------------

        model = None

        _cleanup_memory()

        _log(
            ">>> EMBEDDING: Index released; "
            "cached embedding model retained"
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

        "embedding_max_seq_length":
            EMBEDDING_MAX_SEQ_LENGTH,

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

    query_vector = None

    try:

        # ----------------------------------------------------
        # Retrieval MUST use exactly the same model that
        # generated the stored FAISS vectors.
        # ----------------------------------------------------

        retrieval_model_name = (
            EMBEDDING_MODEL_NAME
        )

        if manifest:

            stored_model = manifest.get(
                "embedding_model"
            )

            if stored_model:

                retrieval_model_name = (
                    stored_model
                )

        if (
            retrieval_model_name
            != EMBEDDING_MODEL_NAME
        ):

            _log(
                ">>> RETRIEVAL: Manifest model differs "
                f"from configured model. "
                f"Using manifest model: "
                f"{retrieval_model_name}"
            )

            # ------------------------------------------------
            # This should normally never happen.
            #
            # Use cached loading for the exact stored model.
            # ------------------------------------------------

            model = _load_embedding_model_cached(
                retrieval_model_name
            )

        else:

            _log(
                ">>> RETRIEVAL: Reusing "
                f"{EMBEDDING_MODEL_NAME}"
            )

            model = _load_embedding_model()

        _log(
            ">>> RETRIEVAL: Encoding query"
        )

        query_vector = _encode_texts(
            model,
            [question],
            batch_size=1,
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
