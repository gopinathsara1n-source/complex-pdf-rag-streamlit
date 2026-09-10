
from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import Callable, Any

import numpy as np


INDEX_ROOT = Path(os.getenv("RAG_INDEX_ROOT", "storage"))
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

Progress = Callable[[float, str], None] | None


def _progress(cb, frac, message):
    if cb:
        cb(float(frac), message)


def _doc_dir(document_id):
    return INDEX_ROOT / document_id


def load_manifest(document_id):
    p = _doc_dir(document_id) / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _iter_collection(doc, names):
    for name in names:
        value = getattr(doc, name, None)

        if value is not None:
            try:
                return list(value)
            except TypeError:
                pass

    return []


def _page(obj):
    for name in ("page_no", "page", "page_number"):
        value = getattr(obj, name, None)

        if value is not None:
            try:
                return int(value)
            except Exception:
                return None

    return None


def _text(obj):
    for name in ("text", "caption", "label"):
        value = getattr(obj, name, None)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _convert_pdf(pdf_path):
    """
    Convert PDF using Docling.

    OCR is intentionally disabled because this project does not require
    OCR/RapidOCR for its target PDFs.

    Table structure extraction remains enabled.
    Picture extraction remains enabled so that charts/figures can be
    passed to the Gemini visual-description stage.
    """

    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    opts = PdfPipelineOptions()

    # OCR is not required for this project.
    # This prevents Docling from initializing RapidOCR and downloading
    # PP-OCR model files.
    opts.do_ocr = False

    # Keep table extraction enabled.
    opts.do_table_structure = True

    # Keep picture extraction enabled for chart/figure processing.
    opts.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=opts
            )
        }
    )

    return converter.convert(pdf_path)


def _find_visual_labels(doc):
    labels = []

    for item in _iter_collection(
        doc,
        ["texts", "text_items", "texts_and_tables"],
    ):
        txt = _text(item)

        if re.search(
            r"\b(chart|figure|fig\.?)\b",
            txt,
            re.I,
        ):
            labels.append(
                {
                    "page": _page(item),
                    "label": txt,
                }
            )

    return labels


def _selected_pictures(doc, labels):
    pictures = _iter_collection(
        doc,
        ["pictures", "picture_items"],
    )

    pages = {
        x["page"]
        for x in labels
        if x.get("page") is not None
    }

    if not pages:
        return []

    return [
        (picture, _page(picture))
        for picture in pictures
        if _page(picture) in pages
    ]


def _picture_image(picture, doc):
    if hasattr(picture, "get_image"):
        return picture.get_image(doc)

    return getattr(picture, "image", None)


def _gemini_describe(image, api_key, model_name, page):
    from google import genai

    client = genai.Client(api_key=api_key)

    prompt = f"""
Describe ONLY visible information in this PDF chart/figure for RAG retrieval.

Page: {page}

Include:
- title
- type of visual
- labels
- legend
- units
- exact readable numbers
- dates
- comparisons
- trends
- forecasts/annotations
- visible source

Do not invent values.
Do not describe unrelated page content.

Return concise factual prose.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, image],
    )

    return (response.text or "").strip()


def _visual_pass(doc, api_key, model_name, cb):
    labels = _find_visual_labels(doc)
    selected = _selected_pictures(doc, labels)

    descriptions = []

    total = max(1, len(selected))

    for i, (picture, page) in enumerate(selected, 1):

        _progress(
            cb,
            0.35 + 0.17 * ((i - 1) / total),
            f"Describing visual {i}/{total}…",
        )

        image = None

        try:
            image = _picture_image(picture, doc)

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
        f"Visual pass complete: {len(descriptions)} descriptions.",
    )

    return descriptions


def _clean_text(text):
    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    if len(lines) < 6:
        return "\n".join(lines)

    counts = {}

    for line in lines:
        if len(line) <= 180:
            counts[line] = counts.get(line, 0) + 1

    repeated = {
        k
        for k, v in counts.items()
        if v >= 3
    }

    return "\n".join(
        x
        for x in lines
        if x not in repeated
    )


def _table_text(table, doc):
    fn = getattr(table, "export_to_markdown", None)

    if callable(fn):
        try:
            return fn(doc=doc)
        except TypeError:
            try:
                return fn(doc)
            except Exception:
                pass
        except Exception:
            pass

    fn = getattr(table, "to_markdown", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass

    return str(table)


def _canonical_elements(doc):
    elements = []

    # -----------------------------
    # Text elements
    # -----------------------------

    for item in _iter_collection(
        doc,
        ["texts", "text_items"],
    ):
        txt = _clean_text(_text(item))

        if txt:
            elements.append(
                {
                    "kind": "text",
                    "page": _page(item),
                    "text": txt,
                }
            )

    # -----------------------------
    # Table elements
    # -----------------------------

    for table in _iter_collection(
        doc,
        ["tables", "table_items"],
    ):
        txt = _table_text(table, doc).strip()
        if txt:
            elements.append(
                {
                    "kind": "table",
                    "page": _page(table),
                    "text": txt,
                }
            )

    return elements


def _chunk(
    elements,
    target=1800,
    maximum=3200,
):
    result = []

    buffer = []
    length = 0

    def flush():
        nonlocal buffer, length

        if buffer:
            result.append(
                {
                    "kind": "text",
                    "page": buffer[0].get("page"),
                    "text": "\n\n".join(
                        x["text"]
                        for x in buffer
                    ).strip(),
                }
            )

        buffer = []
        length = 0

    for el in elements:

        text = el.get("text", "").strip()
        kind = el.get("kind")

        if not text:
            continue

        # Tables and visuals are atomic.
        if kind in {"table", "visual"}:
            flush()

            result.append(
                {
                    "kind": kind,
                    "page": el.get("page"),
                    "text": text,
                }
            )

            continue

        # Prevent normal text chunks from exceeding maximum size.
        if (
            buffer
            and length + len(text) + 2 > maximum
        ):
            flush()

        buffer.append(el)

        length += len(text) + 2

        # Target chunk size reached.
        if length >= target:
            flush()

    flush()

    return result


def _embed_index(
    chunks,
    index_path,
    cb,
    batch_size=32,
):
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "BAAI/bge-m3"
    )

    index = None

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):
        batch = chunks[
            start:start + batch_size
        ]

        vectors = model.encode(
            [
                x["text"]
                for x in batch
            ],
            batch_size=min(
                batch_size,
                len(batch),
            ),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        if index is None:
            index = faiss.IndexFlatIP(
                vectors.shape[1]
            )

        index.add(vectors)

        del vectors
        gc.collect()

        _progress(
            cb,
            0.62
            + 0.20
            * (
                (start + len(batch))
                / len(chunks)
            ),
            f"Embedding chunks "
            f"{start + len(batch)}/{len(chunks)}…",
        )

    if index is None:
        raise RuntimeError(
            "No chunks were produced."
        )

    faiss.write_index(
        index,
        str(index_path),
    )

    del index
    del model
    gc.collect()


def build_or_load_document(
    pdf_bytes,
    source_name,
    document_id,
    api_key,
    visual_model,
    progress_callback=None,
):
    # ---------------------------------
    # Reuse existing processed document
    # ---------------------------------

    existing = load_manifest(
        document_id
    )

    if (
        existing
        and (
            _doc_dir(document_id)
            / "index.faiss"
        ).exists()
    ):
        _progress(
            progress_callback,
            1.0,
            "Existing processed index loaded.",
        )

        return existing

    # ---------------------------------
    # Prepare document directory
    # ---------------------------------

    doc_dir = _doc_dir(document_id)
    doc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = doc_dir / "source.pdf"

    pdf_path.write_bytes(pdf_bytes)

    # ---------------------------------
    # Docling conversion
    # ---------------------------------

    _progress(
        progress_callback,
        0.03,
        "Converting PDF with Docling…",
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

    # ---------------------------------
    # Visual description pass
    # ---------------------------------

    visual_descriptions = _visual_pass(
        doc,
        api_key,
        visual_model,
        progress_callback,
    )

    # ---------------------------------
    # Canonical document
    # ---------------------------------

    _progress(
        progress_callback,
        0.54,
        "Building canonical structure…",
    )

    elements = _canonical_elements(
        doc
    )

    elements.extend(
        visual_descriptions
    )

    # ---------------------------------
    # Release heavy Docling graph
    # before embedding
    # ---------------------------------

    del result
    del doc
    gc.collect()

    # ---------------------------------
    # Structure-aware chunking
    # ---------------------------------

    chunks = _chunk(elements)

    del elements
    gc.collect()

    if not chunks:
        raise RuntimeError(
            "No usable text/table/visual "
            "content was extracted."
        )

    # ---------------------------------
    # Save chunks
    # ---------------------------------

    _atomic_json(
        doc_dir / "chunks.json",
        chunks,
    )

    _progress(
        progress_callback,
        0.60,
        f"Prepared {len(chunks)} chunks.",
    )

    # ---------------------------------
    # Create FAISS index
    # ---------------------------------

    _embed_index(
        chunks,
        doc_dir / "index.faiss",
        progress_callback,
    )

    # ---------------------------------
    # Manifest
    # ---------------------------------

    manifest = {
        "document_id": document_id,
        "source_name": source_name,
        "pages": pages,
        "num_chunks": len(chunks),
        "embedding_model": "BAAI/bge-m3",
        "index_type": (
            "FAISS IndexFlatIP + "
            "normalized embeddings"
        ),
        "chunking": (
            "target 1800 / max 3200; "
            "tables and visuals atomic"
        ),
        "visual_descriptions": len(
            visual_descriptions
        ),
        "ocr": False,
        "table_structure": True,
        "picture_images": True,
        "notebook_aligned": True,
    }

    _atomic_json(
        doc_dir / "manifest.json",
        manifest,
    )

    # ---------------------------------
    # Memory cleanup
    # ---------------------------------

    del chunks
    del visual_descriptions
    gc.collect()

    _progress(
        progress_callback,
        1.0,
        "Index saved. Ready for Q&A.",
    )

    return manifest


def _retrieve(
    document_id,
    question,
    top_k,
):
    import faiss
    from sentence_transformers import SentenceTransformer

    doc_dir = _doc_dir(
        document_id
    )

    chunks = json.loads(
        (
            doc_dir
            / "chunks.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    index = faiss.read_index(
        str(
            doc_dir
            / "index.faiss"
        )
    )

    model = SentenceTransformer(
        "BAAI/bge-m3"
    )

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, ids = index.search(
        query_vector,
        min(
            top_k,
            index.ntotal,
        ),
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

            results.append(item)

    del query_vector
    del model
    del index
    del chunks

    gc.collect()

    return results


def _answer(
    question,
    context,
    api_key,
    model_name,
):
    from google import genai

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
Answer using ONLY the retrieved PDF context below.

Do not use outside knowledge.

If the context is insufficient, say so.

Preserve numbers, dates, units, names
and table relationships exactly.

Use visual descriptions only for
explicitly visible information.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Give a concise direct answer and mention
page numbers when useful.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return (
        response.text or ""
    ).strip()


def answer_question(
    document_id,
    question,
    api_key,
    answer_model,
    top_k=5,
):
    if not api_key:
        raise ValueError(
            "Gemini API key is required."
        )

    results = _retrieve(
        document_id,
        question,
        top_k,
    )

    context = "\n\n---\n\n".join(
        f"[Page {r.get('page', '?')} "
        f"| {r.get('kind')} "
        f"| score {r['score']:.4f}]\n"
        f"{r['text']}"
        for r in results
    )

    sources = [
        {
            "page": r.get("page"),
            "kind": r.get("kind"),
            "score": r["score"],
        }
        for r in results
    ]

    answer = _answer(
        question,
        context,
        api_key,
        answer_model,
    )

    del results
    gc.collect()

    return {
        "answer": answer,
        "sources": sources,
    }
