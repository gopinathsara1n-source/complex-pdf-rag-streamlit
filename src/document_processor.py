from pathlib import Path
import json
import re
import tempfile
import shutil
from collections import defaultdict

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_table_cells(table_dict):
    cells = []

    if not isinstance(table_dict, dict):
        return cells

    for key in ("data", "table_cells", "cells"):
        value = table_dict.get(key)

        if isinstance(value, list):
            cells.extend(value)

    return cells


def table_to_text(table_dict):
    if not isinstance(table_dict, dict):
        return ""

    cells = extract_table_cells(table_dict)

    if not cells:
        return ""

    rows = defaultdict(list)

    for cell in cells:
        if not isinstance(cell, dict):
            continue

        text = (
            cell.get("text")
            or cell.get("content")
            or cell.get("value")
            or ""
        )

        text = clean_text(text)

        row_index = (
            cell.get("row_index")
            if cell.get("row_index") is not None
            else cell.get("row")
        )

        col_index = (
            cell.get("col_index")
            if cell.get("col_index") is not None
            else cell.get("column")
        )

        if row_index is None:
            row_index = 0

        if col_index is None:
            col_index = len(rows[row_index])

        rows[row_index].append((col_index, text))

    if not rows:
        return ""

    output = []

    for row_index in sorted(rows):
        row = sorted(rows[row_index], key=lambda x: x[0])
        values = [value for _, value in row]

        if any(values):
            output.append(" | ".join(values))

    return "\n".join(output).strip()


def get_page(item):
    try:
        prov = getattr(item, "prov", None)

        if prov:
            first = prov[0]

            page_no = getattr(first, "page_no", None)

            if page_no is not None:
                return int(page_no)

            if isinstance(first, dict):
                page_no = first.get("page_no")

                if page_no is not None:
                    return int(page_no)
    except Exception:
        pass

    return None


def get_text(item):
    for attr in ("text", "content", "caption"):

        try:
            value = getattr(item, attr, None)

            if value:
                return clean_text(value)
        except Exception:
            pass

    try:
        if hasattr(item, "export_to_dict"):
            data = item.export_to_dict()

            if isinstance(data, dict):
                for key in ("text", "content", "caption"):
                    value = data.get(key)

                    if value:
                        return clean_text(value)
    except Exception:
        pass

    return ""


def extract_visual_manifest(doc, image_dir):
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    try:
        items = doc.iterate_items()
    except Exception:
        items = []

    visual_index = 0

    for item in items:
        label = str(getattr(item, "label", "")).lower()

        if "picture" not in label and "image" not in label:
            continue

        visual_index += 1

        page = get_page(item)

        image_path = None

        try:
            image = item.get_image(doc)

            if image is not None:
                image_path = image_dir / f"visual_{visual_index:04d}.png"
                image.save(image_path)
        except Exception:
            image_path = None

        caption = get_text(item)

        manifest.append(
            {
                "visual_id": f"visual_{visual_index:04d}",
                "page": page,
                "image_path": str(image_path) if image_path else None,
                "caption": caption,
            }
        )

    return manifest


def build_canonical_document(doc):
    canonical = []

    try:
        document_dict = doc.export_to_dict()
    except Exception:
        document_dict = {}

    item_lookup = {}

    if isinstance(document_dict, dict):
        for item in document_dict.get("texts", []):
            if isinstance(item, dict):
                ref = item.get("self_ref")

                if ref:
                    item_lookup[ref] = item

        for item in document_dict.get("tables", []):
            if isinstance(item, dict):
                ref = item.get("self_ref")

                if ref:
                    item_lookup[ref] = item

        for item in document_dict.get("pictures", []):
            if isinstance(item, dict):
                ref = item.get("self_ref")

                if ref:
                    item_lookup[ref] = item

        for item in document_dict.get("groups", []):
            if isinstance(item, dict):
                ref = item.get("self_ref")

                if ref:
                    item_lookup[ref] = item

    try:
        items = doc.iterate_items()
    except Exception:
        items = []

    for index, item in enumerate(items):

        label = str(getattr(item, "label", "")).lower()
        page = get_page(item)

        text = ""

        if "table" in label:
            try:
                ref = getattr(item, "self_ref", None)

                if ref is None:
                    try:
                        item_dict = item.export_to_dict()
                        ref = item_dict.get("self_ref")
                    except Exception:
                        item_dict = None

                table_dict = item_lookup.get(ref)

                if table_dict is None:
                    try:
                        table_dict = item.export_to_dict()
                    except Exception:
                        table_dict = None

                text = table_to_text(table_dict)

            except Exception:
                text = ""

        elif "picture" in label or "image" in label:
            text = get_text(item)

        else:
            text = get_text(item)

        text = clean_text(text)

        if not text:
            continue

        canonical.append(
            {
                "element_id": index,
                "type": (
                    "table"
                    if "table" in label
                    else "visual"
                    if ("picture" in label or "image" in label)
                    else "text"
                ),
                "page": page,
                "text": text,
            }
        )

    return canonical


def is_probable_page_number(text):
    text = clean_text(text)

    if not text:
        return False

    if re.fullmatch(r"\d{1,4}", text):
        return True

    if re.fullmatch(r"(page\s*)?\d{1,4}", text.lower()):
        return True

    return False


def split_sentences(text):
    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9(])",
        text
    )

    sentences = [
        clean_text(sentence)
        for sentence in sentences
        if clean_text(sentence)
    ]

    return sentences


def build_chunks(
    canonical,
    max_chars=2200,
    overlap_chars=300,
):
    chunks = []

    current_parts = []
    current_length = 0

    def flush():
        nonlocal current_parts
        nonlocal current_length

        if not current_parts:
            return

        text = "\n\n".join(current_parts).strip()

        if text:
            first_page = None
            last_page = None
            element_ids = []

            for part in current_parts:
                metadata = part["metadata"]

                if metadata.get("page") is not None:
                    if first_page is None:
                        first_page = metadata["page"]

                    last_page = metadata["page"]

                element_ids.append(metadata.get("element_id"))

            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": text,
                    "metadata": {
                        "page": first_page,
                        "last_page": last_page,
                        "element_ids": element_ids,
                        "types": list(
                            dict.fromkeys(
                                p["metadata"].get("type")
                                for p in current_parts
                            )
                        ),
                    },
                }
            )

        if overlap_chars > 0 and text:
            current_parts = [
                {
                    "text": text[-overlap_chars:],
                    "metadata": {
                        "page": last_page,
                        "element_id": None,
                        "type": "overlap",
                    },
                }
            ]

            current_length = len(text[-overlap_chars:])
        else:
            current_parts = []
            current_length = 0

    for element in canonical:

        text = clean_text(element.get("text", ""))

        if not text:
            continue

        if is_probable_page_number(text):
            continue

        element_type = element.get("type", "text")
        page = element.get("page")
        element_id = element.get("element_id")

        sentences = split_sentences(text)

        if not sentences:
            sentences = [text]

        for sentence in sentences:

            if len(sentence) > max_chars:

                words = sentence.split()
                partial = ""

                for word in words:

                    candidate = (
                        f"{partial} {word}".strip()
                        if partial
                        else word
                    )

                    if len(candidate) > max_chars:

                        if partial:
                            part = {
                                "text": partial,
                                "metadata": {
                                    "page": page,
                                    "element_id": element_id,
                                    "type": element_type,
                                },
                            }

                            current_parts.append(part)
                            current_length += len(partial)

                            if current_length >= max_chars:
                                flush()

                        partial = word

                    else:
                        partial = candidate

                if partial:
                    part = {
                        "text": partial,
                        "metadata": {
                            "page": page,
                            "element_id": element_id,
                            "type": element_type,
                        },
                    }

                    current_parts.append(part)
                    current_length += len(partial)

                if current_length >= max_chars:
                    flush()

                continue

            part = {
                "text": sentence,
                "metadata": {
                    "page": page,
                    "element_id": element_id,
                    "type": element_type,
                },
            }

            if (
                current_length > 0
                and current_length + len(sentence) + 2 > max_chars
            ):
                flush()

            current_parts.append(part)
            current_length += len(sentence) + 2

    if current_parts:
        flush()

    return chunks


def process_pdf(
    pdf_path: Path,
    work_dir: Path,
    api_key: str,
    progress_callback=None,
):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    image_dir = work_dir / "images"
    visual_json = work_dir / "visual_descriptions.json"

    if progress_callback:
        progress_callback(
            "✓ PDF loaded — starting Docling extraction",
            0.05,
        )

    pipeline_options = PdfPipelineOptions()

    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_description = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    if progress_callback:
        progress_callback(
            "⏳ Docling is extracting text, tables and visuals...",
            0.10,
        )

    result = converter.convert(str(pdf_path))
    doc = result.document

    if progress_callback:
        progress_callback(
            "✓ Docling extraction completed",
            0.45,
        )

    visual_manifest = extract_visual_manifest(
        doc,
        image_dir,
    )

    if progress_callback:
        progress_callback(
            f"✓ Extracted {len(visual_manifest)} visual elements",
            0.55,
        )

    try:
        with open(
            visual_json,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                visual_manifest,
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass

    canonical = build_canonical_document(doc)

    if progress_callback:
        progress_callback(
            f"✓ Built canonical document with {len(canonical)} elements",
            0.65,
        )

    chunks = build_chunks(canonical)

    if progress_callback:
        progress_callback(
            f"✓ Created {len(chunks)} structure-aware chunks",
            0.80,
        )

    canonical_path = work_dir / "canonical_document.json"
    chunks_path = work_dir / "chunks.json"

    try:
        with open(
            canonical_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                canonical,
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(
            chunks_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass

    if progress_callback:
        progress_callback(
            "✓ PDF processing completed",
            1.0,
        )

    return canonical, chunks, work_dir
