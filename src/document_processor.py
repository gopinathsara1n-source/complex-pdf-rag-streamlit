from pathlib import Path
import json
import re
import tempfile
import shutil
from collections import defaultdict

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from .config import (
    TARGET_CHARS,
    MAX_CHARS,
    MIN_STANDALONE_CHARS,
    VISUAL_MIN_WIDTH,
    VISUAL_MIN_HEIGHT,
)
from .visuals import generate_visual_descriptions


def clean_text(text):
    text = "" if text is None else str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_table_cells(table):
    cells = []
    data = table.get("data", {}) if isinstance(table, dict) else {}
    grid = data.get("grid", []) if isinstance(data, dict) else []

    if isinstance(grid, list):
        for row_idx, row in enumerate(grid):
            if not isinstance(row, list):
                continue
            for col_idx, cell in enumerate(row):
                if isinstance(cell, dict):
                    value = cell.get("text") or cell.get("value") or ""
                    cells.append({
                        "row": row_idx,
                        "col": col_idx,
                        "text": clean_text(value),
                    })
    return cells


def table_to_text(table):
    cells = extract_table_cells(table)
    if not cells:
        return clean_text(table.get("text") or table.get("content") or "")

    max_row = max(c["row"] for c in cells)
    max_col = max(c["col"] for c in cells)
    matrix = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    for cell in cells:
        matrix[cell["row"]][cell["col"]] = cell["text"]

    lines = []
    for row in matrix:
        if any(x.strip() for x in row):
            lines.append(" | ".join(x.strip() if x.strip() else "-" for x in row))

    return "\n".join(lines)


def get_page(item):
    try:
        if item.prov:
            return item.prov[0].page_no
    except Exception:
        pass
    return None


def get_text(item):
    for attr in ("text", "orig"):
        try:
            value = getattr(item, attr, None)
            if value:
                return clean_text(value)
        except Exception:
            pass
    return ""


def extract_visual_manifest(doc, image_dir: Path):
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for idx, picture in enumerate(doc.pictures, start=1):
        picture_id = f"picture_{idx:03d}"

        try:
            image = picture.get_image(doc)
            if image is None:
                continue

            page = get_page(picture)
            image_path = image_dir / f"{picture_id}.png"
            image.save(image_path)

            bbox = None
            try:
                if picture.prov:
                    bbox = picture.prov[0].bbox.model_dump()
            except Exception:
                bbox = None

            width = float(bbox.get("width", image.width)) if bbox else float(image.width)
            height = float(bbox.get("height", image.height)) if bbox else float(image.height)

            if width < VISUAL_MIN_WIDTH or height < VISUAL_MIN_HEIGHT:
                continue

            records.append({
                "picture_id": picture_id,
                "page": page,
                "image_path": str(image_path),
                "width": round(width, 2),
                "height": round(height, 2),
                "area": round(width * height, 2),
            })

        except Exception:
            continue

    return sorted(
        records,
        key=lambda x: (x.get("page") or 0, x["picture_id"])
    )


def build_canonical_document(doc, visual_results, docling_dict=None):
    visual_by_id = {
        x["picture_id"]: x
        for x in visual_results
        if x.get("status") == "success"
    }

    docling_dict = docling_dict or {}

    tables_by_ref = {
        item.get("self_ref"): item
        for item in docling_dict.get("tables", [])
        if item.get("self_ref")
    }

    picture_id_by_ref = {}

    for idx, picture in enumerate(doc.pictures, start=1):
        try:
            picture_id_by_ref[picture.self_ref] = f"picture_{idx:03d}"
        except Exception:
            pass

    elements = []

    for position, (item, _level) in enumerate(doc.iterate_items()):
        page = get_page(item)
        name = item.__class__.__name__.lower()

        if "picture" in name:
            try:
                picture_id = picture_id_by_ref.get(item.self_ref)
            except Exception:
                picture_id = None

            if picture_id and picture_id in visual_by_id:
                v = visual_by_id[picture_id]

                elements.append({
                    "element_id": f"element_{len(elements)+1:06d}",
                    "type": "visual",
                    "position": position,
                    "page": page,
                    "picture_id": picture_id,
                    "image_path": v.get("image_path"),
                    "visual_description": v.get("description"),
                    "source": {
                        "docling_ref": getattr(item, "self_ref", None),
                        "gemini_model": v.get("model"),
                    },
                })

            continue

        if "table" in name:
            raw = tables_by_ref.get(
                getattr(item, "self_ref", None),
                {}
            )

            elements.append({
                "element_id": f"element_{len(elements)+1:06d}",
                "type": "table",
                "position": position,
                "page": page,
                "text": table_to_text(raw),
                "source": {
                    "docling_ref": getattr(item, "self_ref", None)
                },
            })

            continue

        text = get_text(item)

        if not text:
            continue

        label = getattr(item, "label", "text")
        label_value = getattr(label, "value", label)

        elements.append({
            "element_id": f"element_{len(elements)+1:06d}",
            "type": "text",
            "position": position,
            "page": page,
            "text": text,
            "label": str(label_value),
            "source": {
                "docling_ref": getattr(item, "self_ref", None)
            },
        })

    return {
        "schema_version": "1.0",
        "document_name": getattr(doc, "name", None),
        "source": {
            "type": "pdf"
        },
        "statistics": {
            "total_elements": len(elements),
            "text_elements": sum(x["type"] == "text" for x in elements),
            "table_elements": sum(x["type"] == "table" for x in elements),
            "visual_elements": sum(x["type"] == "visual" for x in elements),
        },
        "elements": elements,
    }


def is_probable_page_number(text):
    text = clean_text(text)

    return bool(
        re.fullmatch(r"\d{1,4}", text)
        or re.fullmatch(r"[ivxlcdmIVXLCDM]{1,8}", text)
    )


def split_sentences(text):
    text = clean_text(text)

    if not text:
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    out = []

    for paragraph in paragraphs:
        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z₹0-9(\[])",
            paragraph.strip()
        )

        out.extend(
            p.strip()
            for p in parts
            if p.strip()
        )

    return out


def build_chunks(canonical):
    elements = [
        e
        for e in canonical["elements"]
        if not (
            e["type"] == "text"
            and e.get("label") in {
                "page_header",
                "page_footer"
            }
        )
    ]

    chunks = []

    current_parts = []
    current_ids = []
    current_types = []
    current_pages = []
    current_section = ""

    def current_text():
        return "\n\n".join(
            x
            for x in current_parts
            if x.strip()
        ).strip()

    def flush():
        nonlocal current_parts, current_ids, current_types, current_pages

        text = current_text()

        if not text:
            current_parts = []
            current_ids = []
            current_types = []
            current_pages = []
            return

        chunks.append({
            "chunk_id": f"chunk_{len(chunks):06d}",
            "page_start": min(current_pages) if current_pages else None,
            "page_end": max(current_pages) if current_pages else None,
            "section": current_section,
            "content": text,
            "element_ids": list(current_ids),
            "element_types": list(current_types),
            "has_table": "table" in current_types,
            "has_visual": "visual" in current_types,
        })

        current_parts = []
        current_ids = []
        current_types = []
        current_pages = []

    def add(element, content):
        content = clean_text(content)

        if not content:
            return

        current_parts.append(content)
        current_ids.append(element["element_id"])
        current_types.append(element["type"])

        if element.get("page") is not None:
            current_pages.append(element["page"])

    for element in elements:
        typ = element["type"]
        text = clean_text(element.get("text", ""))

        if typ == "text" and element.get("label") == "section_header":
            if current_text():
                flush()

            current_section = text
            current_parts = [text]
            current_ids = [element["element_id"]]
            current_types = ["text"]

            if element.get("page") is not None:
                current_pages = [element["page"]]

            continue

        if typ == "table":
            if current_text():
                flush()

            add(element, text)
            flush()
            continue

        if typ == "visual":
            if current_text():
                flush()

            visual_text = (
                f"[VISUAL]\n"
                f"Picture ID: {element.get('picture_id')}\n"
                f"Page: {element.get('page')}\n\n"
                f"{clean_text(element.get('visual_description', ''))}"
            )

            if element.get("visual_description"):
                add(element, visual_text)
                flush()

            continue

        if typ != "text" or not text or is_probable_page_number(text):
            continue

        for sentence in split_sentences(text):
            candidate = current_text()

            proposed = (
                f"{candidate}\n\n{sentence}"
                if candidate
                else sentence
            )

            if candidate and len(proposed) > TARGET_CHARS:
                flush()

            if len(sentence) <= MAX_CHARS:
                add(element, sentence)

            else:
                words = sentence.split()
                buffer = []

                for word in words:
                    test = " ".join(buffer + [word])

                    if len(test) <= MAX_CHARS:
                        buffer.append(word)

                    else:
                        if buffer:
                            add(element, " ".join(buffer))
                            flush()

                        buffer = [word]

                if buffer:
                    add(element, " ".join(buffer))

    flush()

    cleaned = []

    for chunk in chunks:
        if len(chunk["content"]) >= MIN_STANDALONE_CHARS:
            cleaned.append(chunk)
            continue

        if (
            cleaned
            and len(cleaned[-1]["content"]) + 2 + len(chunk["content"])
            <= MAX_CHARS
        ):
            prev = cleaned[-1]

            prev["content"] += "\n\n" + chunk["content"]

            prev["page_end"] = max(
                prev["page_end"] or 0,
                chunk["page_end"] or 0
            )

            prev["element_ids"].extend(
                chunk["element_ids"]
            )

            prev["element_types"].extend(
                chunk["element_types"]
            )

            prev["has_table"] |= chunk["has_table"]
            prev["has_visual"] |= chunk["has_visual"]

        else:
            cleaned.append(chunk)

    for i, chunk in enumerate(cleaned):
        chunk["chunk_id"] = f"chunk_{i:06d}"

    return cleaned


def process_pdf(pdf_path: Path, work_dir: Path, api_key: str, progress_callback=None):
    work_dir.mkdir(parents=True, exist_ok=True)

    image_dir = work_dir / "images"
    visual_json = work_dir / "visual_descriptions.json"

    if progress_callback:
        progress_callback(
            "✓ PDF loaded — starting Docling extraction",
            0.05
        )

    # Streamlit Cloud uses a read-only Python site-packages directory.
    # RapidOCR may otherwise try to save its model there.
    # Use a writable temporary directory for Docling model artifacts.
    docling_artifacts_dir = (
        Path(tempfile.gettempdir())
        / "docling_models"
    )

    docling_artifacts_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pipeline_options = PdfPipelineOptions(
        artifacts_path=str(docling_artifacts_dir)
    )

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

    result = converter.convert(str(pdf_path))
    doc = result.document

    if progress_callback:
        progress_callback(
            "✓ Content extracted with Docling",
            0.35
        )

    visual_manifest = extract_visual_manifest(
        doc,
        image_dir
    )

    if progress_callback:
        progress_callback(
            f"✓ Visual candidates extracted ({len(visual_manifest)})",
            0.45
        )

    visual_results = (
        generate_visual_descriptions(
            visual_manifest,
            visual_json,
            api_key
        )
        if visual_manifest
        else []
    )

    if progress_callback:
        progress_callback(
            "✓ Visual understanding completed",
            0.60
        )

    docling_dict = result.document.export_to_dict()

    canonical = build_canonical_document(
        doc,
        visual_results,
        docling_dict
    )

    if progress_callback:
        progress_callback(
            "✓ Canonical document created",
            0.70
        )

    chunks = build_chunks(canonical)

    if progress_callback:
        progress_callback(
            f"✓ Document chunked ({len(chunks):,} chunks)",
            0.80
        )

    return canonical, chunks, work_dir
