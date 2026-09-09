from pathlib import Path
import json
import re
from collections import defaultdict

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


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
        text,
    )

    sentences = [
        clean_text(sentence)
        for sentence in sentences
        if clean_text(sentence)
    ]

    return sentences


# ============================================================
# DOCLING ITEM HELPERS
# ============================================================

def unpack_item(item_data):
    """
    Docling versions may return:
        item
    or:
        (item, level)

    Normalize both forms.
    """

    if isinstance(item_data, tuple):

        if len(item_data) >= 1:
            return item_data[0]

    return item_data


def get_item_label(item):
    try:
        label = getattr(item, "label", None)

        if label is not None:
            return str(label).lower()

    except Exception:
        pass

    try:
        if hasattr(item, "export_to_dict"):

            data = item.export_to_dict()

            if isinstance(data, dict):

                label = (
                    data.get("label")
                    or data.get("type")
                    or data.get("kind")
                    or ""
                )

                return str(label).lower()

    except Exception:
        pass

    return ""


def get_page(item):
    try:

        prov = getattr(
            item,
            "prov",
            None,
        )

        if prov:

            first = prov[0]

            page_no = getattr(
                first,
                "page_no",
                None,
            )

            if page_no is not None:
                return int(page_no)

            if isinstance(first, dict):

                page_no = first.get(
                    "page_no"
                )

                if page_no is not None:
                    return int(page_no)

    except Exception:
        pass

    return None


def get_self_ref(item):
    try:

        ref = getattr(
            item,
            "self_ref",
            None,
        )

        if ref:
            return ref

    except Exception:
        pass

    try:

        if hasattr(
            item,
            "export_to_dict",
        ):

            data = item.export_to_dict()

            if isinstance(data, dict):

                ref = data.get(
                    "self_ref"
                )

                if ref:
                    return ref

    except Exception:
        pass

    return None


def get_text(item):
    """
    Extract textual content from a Docling item.

    Uses multiple strategies because different
    Docling item classes expose their text differently.
    """

    # --------------------------------------------------------
    # Direct attributes
    # --------------------------------------------------------

    for attr in (
        "text",
        "content",
        "caption",
    ):

        try:

            value = getattr(
                item,
                attr,
                None,
            )

            if value:

                cleaned = clean_text(
                    value
                )

                if cleaned:
                    return cleaned

        except Exception:
            pass

    # --------------------------------------------------------
    # Docling text representation
    # --------------------------------------------------------

    try:

        if hasattr(
            item,
            "text",
        ):

            value = item.text

            if value:

                cleaned = clean_text(
                    value
                )

                if cleaned:
                    return cleaned

    except Exception:
        pass

    # --------------------------------------------------------
    # Exported dictionary
    # --------------------------------------------------------

    try:

        if hasattr(
            item,
            "export_to_dict",
        ):

            data = item.export_to_dict()

            if isinstance(
                data,
                dict,
            ):

                for key in (
                    "text",
                    "content",
                    "caption",
                ):

                    value = data.get(
                        key
                    )

                    if value:

                        cleaned = clean_text(
                            value
                        )

                        if cleaned:
                            return cleaned

    except Exception:
        pass

    return ""


# ============================================================
# TABLE HELPERS
# ============================================================

def extract_table_cells(table_dict):

    cells = []

    if not isinstance(
        table_dict,
        dict,
    ):
        return cells

    for key in (
        "data",
        "table_cells",
        "cells",
    ):

        value = table_dict.get(
            key
        )

        if isinstance(
            value,
            list,
        ):

            cells.extend(value)

    return cells


def table_to_text(table_dict):

    if not isinstance(
        table_dict,
        dict,
    ):
        return ""

    cells = extract_table_cells(
        table_dict
    )

    if not cells:
        return ""

    rows = defaultdict(list)

    for cell in cells:

        if not isinstance(
            cell,
            dict,
        ):
            continue

        text = (
            cell.get("text")
            or cell.get("content")
            or cell.get("value")
            or ""
        )

        text = clean_text(
            text
        )

        row_index = cell.get(
            "row_index"
        )

        if row_index is None:
            row_index = cell.get(
                "row"
            )

        col_index = cell.get(
            "col_index"
        )

        if col_index is None:
            col_index = cell.get(
                "column"
            )

        if row_index is None:
            row_index = 0

        if col_index is None:
            col_index = len(
                rows[row_index]
            )

        rows[row_index].append(
            (
                col_index,
                text,
            )
        )

    if not rows:
        return ""

    output = []

    for row_index in sorted(
        rows
    ):

        row = sorted(
            rows[row_index],
            key=lambda x: x[0],
        )

        values = [
            value
            for _, value in row
        ]

        if any(values):

            output.append(
                " | ".join(values)
            )

    return "\n".join(
        output
    ).strip()


# ============================================================
# DOCX/DOCLING EXPORT LOOKUP
# ============================================================

def build_item_lookup(doc):

    lookup = {}

    try:

        document_dict = (
            doc.export_to_dict()
        )

    except Exception:

        document_dict = {}

    if not isinstance(
        document_dict,
        dict,
    ):
        return lookup

    for key in (
        "texts",
        "tables",
        "pictures",
        "groups",
        "body",
    ):

        items = document_dict.get(
            key,
            []
        )

        if not isinstance(
            items,
            list,
        ):
            continue

        for item in items:

            if not isinstance(
                item,
                dict,
            ):
                continue

            ref = item.get(
                "self_ref"
            )

            if ref:
                lookup[ref] = item

    return lookup


# ============================================================
# ITERATE DOCLING ITEMS
# ============================================================

def iterate_doc_items(doc):

    """
    Compatibility wrapper for different Docling versions.

    Some versions return:
        item

    while others return:
        (item, level)
    """

    try:

        iterator = doc.iterate_items()

    except Exception:

        return

    for item_data in iterator:

        item = unpack_item(
            item_data
        )

        if item is None:
            continue

        yield item


# ============================================================
# VISUAL EXTRACTION
# ============================================================

def extract_visual_manifest(
    doc,
    image_dir,
):

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    visual_index = 0

    for item in iterate_doc_items(
        doc
    ):

        label = get_item_label(
            item
        )

        if (
            "picture" not in label
            and "image" not in label
        ):
            continue

        visual_index += 1

        page = get_page(
            item
        )

        image_path = None

        try:

            image = item.get_image(
                doc
            )

            if image is not None:

                image_path = (
                    image_dir
                    / f"visual_{visual_index:04d}.png"
                )

                image.save(
                    image_path
                )

        except Exception:

            image_path = None

        caption = get_text(
            item
        )

        manifest.append(
            {
                "visual_id":
                    f"visual_{visual_index:04d}",

                "page":
                    page,

                "image_path":
                    str(image_path)
                    if image_path
                    else None,

                "caption":
                    caption,
            }
        )

    return manifest


# ============================================================
# CANONICAL DOCUMENT
# ============================================================

def build_canonical_document(
    doc,
):

    canonical = []

    item_lookup = build_item_lookup(
        doc
    )

    index = 0

    for item in iterate_doc_items(
        doc
    ):

        label = get_item_label(
            item
        )

        page = get_page(
            item
        )

        element_type = "text"

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        if "table" in label:

            element_type = "table"

            text = ""

            try:

                ref = get_self_ref(
                    item
                )

                table_dict = (
                    item_lookup.get(
                        ref
                    )
                )

                if table_dict is None:

                    try:

                        table_dict = (
                            item.export_to_dict()
                        )

                    except Exception:

                        table_dict = None

                text = table_to_text(
                    table_dict
                )

            except Exception:

                text = ""

        # ----------------------------------------------------
        # PICTURE / IMAGE
        # ----------------------------------------------------

        elif (
            "picture" in label
            or "image" in label
        ):

            element_type = "visual"

            text = get_text(
                item
            )

        # ----------------------------------------------------
        # NORMAL TEXT
        # ----------------------------------------------------

        else:

            text = get_text(
                item
            )

        text = clean_text(
            text
        )

        if not text:
            continue

        canonical.append(
            {
                "element_id":
                    index,

                "type":
                    element_type,

                "page":
                    page,

                "text":
                    text,
            }
        )

        index += 1

    return canonical


# ============================================================
# CHUNKING
# ============================================================

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

        text = "\n\n".join(
            part["text"]
            for part in current_parts
        ).strip()

        if not text:

            current_parts = []
            current_length = 0

            return

        pages = []
        element_ids = []
        types = []

        for part in current_parts:

            metadata = part[
                "metadata"
            ]

            page = metadata.get(
                "page"
            )

            if page is not None:
                pages.append(
                    page
                )

            element_id = metadata.get(
                "element_id"
            )

            if element_id is not None:
                element_ids.append(
                    element_id
                )

            element_type = metadata.get(
                "type"
            )

            if element_type:
                types.append(
                    element_type
                )

        first_page = (
            min(pages)
            if pages
            else None
        )

        last_page = (
            max(pages)
            if pages
            else None
        )

        unique_types = list(
            dict.fromkeys(
                types
            )
        )

        has_table = (
            "table"
            in unique_types
        )

        has_visual = (
            "visual"
            in unique_types
        )

        # ----------------------------------------------------
        # SECTION
        # ----------------------------------------------------

        section = None

        for part in current_parts:

            part_section = (
                part["metadata"].get(
                    "section"
                )
            )

            if part_section:

                section = part_section

                break

        # ----------------------------------------------------
        # COMPATIBLE CHUNK FORMAT
        # ----------------------------------------------------

        chunks.append(
            {
                "chunk_id":
                    len(chunks),

                "content":
                    text,

                "page_start":
                    first_page,

                "page_end":
                    last_page,

                "section":
                    section,

                "element_ids":
                    element_ids,

                "element_types":
                    unique_types,

                "has_table":
                    has_table,

                "has_visual":
                    has_visual,
            }
        )

        # ----------------------------------------------------
        # OVERLAP
        # ----------------------------------------------------

        if overlap_chars > 0:

            overlap_text = (
                text[
                    -overlap_chars:
                ]
            )

            current_parts = [
                {
                    "text":
                        overlap_text,

                    "metadata":
                        {
                            "page":
                                last_page,

                            "element_id":
                                None,

                            "type":
                                "overlap",

                            "section":
                                section,
                        },
                }
            ]

            current_length = len(
                overlap_text
            )

        else:

            current_parts = []
            current_length = 0

    # ========================================================
    # PROCESS CANONICAL ELEMENTS
    # ========================================================

    for element in canonical:

        text = clean_text(
            element.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        if is_probable_page_number(
            text
        ):
            continue

        element_type = element.get(
            "type",
            "text",
        )

        page = element.get(
            "page"
        )

        element_id = element.get(
            "element_id"
        )

        section = element.get(
            "section"
        )

        sentences = split_sentences(
            text
        )

        if not sentences:
            sentences = [
                text
            ]

        # ====================================================
        # LONG SENTENCE
        # ====================================================

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
                                "text":
                                    partial,

                                "metadata":
                                    {
                                        "page":
                                            page,

                                        "element_id":
                                            element_id,

                                        "type":
                                            element_type,

                                        "section":
                                            section,
                                    },
                            }

                            current_parts.append(
                                part
                            )

                            current_length += (
                                len(partial)
                                + 2
                            )

                            if (
                                current_length
                                >= max_chars
                            ):
                                flush()

                        partial = word

                    else:

                        partial = candidate

                if partial:

                    part = {
                        "text":
                            partial,

                        "metadata":
                            {
                                "page":
                                    page,

                                "element_id":
                                    element_id,

                                "type":
                                    element_type,

                                "section":
                                    section,
                            },
                    }

                    current_parts.append(
                        part
                    )

                    current_length += (
                        len(partial)
                        + 2
                    )

                if current_length >= max_chars:
                    flush()

                continue

            # =================================================
            # NORMAL SENTENCE
            # =================================================

            additional_length = (
                len(sentence)
                + 2
            )

            if (
                current_length > 0
                and
                current_length
                + additional_length
                > max_chars
            ):

                flush()

            part = {
                "text":
                    sentence,

                "metadata":
                    {
                        "page":
                            page,

                        "element_id":
                            element_id,

                        "type":
                            element_type,

                        "section":
                            section,
                    },
            }

            current_parts.append(
                part
            )

            current_length += (
                additional_length
            )

    # ========================================================
    # FINAL CHUNK
    # ========================================================

    if current_parts:
        flush()

    return chunks


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    pdf_path: Path,
    work_dir: Path,
    api_key: str,
    progress_callback=None,
):

    pdf_path = Path(
        pdf_path
    )

    work_dir = Path(
        work_dir
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_dir = (
        work_dir / "images"
    )

    visual_json = (
        work_dir
        / "visual_descriptions.json"
    )

    if progress_callback:

        progress_callback(
            "✓ PDF loaded — starting Docling extraction",
            0.05,
        )

    # ========================================================
    # DOCLING PIPELINE
    # ========================================================

    pipeline_options = (
        PdfPipelineOptions()
    )

    # OCR disabled to avoid RapidOCR attempting
    # to write its model into read-only site-packages
    # on Streamlit Community Cloud.
    #
    # Digital PDFs will still use their embedded text.
    pipeline_options.do_ocr = False

    pipeline_options.do_table_structure = True

    pipeline_options.generate_picture_images = True

    pipeline_options.do_picture_description = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF:
                PdfFormatOption(
                    pipeline_options=
                        pipeline_options
                )
        }
    )

    if progress_callback:

        progress_callback(
            "⏳ Docling is extracting text, tables and visuals...",
            0.10,
        )

    # ========================================================
    # CONVERT PDF
    # ========================================================

    result = converter.convert(
        str(pdf_path)
    )

    doc = result.document

    if progress_callback:

        progress_callback(
            "✓ Docling extraction completed",
            0.45,
        )

    # ========================================================
    # VISUALS
    # ========================================================

    visual_manifest = (
        extract_visual_manifest(
            doc,
            image_dir,
        )
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

    # ========================================================
    # CANONICAL DOCUMENT
    # ========================================================

    canonical_elements = (
        build_canonical_document(
            doc
        )
    )

    if progress_callback:

        progress_callback(
            f"✓ Built canonical document with {len(canonical_elements)} elements",
            0.65,
        )

    # ========================================================
    # CHUNKS
    # ========================================================

    chunks = build_chunks(
        canonical_elements
    )

    if progress_callback:

        progress_callback(
            f"✓ Created {len(chunks)} structure-aware chunks",
            0.80,
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    text_count = sum(
        1
        for item in canonical_elements
        if item.get("type") == "text"
    )

    table_count = sum(
        1
        for item in canonical_elements
        if item.get("type") == "table"
    )

    visual_count = sum(
        1
        for item in canonical_elements
        if item.get("type") == "visual"
    )

    pages = [
        item.get("page")
        for item in canonical_elements
        if item.get("page") is not None
    ]

    statistics = {
        "total_elements":
            len(canonical_elements),

        "text_elements":
            text_count,

        "table_elements":
            table_count,

        "visual_elements":
            visual_count,

        "total_chunks":
            len(chunks),

        "pages_detected":
            len(set(pages)) if pages else 0,

        "first_page":
            min(pages) if pages else None,

        "last_page":
            max(pages) if pages else None,
    }

    # ========================================================
    # CANONICAL OUTPUT
    # ========================================================

    canonical_output = {
        "elements":
            canonical_elements,

        "statistics":
            statistics,
    }

    canonical_path = (
        work_dir
        / "canonical_document.json"
    )

    chunks_path = (
        work_dir
        / "chunks.json"
    )

    try:

        with open(
            canonical_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                canonical_output,
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

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if not chunks:

        raise ValueError(
            "Docling extracted the PDF, "
            "but no retrieval chunks were produced. "
            f"Canonical elements: {len(canonical_elements)}. "
            f"Statistics: {statistics}"
        )

    if progress_callback:

        progress_callback(
            "✓ PDF processing completed",
            1.0,
        )

    return (
        canonical_output,
        chunks,
        work_dir,
    )
