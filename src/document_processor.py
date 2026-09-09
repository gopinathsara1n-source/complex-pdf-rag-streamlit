from pathlib import Path
import json
import re
from collections import defaultdict

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from docling.datamodel.base_models import (
    InputFormat,
)

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace(
        "\x00",
        " ",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def is_probable_page_number(text):

    text = clean_text(
        text
    )

    if not text:
        return False

    if re.fullmatch(
        r"\d{1,4}",
        text,
    ):

        return True

    if re.fullmatch(
        r"(page\s*)?\d{1,4}",
        text.lower(),
    ):

        return True

    return False


def split_sentences(text):

    text = clean_text(
        text
    )

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9(])",
        text,
    )

    return [
        clean_text(x)
        for x in sentences
        if clean_text(x)
    ]


# ============================================================
# DOCLING EXPORT
# ============================================================

def export_docling_dict(doc):

    try:

        data = doc.export_to_dict()

        if isinstance(
            data,
            dict,
        ):

            return data

    except Exception:
        pass

    return {}


# ============================================================
# REFERENCE LOOKUP
# ============================================================

def build_reference_lookup(
    document_dict,
):

    lookup = {}

    if not isinstance(
        document_dict,
        dict,
    ):

        return lookup

    for collection_name in (
        "texts",
        "tables",
        "pictures",
        "groups",
        "key_value_items",
        "form_items",
    ):

        collection = document_dict.get(
            collection_name,
            [],
        )

        if not isinstance(
            collection,
            list,
        ):

            continue

        for item in collection:

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
# REFERENCE TRAVERSAL
# ============================================================

def traverse_children(
    children,
    lookup,
    visited=None,
):

    if visited is None:
        visited = set()

    if not isinstance(
        children,
        list,
    ):

        return

    for child_ref in children:

        if not isinstance(
            child_ref,
            dict,
        ):

            continue

        ref = child_ref.get(
            "$ref"
        )

        if not ref:
            continue

        if ref in visited:
            continue

        item = lookup.get(
            ref
        )

        if item is None:
            continue

        # ----------------------------------------------------
        # GROUP
        # ----------------------------------------------------

        if ref.startswith(
            "#/groups/"
        ):

            visited.add(
                ref
            )

            group_children = item.get(
                "children",
                [],
            )

            yield from traverse_children(
                group_children,
                lookup,
                visited,
            )

        # ----------------------------------------------------
        # TEXT / TABLE / PICTURE
        # ----------------------------------------------------

        else:

            visited.add(
                ref
            )

            yield item


# ============================================================
# PAGE EXTRACTION
# ============================================================

def get_page_from_dict(
    item,
):

    if not isinstance(
        item,
        dict,
    ):

        return None

    prov = item.get(
        "prov",
        [],
    )

    if not isinstance(
        prov,
        list,
    ):

        return None

    for provenance in prov:

        if not isinstance(
            provenance,
            dict,
        ):

            continue

        page_no = provenance.get(
            "page_no"
        )

        if page_no is not None:

            try:

                return int(
                    page_no
                )

            except Exception:

                pass

    return None


# ============================================================
# TEXT EXTRACTION
# ============================================================

def get_text_from_dict(
    item,
):

    if not isinstance(
        item,
        dict,
    ):

        return ""

    for key in (
        "text",
        "orig",
        "content",
        "caption",
    ):

        value = item.get(
            key
        )

        if isinstance(
            value,
            str,
        ):

            value = clean_text(
                value
            )

            if value:
                return value

    captions = item.get(
        "captions"
    )

    if isinstance(
        captions,
        list,
    ):

        parts = []

        for caption in captions:

            if not isinstance(
                caption,
                dict,
            ):

                continue

            value = (
                caption.get("text")
                or caption.get("orig")
                or caption.get("content")
                or ""
            )

            value = clean_text(
                value
            )

            if value:
                parts.append(
                    value
                )

        if parts:

            return " ".join(
                parts
            )

    return ""


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table_cells(
    table_dict,
):

    if not isinstance(
        table_dict,
        dict,
    ):

        return []

    data = table_dict.get(
        "data"
    )

    if isinstance(
        data,
        dict,
    ):

        cells = data.get(
            "table_cells"
        )

        if isinstance(
            cells,
            list,
        ):

            return cells

    for key in (
        "table_cells",
        "cells",
    ):

        cells = table_dict.get(
            key
        )

        if isinstance(
            cells,
            list,
        ):

            return cells

    return []


def table_to_text(
    table_dict,
):

    cells = extract_table_cells(
        table_dict
    )

    if not cells:
        return ""

    rows = defaultdict(
        list
    )

    for cell in cells:

        if not isinstance(
            cell,
            dict,
        ):

            continue

        text = clean_text(
            cell.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        row_index = cell.get(
            "start_row_offset_idx"
        )

        if row_index is None:

            row_index = cell.get(
                "row_index"
            )

        if row_index is None:

            row_index = cell.get(
                "row"
            )

        col_index = cell.get(
            "start_col_offset_idx"
        )

        if col_index is None:

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
                " | ".join(
                    values
                )
            )

    return "\n".join(
        output
    ).strip()


# ============================================================
# CANONICAL DOCUMENT
# ============================================================

def build_canonical_document(
    document_dict,
):

    canonical = []

    if not isinstance(
        document_dict,
        dict,
    ):

        return canonical

    lookup = build_reference_lookup(
        document_dict
    )

    body = document_dict.get(
        "body",
        {},
    )

    if not isinstance(
        body,
        dict,
    ):

        return canonical

    body_children = body.get(
        "children",
        [],
    )

    visited = set()

    ordered_items = traverse_children(
        body_children,
        lookup,
        visited,
    )

    element_id = 0

    current_section = None

    for item in ordered_items:

        if not isinstance(
            item,
            dict,
        ):

            continue

        label = str(
            item.get(
                "label",
                "",
            )
        ).lower()

        page = get_page_from_dict(
            item
        )

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        if (
            label == "table"
            or "table" in label
        ):

            text = table_to_text(
                item
            )

            element_type = "table"

        # ----------------------------------------------------
        # PICTURE
        # ----------------------------------------------------

        elif (
            label == "picture"
            or "picture" in label
            or "image" in label
        ):

            text = get_text_from_dict(
                item
            )

            element_type = "visual"

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        else:

            text = get_text_from_dict(
                item
            )

            element_type = "text"

        text = clean_text(
            text
        )

        if not text:
            continue

        # ----------------------------------------------------
        # SECTION
        # ----------------------------------------------------

        if label in (
            "section_header",
            "title",
            "heading",
        ):

            current_section = text

        canonical.append(
            {
                "element_id":
                    element_id,

                "type":
                    element_type,

                "page":
                    page,

                "section":
                    current_section,

                "text":
                    text,
            }
        )

        element_id += 1

    return canonical


# ============================================================
# VISUAL MANIFEST
# ============================================================

def extract_visual_manifest(
    document_dict,
    image_dir,
):

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    pictures = document_dict.get(
        "pictures",
        [],
    )

    if not isinstance(
        pictures,
        list,
    ):

        return manifest

    for index, picture in enumerate(
        pictures,
        start=1,
    ):

        if not isinstance(
            picture,
            dict,
        ):

            continue

        page = get_page_from_dict(
            picture
        )

        caption = get_text_from_dict(
            picture
        )

        # ----------------------------------------------------
        # Picture files are currently not decoded.
        #
        # Visual descriptions are not part of the current
        # deployment-safe pipeline.
        # ----------------------------------------------------

        image_path = None

        manifest.append(
            {
                "visual_id":
                    f"visual_{index:04d}",

                "page":
                    page,

                "image_path":
                    image_path,

                "caption":
                    caption,
            }
        )

    return manifest


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

        content = "\n\n".join(
            part["text"]
            for part in current_parts
        ).strip()

        if not content:

            current_parts = []
            current_length = 0

            return

        pages = []
        element_ids = []
        element_types = []
        sections = []

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

                element_types.append(
                    element_type
                )

            section = metadata.get(
                "section"
            )

            if section:

                sections.append(
                    section
                )

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

        unique_types = list(
            dict.fromkeys(
                element_types
            )
        )

        section = (
            sections[-1]
            if sections
            else None
        )

        chunks.append(
            {
                "chunk_id":
                    len(chunks),

                "content":
                    content,

                "page_start":
                    page_start,

                "page_end":
                    page_end,

                "section":
                    section,

                "element_ids":
                    element_ids,

                "element_types":
                    unique_types,

                "has_table":
                    "table"
                    in unique_types,

                "has_visual":
                    "visual"
                    in unique_types,
            }
        )

        # ----------------------------------------------------
        # OVERLAP
        # ----------------------------------------------------

        if overlap_chars > 0:

            overlap_text = (
                content[
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
                                page_end,

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
    # ELEMENT LOOP
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

        for sentence in sentences:

            # ------------------------------------------------
            # LONG SENTENCE
            # ------------------------------------------------

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

            # ------------------------------------------------
            # NORMAL SENTENCE
            # ------------------------------------------------

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
    # FINAL FLUSH
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
        work_dir
        / "images"
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
    # DOCLING CONFIGURATION
    # ========================================================

    pipeline_options = (
        PdfPipelineOptions()
    )

    # --------------------------------------------------------
    # OCR disabled for Streamlit Cloud deployment.
    #
    # This prevents RapidOCR from attempting to write model
    # files into the read-only Python site-packages directory.
    # --------------------------------------------------------

    pipeline_options.do_ocr = False

    # --------------------------------------------------------
    # Keep table extraction.
    # --------------------------------------------------------

    pipeline_options.do_table_structure = True

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Picture image generation is disabled for now because
    # the current RAG pipeline does not consume the generated
    # image files.
    #
    # This reduces memory and disk usage on Streamlit Cloud.
    # --------------------------------------------------------

    pipeline_options.generate_picture_images = False

    pipeline_options.do_picture_description = False

    # ========================================================
    # DOCLING CONVERTER
    # ========================================================

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
            "⏳ Docling is extracting text and tables...",
            0.10,
        )

    # ========================================================
    # PDF CONVERSION
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
    # EXPORT DOCLING DOCUMENT
    # ========================================================

    document_dict = export_docling_dict(
        doc
    )

    if not document_dict:

        raise ValueError(
            "Docling conversion completed, "
            "but export_to_dict() returned no document data."
        )

    # ========================================================
    # RAW COUNTS
    # ========================================================

    raw_text_count = len(
        document_dict.get(
            "texts",
            [],
        )
    )

    raw_table_count = len(
        document_dict.get(
            "tables",
            [],
        )
    )

    raw_picture_count = len(
        document_dict.get(
            "pictures",
            [],
        )
    )

    if progress_callback:

        progress_callback(
            (
                "✓ Docling found "
                f"{raw_text_count:,} text elements, "
                f"{raw_table_count:,} tables and "
                f"{raw_picture_count:,} visuals"
            ),
            0.50,
        )

    # ========================================================
    # VISUAL MANIFEST
    # ========================================================

    visual_manifest = (
        extract_visual_manifest(
            document_dict,
            image_dir,
        )
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

    if progress_callback:

        progress_callback(
            (
                f"✓ Extracted "
                f"{len(visual_manifest)} visual elements"
            ),
            0.55,
        )

    # ========================================================
    # CANONICAL DOCUMENT
    # ========================================================

    canonical_elements = (
        build_canonical_document(
            document_dict
        )
    )

    if progress_callback:

        progress_callback(
            (
                "✓ Built canonical document with "
                f"{len(canonical_elements):,} elements"
            ),
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
            (
                "✓ Created "
                f"{len(chunks):,} structure-aware chunks"
            ),
            0.80,
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    text_count = sum(
        1
        for item in canonical_elements
        if item.get("type")
        == "text"
    )

    table_count = sum(
        1
        for item in canonical_elements
        if item.get("type")
        == "table"
    )

    visual_count = sum(
        1
        for item in canonical_elements
        if item.get("type")
        == "visual"
    )

    pages = [
        item.get("page")
        for item in canonical_elements
        if item.get("page")
        is not None
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

        "raw_text_elements":
            raw_text_count,

        "raw_table_elements":
            raw_table_count,

        "raw_visual_elements":
            raw_picture_count,

        "total_chunks":
            len(chunks),

        "pages_detected":
            len(set(pages))
            if pages
            else 0,

        "first_page":
            min(pages)
            if pages
            else None,

        "last_page":
            max(pages)
            if pages
            else None,
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

    docling_path = (
        work_dir
        / "docling_document.json"
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

        # ----------------------------------------------------
        # Raw Docling structure.
        # ----------------------------------------------------

        with open(
            docling_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                document_dict,
                f,
                ensure_ascii=False,
                indent=2,
            )

    except Exception:
        pass

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if not canonical_elements:

        raise ValueError(
            "Docling extracted the PDF, "
            "but no canonical elements were produced. "
            f"Raw text elements: {raw_text_count}. "
            f"Raw tables: {raw_table_count}. "
            f"Raw pictures: {raw_picture_count}."
        )

    if not chunks:

        raise ValueError(
            "Canonical elements were produced, "
            "but no retrieval chunks were created. "
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
