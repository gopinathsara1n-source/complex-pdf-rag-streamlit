import os, json, time, hashlib, re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
from google import genai
from google.genai import types

from config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, SAFE_RPM, DAILY_SAFE_LIMIT,
    MIN_REQUEST_INTERVAL, MAX_RETRIES, VISION_MODEL, ANSWER_MODEL
)

def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return genai.Client(api_key=key)

def pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def _retry_sleep(attempt: int):
    time.sleep(min(60, 2 ** attempt))

def embed_one(client, text: str, task: str = "document", last_call=[0.0]):
    # Keep requests below the tested RPM ceiling.
    wait = MIN_REQUEST_INTERVAL - (time.time() - last_call[0])
    if wait > 0:
        time.sleep(wait)

    if task == "query":
        content = f"task: search result | query: {text}"
    else:
        content = text

    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=content,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIM
                ),
            )
            last_call[0] = time.time()
            values = np.asarray(result.embeddings[0].values, dtype="float32")
            faiss.normalize_L2(values.reshape(1, -1))
            return values
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            _retry_sleep(attempt)

def describe_image(client, image_path: str, page: int):
    # The exact image understanding step follows the tested notebook idea:
    # describe charts/visuals rather than inventing page-wide summaries.
    from PIL import Image
    image = Image.open(image_path)
    prompt = f"""You are extracting a visual from page {page} of a business report.
Describe ONLY the chart, graph, diagram, or other visual shown.
Capture titles, axes, legends, entities, values, units, years, labels and comparisons.
Do not invent missing values. Return concise factual prose suitable for RAG."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=VISION_MODEL,
                contents=[prompt, image],
            )
            return response.text.strip()
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            _retry_sleep(attempt)

def chunk_text(text: str, max_chars: int = 2600, overlap: int = 300):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        piece = text[start:end]
        # Prefer natural boundary.
        if end < len(text):
            cut = max(piece.rfind(". "), piece.rfind("\n"), piece.rfind("; "))
            if cut > max_chars * 0.55:
                end = start + cut + 1
                piece = text[start:end]
        out.append(piece.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return out

def build_chunks_from_docling(conversion_result, visual_descriptions=None):
    visual_descriptions = visual_descriptions or {}
    chunks = []
    page_text = {}

    # Generic Docling export route. It deliberately keeps page/source metadata.
    doc = conversion_result.document
    markdown = doc.export_to_markdown()
    pages = markdown.split("\n---\n")

    for idx, page_text_value in enumerate(pages, start=1):
        page_text[idx] = page_text_value.strip()

    for page, txt in page_text.items():
        for j, piece in enumerate(chunk_text(txt)):
            chunks.append({
                "chunk_id": f"chunk_{len(chunks):06d}",
                "page": page,
                "type": "text",
                "section": "",
                "text": piece,
                "source": "docling",
            })

    for page, desc in visual_descriptions.items():
        if desc:
            chunks.append({
                "chunk_id": f"chunk_{len(chunks):06d}",
                "page": int(page),
                "type": "visual",
                "section": "Visual description",
                "text": desc,
                "source": "gemini_vision",
            })

    # Tables are preserved in markdown by Docling. Mark chunks containing
    # markdown table syntax so retrieval can give them a stronger score.
    for c in chunks:
        if "|" in c["text"] and ("---" in c["text"] or "| " in c["text"]):
            c["type"] = "table"

    return chunks

def format_document_for_embedding(chunk: dict) -> str:
    section = chunk.get("section") or chunk.get("type", "document")
    return f"title: {section} | text: {chunk['text']}"

def build_faiss(chunks: List[dict], embeddings: np.ndarray):
    matrix = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index

def lexical_terms(question: str):
    return [x.lower() for x in re.findall(r"[A-Za-z0-9][A-Za-z0-9._%-]*", question)]

def lexical_score(question: str, chunk: dict):
    q = lexical_terms(question)
    text = chunk.get("text", "").lower()
    if not q:
        return 0.0
    hits = sum(1 for term in q if term in text)
    # Tables receive a modest boost for entity/value lookup questions.
    boost = 0.08 if chunk.get("type") == "table" and hits else 0.0
    return min(0.20, 0.20 * hits / len(q)) + boost

def retrieve(question, chunks, index, embeddings_client, top_k=6):
    qv = embed_one(embeddings_client, question, task="query")
    scores, ids = index.search(qv.reshape(1, -1), min(len(chunks), max(20, top_k * 4)))
    candidates = []
    for semantic, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        c = dict(chunks[int(idx)])
        lex = lexical_score(question, c)
        c["_semantic"] = float(semantic)
        c["_score"] = float(0.82 * semantic + 0.18 * lex)
        candidates.append(c)
    candidates.sort(key=lambda x: x["_score"], reverse=True)
    return candidates[:top_k]

def answer_question(client, question, retrieved, history):
    context = "\n\n".join(
        f"[Source {i+1} | page {c.get('page')} | {c.get('type')} | {c.get('chunk_id')}]\n{c.get('text','')}"
        for i, c in enumerate(retrieved)
    )
    recent = history[-6:] if history else []
    hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    prompt = f"""You are a precise document assistant.
Answer the user's question ONLY from the supplied document context.
Use conversation history only to resolve follow-up references.
If the context does not support an answer, say that the information was not found.
Do not invent values. For numeric questions, preserve units and distinguish years/periods.
When useful, cite sources inline as [Page X].

Conversation history:
{hist_text}

Document context:
{context}

User question:
{question}
"""
    response = client.models.generate_content(model=ANSWER_MODEL, contents=prompt)
    return response.text.strip()

def process_pdf(pdf_path: str, work_dir: Path, progress_cb=None):
    """Run the tested Docling → visual → chunks pipeline."""
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    work_dir.mkdir(parents=True, exist_ok=True)
    images_dir = work_dir / "images"
    images_dir.mkdir(exist_ok=True)

    options = PdfPipelineOptions()
    options.do_table_structure = True
    options.generate_picture_images = True
    options.do_picture_description = False

    if progress_cb: progress_cb("Running Docling extraction…", 0.05)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    if progress_cb: progress_cb("Docling extraction complete. Preparing visual descriptions…", 0.45)

    client = get_client()
    visual_descriptions = {}

    # Try to discover pictures from Docling's exported picture objects.
    try:
        pictures = list(result.document.pictures)
    except Exception:
        pictures = []

    for i, picture in enumerate(pictures):
        page = getattr(getattr(picture, "prov", None), "page_no", None)
        if page is None:
            page = getattr(picture, "page_no", i + 1)
        try:
            img = picture.get_image(result.document)
        except Exception:
            img = None
        if img is None:
            continue
        img_path = images_dir / f"page_{page}_picture_{i}.png"
        img.save(img_path)
        try:
            visual_descriptions[int(page)] = describe_image(client, str(img_path), int(page))
        except Exception as exc:
            visual_descriptions[int(page)] = f"Visual extraction failed on page {page}: {exc}"

    if progress_cb: progress_cb("Building structure-aware chunks…", 0.65)
    chunks = build_chunks_from_docling(result, visual_descriptions)
    (work_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    if progress_cb: progress_cb(f"Embedding {len(chunks)} chunks with Gemini Embedding 2…", 0.75)
    vectors = []
    checkpoint = work_dir / "embedding_checkpoint.npz"
    completed = {}
    if checkpoint.exists():
        try:
            z = np.load(checkpoint, allow_pickle=True)
            completed = {int(k): z["vectors"][i] for i, k in enumerate(z["indices"].tolist())}
        except Exception:
            completed = {}

    remaining = [i for i in range(len(chunks)) if i not in completed]
    if len(remaining) > DAILY_SAFE_LIMIT:
        remaining_today = remaining[:DAILY_SAFE_LIMIT]
        quota_message = f"Only {DAILY_SAFE_LIMIT} embedding requests are scheduled in this run; resume later."
    else:
        remaining_today = remaining
        quota_message = ""

    for n, idx in enumerate(remaining_today, start=1):
        vectors_for_chunk = embed_one(client, format_document_for_embedding(chunks[idx]))
        completed[idx] = vectors_for_chunk
        if n % 25 == 0 or n == len(remaining_today):
            inds = np.array(sorted(completed), dtype=np.int32)
            vals = np.stack([completed[i] for i in inds]).astype("float32")
            np.savez_compressed(checkpoint, indices=inds, vectors=vals)
        if progress_cb:
            frac = 0.75 + 0.20 * (n / max(1, len(remaining_today)))
            progress_cb(f"Embedding {n}/{len(remaining_today)}", frac)

    missing = [i for i in range(len(chunks)) if i not in completed]
    if missing:
        raise RuntimeError(f"Embedding quota reached before completion. {len(missing)} chunks remain. {quota_message}")

    vectors = np.stack([completed[i] for i in range(len(chunks))]).astype("float32")
    np.save(work_dir / "embeddings.npy", vectors)
    index = build_faiss(chunks, vectors)
    faiss.write_index(index, str(work_dir / "faiss.index"))
    (work_dir / "meta.json").write_text(json.dumps({
        "embedding_model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIM,
        "chunks": len(chunks),
        "quota_note": quota_message,
    }, indent=2), encoding="utf-8")
    if progress_cb: progress_cb("Processing complete.", 1.0)
    return {"chunks": chunks, "index": index, "client": client, "quota_note": quota_message}

def load_index(work_dir: Path):
    chunks = json.loads((work_dir / "chunks.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(work_dir / "faiss.index"))
    return chunks, index, get_client()
