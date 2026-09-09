# Complex PDF RAG Document Assistant

A production-structured Streamlit application built from the validated project pipeline in `RAG_DRAFT_FINAL.ipynb`.

## What the project preserves

The implementation keeps the project's demonstrated architecture:

**PDF → Docling → canonical document → structure-aware chunks → BGE-M3 embeddings → FAISS Inner Product → retrieval → Gemini grounded answer**

The complex-PDF handling also preserves the visual branch:

**Docling picture extraction → visual filtering → Gemini visual description → visual retrieval chunk**

The working notebook explicitly uses Docling table structure extraction and picture extraction, BGE-M3 embeddings with normalized vectors, FAISS `IndexFlatIP`, and Gemini retry/fallback generation.

## Two application sections

1. **Example PDF**
   - Bundled pre-processed ANN Projects prospectus.
   - Uses the supplied validated chunks, embeddings, metadata and FAISS index.
   - No reprocessing is required.

2. **Upload Your Own PDF**
   - Validates the uploaded PDF through the processing pipeline.
   - Runs Docling extraction.
   - Preserves tables as intact retrieval units.
   - Generates descriptions for retained visuals with Gemini.
   - Builds the canonical representation.
   - Performs the final structure-aware chunking.
   - Creates BGE-M3 embeddings.
   - Builds FAISS.
   - Enables grounded Q&A with source/page display.

## Repository structure

```text
complex_pdf_rag_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_processor.py
│   ├── rag.py
│   ├── ui.py
│   └── visuals.py
└── data/
    └── example/
        ├── 1788150561832.pdf
        ├── chunks/
        │   └── chunks_final_v2.json
        ├── document/
        │   └── canonical_document.json
        └── vectors/
            ├── embeddings.npy
            ├── embedding_metadata.json
            └── faiss.index
```

## Setup

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gemini

Copy `.env.example` to `.env` and set:

```text
GEMINI_API_KEY=...
```

For Streamlit Community Cloud, use the app's Secrets configuration instead of committing `.env`.

The application reads `GEMINI_API_KEY` from the environment and also supports Streamlit `st.secrets`.

### 4. Run

```bash
streamlit run app.py
```

## Important deployment note

The bundled example page is designed to work without reprocessing the example PDF. The supplied FAISS index, embeddings, metadata and chunks are already included.

The upload page is intentionally compute-heavy because the original project is designed for complex PDFs. BGE-M3 model loading and Docling processing can require significant RAM/CPU. For production deployment, use a machine with sufficient resources.

## RAG behavior

The answer prompt is grounded strictly in retrieved context. If the retrieved context does not contain enough information, the assistant is instructed to say that the information is not available rather than inventing an answer.

Sources expose:
- page number/range
- section
- chunk ID
- relevance score
- retrieved content

The retrieval implementation uses normalized BGE-M3 embeddings with FAISS `IndexFlatIP`, which is equivalent to cosine similarity for normalized vectors.

## Why visual processing exists

Complex reports frequently contain information only in charts, diagrams, infographics and table-like graphics. The original notebook therefore extracted pictures with Docling and used Gemini visual understanding before adding those descriptions as independent retrieval content.

## Security

- No API keys are hard-coded.
- `.env` and Streamlit secrets are ignored by Git.
- Uploaded documents are processed in a temporary runtime directory.
- Do not commit private uploaded documents or generated runtime data.

## Validation checklist

Before deployment, verify:

- Example PDF loads.
- Example FAISS index loads.
- Example questions retrieve sources.
- Gemini answer generation works.
- PDF upload works.
- Docling extraction works.
- Tables are retained.
- Visual descriptions can be generated.
- BGE-M3 embeddings are generated.
- FAISS index is created.
- Uploaded-document Q&A returns page/source information.
- Missing/invalid Gemini configuration gives a clear error.
