# Complex PDF RAG Streamlit — Optimized

This package is an optimized Streamlit implementation of the supplied working
notebook's core pipeline.

## Core pipeline preserved

Docling → label-aware visual selection → Gemini visual descriptions →
header/footer cleanup → structure-aware chunking → BGE-M3 embeddings →
normalized FAISS `IndexFlatIP` → top-k retrieval → grounded Gemini answer.

## Memory optimizations

- The complete RAG object is **not** stored in Streamlit session state.
- PDF/index artifacts are disk-backed under `storage/<document_id>/`.
- Embeddings are created in batches of 32 and immediately released.
- No full embedding matrix is retained after FAISS construction.
- Selected visual images are processed one at a time and released.
- The heavy Docling document is released before embedding.
- Re-uploading the same PDF reuses its SHA-256 document index.
- Q&A loads only the persisted index/chunks needed for that operation.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter a Gemini API key and select a Gemini model available to your API account.

## Important

This reduces application-level RAM pressure significantly, but Docling's own
PDF conversion can still require substantial RAM for very large or
image-heavy PDFs. Test a 400–500+ page file on the actual deployment machine
before production.

The example section is included and ready for a prebuilt example index.
