# Complex PDF RAG Streamlit — Gemini Embedding 2

A production-oriented Streamlit chatbot based on the tested notebook pipeline:

PDF → Docling → text/tables/pictures → Gemini visual understanding → canonical document
→ structure-aware chunks → Gemini Embedding 2 → FAISS → retrieval → Gemini answer

## Features

- Chatbot-style Streamlit UI
- Upload any PDF
- Process uploaded PDFs with Docling
- Extract text, tables and pictures
- Gemini visual descriptions for figures/charts
- Structure-aware chunks
- Gemini Embedding 2 (768 dimensions)
- FAISS cosine/IP retrieval
- Table-aware lexical boosting to improve queries such as "teledensity of Delhi"
- Conversational follow-up questions
- Source cards showing page, chunk and content type
- PDF SHA-256 cache
- Embedding checkpoint/resume
- Clear quota handling for Gemini free-tier limits

## Important Gemini Embedding 2 free-tier limitation

The standard `embed_content` call is intentionally made one chunk at a time. Sending multiple contents to
the standard method can produce one aggregated embedding rather than one vector per chunk. True Gemini
Batch API embedding is a separate paid path.

Therefore, this app throttles requests and checkpoints progress. For a large document with >1,000 chunks,
the free tier may require processing/resuming across days. For large production workloads, use a paid
embedding tier / Batch API or another embedding provider.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_KEY"
```

You can also set `GEMINI_API_KEY` as an environment variable.

## Example PDF

Put an example PDF in `data/example/` if you want the "Example PDF" page to be preloaded.

## Suggested deployment

For Streamlit Community Cloud, add `GEMINI_API_KEY` under App Settings → Secrets.
