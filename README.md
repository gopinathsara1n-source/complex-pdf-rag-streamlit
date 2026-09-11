# Chat with your PDF — Streamlit RAG app

A Streamlit front-end for your Gemini-embedding RAG pipeline (Docling ingestion →
Gemini Vision chart/image descriptions → structure-aware chunking → Gemini
Embedding → FAISS → Gemini answer generation). Upload a PDF, watch it get
indexed, then ask it questions in a chat UI.

## Why this replaces the BGE / sentence-transformers version

Your original BGE-M3 setup loaded a local sentence-transformer model into
memory. That works fine in Colab (lots of RAM, often a GPU), but Streamlit
Community Cloud's free containers typically cap out around 1 GB RAM, which is
usually not enough to load `sentence-transformers` + `torch` + the BGE
weights alongside Streamlit itself — hence the crash. This version calls the
**Gemini Embedding API** (`gemini-embedding-2`) instead, so no embedding
model is loaded locally at all — only lightweight Python packages.

Note: Docling itself still installs some ML dependencies (for layout/table
detection) and is the heaviest part of `requirements.txt`. If you hit memory
limits again on a free tier, that's the next thing to look at (e.g. deploy
on a host with more RAM, or a lighter Docling pipeline configuration).

## Files

- `app.py` — the Streamlit UI (upload, progress, chat).
- `rag_pipeline.py` — the pipeline logic, refactored from your notebook (no
  Colab/Drive dependencies; every stage reports progress back to the UI).
- `requirements.txt` — dependencies.

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the sidebar and either:
- paste your Gemini API key into the **Gemini API key** field, or
- set it once via `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

or as an environment variable:

```bash
export GEMINI_API_KEY="your-key-here"
```

Get a free key at https://aistudio.google.com/apikey.

## Using the app

1. Upload a PDF in the sidebar.
2. Click **Process document**. You'll see live progress through each stage:
   Docling ingestion → describing charts/images with Gemini Vision →
   chunking → embedding with Gemini → building the FAISS index.
3. Once it says "Ready", ask questions in the chat box at the bottom.
   Each answer shows the source chunks (with page numbers, section, and
   whether they came from a table or chart) in a collapsible panel.
4. Click **Clear document & chat** in the sidebar to start over with a new PDF.

## Notes on rate limits

The free Gemini tier enforces requests-per-minute and requests-per-day
limits on embeddings. `rag_pipeline.py` embeds chunks in small batches with
a short delay between requests and batches (see `GEMINI_EMBEDDING_*`
constants near the top of the file) — tune those if you're on a paid tier
and want faster processing, or need to back off further on a very restricted
free quota. Very large PDFs (many hundreds of chunks) will take a while to
embed because of this pacing; a progress bar keeps you posted.

## Deploying

Any host that can run a normal Streamlit app will work (Streamlit Community
Cloud, Render, Fly.io, a VM, etc.). Make sure:
- `GEMINI_API_KEY` is set as a secret/environment variable (or let users
  paste their own key in the sidebar, as the app supports).
- The host gives the container enough RAM/CPU for Docling's PDF parsing —
  a few hundred MB to ~1 GB depending on PDF complexity.
