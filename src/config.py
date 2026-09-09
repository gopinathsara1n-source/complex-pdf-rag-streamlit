from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EXAMPLE_DIR = DATA_DIR / "example"

APP_TITLE = "Complex PDF RAG Document Assistant"

# Validated project configuration from the working notebook.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

TARGET_CHARS = int(os.getenv("TARGET_CHARS", "1800"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "3200"))
MIN_STANDALONE_CHARS = int(os.getenv("MIN_STANDALONE_CHARS", "150"))

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_TEST_TOP_K = int(os.getenv("RETRIEVAL_TEST_TOP_K", "10"))

# Preserved from the notebook's final Gemini generation block.
GEMINI_MODELS = [
    os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.5-flash-lite"),
    os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite"),
]

# Visual understanding models preserved from the notebook.
GEMINI_VISUAL_MODELS = [
    os.getenv("GEMINI_VISUAL_MODEL_1", "gemini-3.5-flash-lite"),
    os.getenv("GEMINI_VISUAL_MODEL_2", "gemini-3.1-flash-lite"),
    os.getenv("GEMINI_VISUAL_MODEL_3", "gemini-3.5-flash"),
    os.getenv("GEMINI_VISUAL_MODEL_4", "gemini-3.7-flash"),
]

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
INITIAL_RETRY_DELAY = float(os.getenv("INITIAL_RETRY_DELAY", "2"))
MAX_RETRY_DELAY = float(os.getenv("MAX_RETRY_DELAY", "30"))
DELAY_BETWEEN_VISUAL_REQUESTS = float(
    os.getenv("DELAY_BETWEEN_VISUAL_REQUESTS", "3")
)

VISUAL_MIN_WIDTH = int(os.getenv("VISUAL_MIN_WIDTH", "200"))
VISUAL_MIN_HEIGHT = int(os.getenv("VISUAL_MIN_HEIGHT", "80"))

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
