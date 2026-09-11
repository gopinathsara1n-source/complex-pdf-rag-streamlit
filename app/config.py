import os

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768

# Conservative defaults based on the tested free-tier configuration.
RPM_LIMIT = 100
SAFE_RPM = 80
RPD_LIMIT = 1000
DAILY_SAFE_LIMIT = 950
MIN_REQUEST_INTERVAL = 60.0 / SAFE_RPM
MAX_RETRIES = 6

VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite")
ANSWER_MODEL = os.getenv("GEMINI_ANSWER_MODEL", "gemini-3.5-flash-lite")
