"""
src/config.py
==============
Single source of truth for every setting used across the project:
API keys, model names, chunking parameters, file paths, and the logger.

Why this file exists (read this if you're new to config management):
----------------------------------------------------------------------
Instead of typing "chunk_size = 500" or pasting an API key into five
different scripts, every other file in this project does:

    from src.config import GEMINI_API_KEY, CHUNK_SIZE, logger

...and gets the same values every time. If you ever want to experiment
with a bigger chunk size, or swap to a different Gemini model, you change
it ONCE, here — not by hunting through crawler.py, chunking.py,
rag_pipeline.py, etc.

Secrets (like GEMINI_API_KEY) are never hardcoded in this file either.
They live in a local ".env" file (which is in .gitignore and therefore
never pushed to GitHub). This file only knows the *names* of the
variables it expects to find in .env, not their values.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Load environment variables from the .env file in the project root.
#    load_dotenv() looks for a file named ".env" and copies its key=value
#    pairs into os.environ, so os.getenv("GEMINI_API_KEY") below can find it.
#    If .env doesn't exist yet (e.g. fresh clone), this line does nothing
#    harmful — os.getenv() will just return None and we handle that below.
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# 2. Project-wide file paths.
#    We compute these relative to this file's location (not the current
#    working directory) so the project works no matter which folder you
#    run a script from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../debdas-rag-chatbot/
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

RAW_BOOK_PATH = DATA_DIR / "raw_debdas.json"        # crawler output (phase 2)
CHUNKS_PATH = DATA_DIR / "debdas_chunks.json"        # chunker output (phase 3)

# Make sure these folders exist even on a completely fresh clone of the repo,
# since logs/, data/, and chroma_db/ are git-ignored (empty folders aren't
# tracked by git at all).
for folder in (DATA_DIR, LOGS_DIR, CHROMA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. Secrets and API settings, read from the environment (.env file).
#    We deliberately do NOT put a default value for the API key — if it's
#    missing, we want the program to fail loudly and early (see the check
#    below), rather than silently trying to call Gemini with an empty key
#    and getting a confusing error three steps later.
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# 4. Model settings.
#    LLM_MODEL: used only at answer-generation time (one call per user
#    question), which is why it's fine for this to be an API call — the
#    volume is low, so free-tier limits are not a concern here.
#
#    EMBEDDING_MODEL: runs LOCALLY (via sentence-transformers), not through
#    an API. This is the model that gets called hundreds of times while
#    indexing the whole book, so keeping it local means indexing NEVER
#    touches a rate limit, no matter how many chunks the book produces.
# ---------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ---------------------------------------------------------------------------
# 5. Chunking settings (explained fully in Phase 3 / chunking.py, and in the
#    README's "Technical Details" section). Centralizing them here means the
#    bonus hit-rate experiment can later import and override just these two
#    numbers to compare strategies, without touching chunking.py itself.
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # characters of overlap

# ---------------------------------------------------------------------------
# 6. Retrieval settings.
#    TOP_K controls how many chunks we hand to the LLM per question. Keeping
#    this small (3-4) is a deliberate token-efficiency choice: fewer chunks
#    in the prompt = fewer input tokens per Gemini call = more questions you
#    can ask before hitting the free-tier TPM/RPD ceiling.
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "6"))

# ---------------------------------------------------------------------------
# 7. Book metadata (used for crawling and for tagging every chunk).
# ---------------------------------------------------------------------------
BOOK_TITLE = "দেবদাস"
BOOK_AUTHOR = "শরৎচন্দ্র চট্টোপাধ্যায় (Sarat Chandra Chattopadhyay)"
BOOK_INDEX_URL = "https://bn.wikisource.org/wiki/দেবদাস_(শরৎচন্দ্র_চট্টোপাধ্যায়)"

# ---------------------------------------------------------------------------
# 8. Logger setup.
#    We configure this once, here, so every other module can just do:
#        from src.config import logger
#        logger.info("some message")
#    and get identically formatted output, in both the terminal and a log
#    file, without repeating logging.basicConfig() in every script.
# ---------------------------------------------------------------------------
LOG_FILE = LOGS_DIR / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),  # utf-8: needed for Bangla text in logs
        logging.StreamHandler(),  # also prints to console
    ],
)

logger = logging.getLogger("debdas_rag")

# ---------------------------------------------------------------------------
# 9. Fail loudly if the API key is missing, but only when this module is
#    actually needed for something that calls Gemini. We warn (not crash)
#    at import time, because things like the crawler and chunker don't need
#    the key at all and shouldn't be blocked by its absence.
# ---------------------------------------------------------------------------
if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY is not set. Crawling/chunking/embedding will still "
        "work, but the RAG pipeline (Gemini LLM calls) will fail until you "
        "add your key to a .env file. See .env.example."
    )
