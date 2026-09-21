"""
src/config.py
=============
Central configuration for the Bangla Book RAG chatbot.

This module is intentionally the single source of truth for:
- environment variables
- model names
- chunking parameters
- retrieval parameters
- project paths
- logging

Secrets are loaded from .env and are never hardcoded.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

RAW_BOOK_PATH = DATA_DIR / "raw_debdas.json"
CHUNKS_PATH = DATA_DIR / "debdas_chunks.json"

VECTORSTORE_MANIFEST_PATH = CHROMA_DIR / "build_manifest.json"


for folder in (DATA_DIR, LOGS_DIR, CHROMA_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Explicitly load the .env file from the project root instead of depending
# on the directory from which Python happens to be launched.
# ---------------------------------------------------------------------------

ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    """Read and validate an integer environment variable."""

    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer, but received {raw_value!r}."
        ) from exc

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}, but received {value}."
        )

    return value


# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Gemini 3.6 Flash is a currently available stable Gemini API model.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")

# Runs locally through sentence-transformers.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = _env_int("CHUNK_SIZE", 500, minimum=100)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100, minimum=0)

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be smaller than CHUNK_SIZE. "
        f"Received CHUNK_SIZE={CHUNK_SIZE}, "
        f"CHUNK_OVERLAP={CHUNK_OVERLAP}."
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K = _env_int("TOP_K", 6, minimum=1)

DENSE_CANDIDATES = _env_int("DENSE_CANDIDATES", 20, minimum=1)
LEXICAL_CANDIDATES = _env_int("LEXICAL_CANDIDATES", 20, minimum=1)

RRF_K = _env_int("RRF_K", 60, minimum=1)

DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "1.0"))
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "2.0"))

if DENSE_WEIGHT <= 0 or LEXICAL_WEIGHT <= 0:
    raise ValueError("DENSE_WEIGHT and LEXICAL_WEIGHT must be positive.")


# ---------------------------------------------------------------------------
# Book metadata
# ---------------------------------------------------------------------------

BOOK_TITLE = "দেবদাস"
BOOK_AUTHOR = "শরৎচন্দ্র চট্টোপাধ্যায় (Sarat Chandra Chattopadhyay)"

BOOK_INDEX_URL = (
    "https://bn.wikisource.org/wiki/"
    "দেবদাস_(শরৎচন্দ্র_চট্টোপাধ্যায়)"
)

EXPECTED_CHAPTER_COUNT = 16


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = LOGS_DIR / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("debdas_rag")


if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY is not set. "
        "Crawling, chunking and vector indexing can still run, "
        "but Gemini answer generation requires GEMINI_API_KEY."
    )