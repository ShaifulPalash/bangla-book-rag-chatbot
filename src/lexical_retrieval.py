"""
src/lexical_retrieval.py
========================
Lexical retrieval for the Bangla Book RAG chatbot.

Uses BM25 over the existing book chunks to complement dense
BGE-M3 retrieval. Dense retrieval captures semantic similarity,
while BM25 is useful when important words in the question also
appear explicitly in the source text.
"""

import json
import re
import unicodedata
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.config import CHUNKS_PATH


def tokenize(text: str) -> list[str]:
    """
    Tokenize Bengali text for BM25.

    Uses Unicode normalization followed by whitespace-based
    tokenization. Bengali combining characters are preserved
    instead of being split by a regex word boundary.

    Punctuation attached to words is removed.
    """
    text = unicodedata.normalize("NFC", text.lower())

    # Replace punctuation/symbol characters with spaces while
    # preserving Bengali letters, Bengali vowel signs, digits,
    # and other combining marks.
    text = re.sub(r"[^\w\u0980-\u09FF\s]", " ", text, flags=re.UNICODE)

    return [token for token in text.split() if token]


@lru_cache(maxsize=1)
def _load_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Load all book chunks and build the BM25 index once.

    The result is cached so repeated questions do not rebuild
    the index for every query.
    """
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run src/chunking.py first."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25, chunks


def search_lexical(question: str, k: int = 10) -> list[tuple[dict, float]]:
    """
    Retrieve the top-k chunks using BM25.

    Returns:
        List of (chunk, bm25_score) tuples ordered from highest
        BM25 score to lowest.
    """
    bm25, chunks = _load_index()

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:k]

    return [
        (chunks[index], float(scores[index]))
        for index in ranked_indices
    ]
