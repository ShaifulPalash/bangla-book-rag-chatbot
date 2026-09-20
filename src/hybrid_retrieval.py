"""
src/hybrid_retrieval.py
=======================
Hybrid retrieval for the Bangla Book RAG chatbot.

Combines dense BGE-M3 retrieval with lexical BM25 retrieval
using weighted Reciprocal Rank Fusion (RRF).

Dense retrieval is weighted normally, while BM25 receives a
higher weight because lexical matching is particularly useful
for explicit Bengali terms and names.
"""

from langchain_core.documents import Document

from src.lexical_retrieval import search_lexical
from src.vectordb import get_vectorstore


DENSE_CANDIDATES = 20
LEXICAL_CANDIDATES = 20

RRF_K = 60

DENSE_WEIGHT = 1.0
LEXICAL_WEIGHT = 2.0


def _document_key(doc: Document) -> tuple[int, int]:
    """Return the stable key identifying a chunk."""
    metadata = doc.metadata

    return (
        int(metadata["chapter_number"]),
        int(metadata["chunk_index_in_chapter"]),
    )


def _chunk_key(chunk: dict) -> tuple[int, int]:
    """Return the stable key identifying a lexical chunk."""
    return (
        int(chunk["chapter_number"]),
        int(chunk["chunk_index_in_chapter"]),
    )


def _chunk_to_document(chunk: dict) -> Document:
    """Convert a JSON chunk into a LangChain Document."""
    metadata = {
        "book_name": chunk["book_name"],
        "author": chunk["author"],
        "chapter_number": chunk["chapter_number"],
        "chapter_name": chunk["chapter_name"],
        "section": chunk["section"],
        "source_url": chunk["source_url"],
        "chunk_index_in_chapter": chunk["chunk_index_in_chapter"],
    }

    return Document(
        page_content=chunk["text"],
        metadata=metadata,
    )


def hybrid_search(
    question: str,
    k: int,
    dense_k: int = DENSE_CANDIDATES,
    lexical_k: int = LEXICAL_CANDIDATES,
) -> list[tuple[Document, float]]:
    """
    Retrieve documents using dense + BM25 retrieval and combine
    them with weighted Reciprocal Rank Fusion.

    Args:
        question: User's question.
        k: Number of final documents to return.
        dense_k: Number of dense candidates.
        lexical_k: Number of BM25 candidates.

    Returns:
        List of (Document, RRF score) tuples ordered by descending
        fused score.
    """
    vectorstore = get_vectorstore()

    dense_results = vectorstore.similarity_search_with_score(
        question,
        k=dense_k,
    )

    lexical_results = search_lexical(
        question,
        k=lexical_k,
    )

    documents: dict[tuple[int, int], Document] = {}
    rrf_scores: dict[tuple[int, int], float] = {}

    # Dense retrieval contribution.
    for rank, (doc, _distance) in enumerate(
        dense_results,
        start=1,
    ):
        key = _document_key(doc)

        documents[key] = doc

        rrf_scores[key] = rrf_scores.get(key, 0.0) + (
            DENSE_WEIGHT / (RRF_K + rank)
        )

    # BM25 retrieval contribution.
    for rank, (chunk, _bm25_score) in enumerate(
        lexical_results,
        start=1,
    ):
        key = _chunk_key(chunk)

        if key not in documents:
            documents[key] = _chunk_to_document(chunk)

        rrf_scores[key] = rrf_scores.get(key, 0.0) + (
            LEXICAL_WEIGHT / (RRF_K + rank)
        )

    ranked_keys = sorted(
        rrf_scores,
        key=lambda key: rrf_scores[key],
        reverse=True,
    )

    return [
        (documents[key], rrf_scores[key])
        for key in ranked_keys[:k]
    ]
