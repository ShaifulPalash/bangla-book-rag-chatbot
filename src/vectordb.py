"""
src/vectordb.py
===============

Builds a persistent Chroma vector store from data/debdas_chunks.json.

The vector collection is rebuilt cleanly whenever this script is run.
This prevents stale chunks from surviving after the source book or
chunking configuration changes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from langchain_chroma import Chroma

from src.config import (
    CHROMA_DIR,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    VECTORSTORE_MANIFEST_PATH,
    logger,
)
from src.embeddings import get_embedding_model


COLLECTION_NAME = "debdas_chunks"


def load_chunks() -> list[dict]:
    """Load and validate chunk data."""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. "
            "Run python src/chunking.py first."
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            f"{CHUNKS_PATH} does not contain any chunks."
        )

    required_fields = {
        "chunk_id",
        "text",
        "book_name",
        "author",
        "chapter_number",
        "chapter_name",
        "section",
        "source_url",
        "chunk_index_in_chapter",
    }

    for index, chunk in enumerate(chunks):
        missing = required_fields - chunk.keys()

        if missing:
            raise ValueError(
                f"Chunk {index} is missing fields: "
                f"{sorted(missing)}"
            )

    return chunks


def _chunks_hash(chunks: list[dict]) -> str:
    """Create a stable fingerprint of the chunk source."""

    serialized = json.dumps(
        chunks,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


def _delete_existing_collection() -> None:
    """Delete the old Chroma collection if it exists."""

    try:
        existing = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=str(CHROMA_DIR),
        )

        existing.delete_collection()

        logger.info(
            "Deleted existing Chroma collection '%s'.",
            COLLECTION_NAME,
        )

    except Exception as exc:
        # An absent collection is harmless. We log and continue because
        # Chroma's behavior differs slightly between versions when opening
        # a non-existing collection.
        logger.info(
            "No existing Chroma collection needed to be deleted: %s",
            exc,
        )


def build_vectorstore(chunks: list[dict]) -> Chroma:
    """Rebuild the persistent Chroma collection from scratch."""

    if not chunks:
        raise ValueError("Cannot build a vector store from zero chunks.")

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "book_name": chunk["book_name"],
            "author": chunk["author"],
            "chapter_number": chunk["chapter_number"],
            "chapter_name": chunk["chapter_name"],
            "section": chunk["section"],
            "source_url": chunk["source_url"],
            "chunk_index_in_chapter": chunk[
                "chunk_index_in_chapter"
            ],
        }
        for chunk in chunks
    ]

    ids = [chunk["chunk_id"] for chunk in chunks]

    logger.info(
        "Preparing to rebuild Chroma collection '%s' with %d chunks.",
        COLLECTION_NAME,
        len(chunks),
    )

    embedding_model = get_embedding_model()

    # Delete the previous collection so removed/renamed chunks cannot
    # survive a rebuild.
    _delete_existing_collection()

    logger.info(
        "Embedding %d chunks using %s...",
        len(texts),
        EMBEDDING_MODEL,
    )

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    count = vectorstore._collection.count()

    if count != len(chunks):
        raise RuntimeError(
            "Chroma build verification failed: "
            f"expected {len(chunks)} records but found {count}."
        )

    manifest = {
        "collection_name": COLLECTION_NAME,
        "count": count,
        "embedding_model": EMBEDDING_MODEL,
        "chunks_sha256": _chunks_hash(chunks),
        "built_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with open(
        VECTORSTORE_MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Vector store verified successfully: %d chunks.",
        count,
    )

    return vectorstore


def get_vectorstore() -> Chroma:
    """Load the existing Chroma collection without re-embedding."""

    if not VECTORSTORE_MANIFEST_PATH.exists():
        raise RuntimeError(
            "Vector store has not been successfully built. "
            "Run python src/vectordb.py first."
        )

    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )

    count = vectorstore._collection.count()

    if count <= 0:
        raise RuntimeError(
            "Chroma collection exists but contains zero documents. "
            "Rebuild it with python src/vectordb.py."
        )

    return vectorstore


def main() -> None:
    """Build and verify the vector database."""

    chunks = load_chunks()

    logger.info(
        "Loaded %d chunks from %s.",
        len(chunks),
        CHUNKS_PATH,
    )

    build_vectorstore(chunks)


if __name__ == "__main__":
    main()