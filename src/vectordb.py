"""
src/vectordb.py
=================
Builds a persisted Chroma vector store from data/debdas_chunks.json
(Phase 3's output), embedding every chunk with the local bge-m3 model
(Phase 4's embeddings.py) and storing it with full citation metadata so
the RAG pipeline (Phase 5) can retrieve relevant passages and cite them.

Why Chroma (over FAISS) for this project:
--------------------------------------------
- Simpler LangChain integration: Chroma.from_texts(...) handles both the
  vector index AND metadata storage in one call, with built-in disk
  persistence. FAISS's LangChain wrapper works too, but requires a bit
  more manual bookkeeping to keep the index and metadata store in sync.
- Native metadata filtering: since every chunk carries chapter/section/
  source_url metadata, Chroma lets us filter or inspect by those fields
  directly, which is convenient for debugging retrieval quality.
- It's the more commonly recommended default for small-to-mid scale RAG
  projects and tutorials, which matters for a beginner-friendly,
  industry-aligned assignment submission.

Why deterministic chunk IDs matter:
--------------------------------------
Each chunk was given a stable id in Phase 3 (e.g. "ch07_003"). We reuse
that same id here as the Chroma document id. This means re-running this
script (e.g. after re-crawling or re-chunking) UPDATES existing entries
in place instead of creating duplicates — Chroma dedupes by id. Without
this, running the pipeline twice would silently double your index.
"""

import json

from langchain_chroma import Chroma

from src.config import CHUNKS_PATH, CHROMA_DIR, logger
from src.embeddings import get_embedding_model

COLLECTION_NAME = "debdas_chunks"


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run src/chunking.py first (Phase 3)."
        )
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vectorstore(chunks: list[dict]) -> Chroma:
    """
    Embeds every chunk and stores it in a persisted Chroma collection.

    We separate the chunk's free text (what gets embedded) from its
    metadata (chapter, section, source_url, etc. — what gets attached
    for citation, not embedded). This is standard RAG practice: you
    search over content, but you cite using metadata.
    """
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "book_name": c["book_name"],
            "author": c["author"],
            "chapter_number": c["chapter_number"],
            "chapter_name": c["chapter_name"],
            "section": c["section"],
            "source_url": c["source_url"],
            "chunk_index_in_chapter": c["chunk_index_in_chapter"],
        }
        for c in chunks
    ]
    ids = [c["chunk_id"] for c in chunks]

    logger.info(
        f"Embedding and indexing {len(texts)} chunks into Chroma "
        f"collection '{COLLECTION_NAME}' at {CHROMA_DIR} ..."
    )

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    logger.info(
        f"Vector store built and persisted. "
        f"Collection now contains {vectorstore._collection.count()} chunks."
    )
    return vectorstore


def get_vectorstore() -> Chroma:
    """
    Loads the existing persisted Chroma collection without re-embedding
    anything. Used by the RAG pipeline (Phase 5) and the Streamlit app
    (Phase 6), which should NOT re-run the (slow) embedding step every
    time someone asks a question — they just open the already-built index.
    """
    embedding_model = get_embedding_model()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )


def main():
    chunks = load_chunks()
    logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}.")
    build_vectorstore(chunks)


if __name__ == "__main__":
    main()
