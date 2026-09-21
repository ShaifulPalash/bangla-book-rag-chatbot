"""
src/chunking.py
===============

Reads the crawled book and splits each chapter into overlapping chunks
with citation metadata.
"""

from __future__ import annotations

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    RAW_BOOK_PATH,
    logger,
)


SEPARATORS = [
    "\n\n",
    "\n",
    "।",
    "৷",
    "?",
    "!",
    " ",
    "",
]


def build_splitter() -> RecursiveCharacterTextSplitter:
    """Create the Bengali-aware recursive text splitter."""

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )


def chunk_book(chapters: list[dict]) -> list[dict]:
    """Split all chapters while preserving source metadata."""

    if not chapters:
        raise ValueError("Cannot chunk an empty book.")

    splitter = build_splitter()

    all_chunks: list[dict] = []
    global_chunk_id = 0

    for chapter in chapters:
        required_fields = {
            "book_name",
            "author",
            "chapter_number",
            "chapter_name",
            "source_url",
            "text",
        }

        missing = required_fields - chapter.keys()

        if missing:
            raise ValueError(
                f"Chapter {chapter.get('chapter_number', '?')} "
                f"is missing fields: {sorted(missing)}"
            )

        pieces = splitter.split_text(chapter["text"])

        if not pieces:
            raise ValueError(
                f"Chapter {chapter['chapter_name']} produced no chunks."
            )

        logger.info(
            "Chapter '%s': %d chars -> %d chunks",
            chapter["chapter_name"],
            len(chapter["text"]),
            len(pieces),
        )

        for local_index, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "chunk_id": (
                        f"ch{chapter['chapter_number']:02d}_"
                        f"{local_index:03d}"
                    ),
                    "book_name": chapter["book_name"],
                    "author": chapter["author"],
                    "chapter_number": chapter["chapter_number"],
                    "chapter_name": chapter["chapter_name"],
                    "section": chapter["chapter_name"],
                    "source_url": chapter["source_url"],
                    "chunk_index_in_chapter": local_index,
                    "global_chunk_index": global_chunk_id,
                    "text": piece,
                }
            )

            global_chunk_id += 1

    if not all_chunks:
        raise ValueError("Chunking produced zero chunks.")

    return all_chunks


def main() -> None:
    """Read the crawler output and write chunked JSON."""

    if not RAW_BOOK_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_BOOK_PATH} not found. "
            "Run python src/crawler.py first."
        )

    with open(
        RAW_BOOK_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        chapters = json.load(file)

    logger.info(
        "Loaded %d chapters. Chunking with size=%d, overlap=%d.",
        len(chapters),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    chunks = chunk_book(chapters)

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    average_length = sum(
        len(chunk["text"])
        for chunk in chunks
    ) / len(chunks)

    logger.info(
        "Chunking complete: %d chunks "
        "(average %.0f characters) saved to %s.",
        len(chunks),
        average_length,
        CHUNKS_PATH,
    )


if __name__ == "__main__":
    main()