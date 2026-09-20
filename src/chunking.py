"""
src/chunking.py
=================
Reads the crawled book (data/raw_debdas.json) and splits each chapter's
text into overlapping chunks suitable for embedding, attaching metadata
(book name, chapter, section, source URL, chunk position) to every chunk
so citations work later in the RAG pipeline.

Why chunk at all?
------------------
An embedding model turns text into a single vector. If we embedded an
entire 16-chapter novel as one vector, a search for "কেন দেবদাস বাড়ি
ছাড়িয়াছিল" (why did Debdas leave home) would have nothing more specific
to match against than "the whole book." Splitting into small, overlapping
passages lets the vector database point to the 3-4 passages that actually
discuss that event, which is the entire point of retrieval.

Why these specific chunk_size / chunk_overlap values (see src/config.py):
---------------------------------------------------------------------------
- CHUNK_SIZE=500 (characters): Bengali conjunct characters tokenize more
  densely than plain English letters, so 500 characters is comfortably
  within bge-m3's 8192-token limit while still holding several full
  sentences of coherent context. It also keeps TOP_K retrieved chunks
  small in total tokens, protecting the Gemini free-tier budget per query.
- CHUNK_OVERLAP=100 (20% of chunk size): without overlap, a sentence
  sitting exactly on a chunk boundary gets split in half and neither
  half makes sense alone. A 20% overlap is a common default that repairs
  boundary sentences in the following chunk without heavily duplicating
  the whole index.
- Bengali-aware separators: Bengali sentences end in "।" (দাঁড়ি), not ".".
  A splitter that only knows about paragraph breaks and spaces will
  happily cut a Bengali sentence in half. We tell it about "।" explicitly
  so it prefers to break there first.
"""

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, RAW_BOOK_PATH, CHUNKS_PATH, logger

# ---------------------------------------------------------------------------
# The splitter tries each separator in order, only falling back to the next
# one if a piece is still too big. Priority: paragraph break > line break >
# Bengali sentence end (।) > space > hard character cut (empty string, last
# resort so we NEVER silently fail to split an oversized chunk).
# ---------------------------------------------------------------------------
SEPARATORS = ["\n\n", "\n", "।", " ", ""]


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )


def chunk_book(chapters: list[dict]) -> list[dict]:
    """
    Splits every chapter's text into overlapping chunks and attaches
    metadata needed for citation later: book name, author, chapter
    number/name, a "section" label, source URL, and this chunk's position
    within its chapter.

    Note on "section": দেবদাস's chapters aren't subdivided into named
    sub-sections in the source text, so we honestly set `section` to the
    chapter name rather than inventing fake sub-headers. `chunk_index`
    gives finer-grained positional detail within the chapter for anyone
    who wants it.
    """
    splitter = build_splitter()
    all_chunks = []
    global_chunk_id = 0

    for chapter in chapters:
        pieces = splitter.split_text(chapter["text"])
        logger.info(
            f"Chapter '{chapter['chapter_name']}': "
            f"{len(chapter['text'])} chars -> {len(pieces)} chunks"
        )

        for local_index, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"ch{chapter['chapter_number']:02d}_{local_index:03d}",
                "book_name": chapter["book_name"],
                "author": chapter["author"],
                "chapter_number": chapter["chapter_number"],
                "chapter_name": chapter["chapter_name"],
                "section": chapter["chapter_name"],  # see note above
                "source_url": chapter["source_url"],
                "chunk_index_in_chapter": local_index,
                "global_chunk_index": global_chunk_id,
                "text": piece,
            })
            global_chunk_id += 1

    return all_chunks


def main():
    if not RAW_BOOK_PATH.exists():
        logger.error(
            f"{RAW_BOOK_PATH} not found. Run src/crawler.py first "
            f"(Phase 2) before chunking."
        )
        return

    with open(RAW_BOOK_PATH, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    logger.info(
        f"Loaded {len(chapters)} chapters. "
        f"Chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP} ..."
    )

    chunks = chunk_book(chapters)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0
    logger.info(
        f"Chunking complete: {len(chunks)} total chunks "
        f"(avg {avg_len:.0f} chars/chunk) saved to {CHUNKS_PATH}"
    )


if __name__ == "__main__":
    main()
