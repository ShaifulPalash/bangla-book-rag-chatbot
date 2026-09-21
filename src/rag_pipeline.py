"""
src/rag_pipeline.py

Core grounded RAG pipeline for Debdas QA.

Question
   ↓
Hybrid Retrieval
   ↓
Top 5 book chunks
   ↓
BOOK CONTEXT injected into Gemini
   ↓
Gemini generates answer
   ↓
Citation extraction
   ↓
Source URL validation
   ↓
Final grounded answer

Fail-closed:
Gemini is only allowed to answer from retrieved book context.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / ".cache"

os.environ["HF_HOME"] = str(
    CACHE_DIR / "huggingface"
)

os.environ["HF_HUB_CACHE"] = str(
    CACHE_DIR / "huggingface" / "hub"
)

os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(
    CACHE_DIR / "sentence_transformers"
)
# other imports below
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


import logging
import re
from functools import lru_cache
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_API_KEY, LLM_MODEL
from src.hybrid_retrieval import hybrid_search


logger = logging.getLogger("debdas_rag")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFUSAL_ANSWER = "এই তথ্যটি বইয়ে পাওয়া যায়নি."

DEFAULT_TOP_K = 5
MAX_TOP_K = 6


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
তুমি "দেবদাস" উপন্যাসভিত্তিক একটি grounded Bengali question-answering assistant।

তোমার একমাত্র জ্ঞানসূত্র হলো নিচে দেওয়া BOOK CONTEXT।

কঠোর নিয়ম:

1. শুধুমাত্র BOOK CONTEXT-এর তথ্য ব্যবহার করে উত্তর দেবে।

2. BOOK CONTEXT-এর বাইরে কোনো তথ্য ব্যবহার করবে না।

3. তথ্য না থাকলে ঠিক এই বাক্যটি দেবে:

এই তথ্যটি বইয়ে পাওয়া যায়নি.

4. উত্তর বাংলায় দেবে।

5. সংক্ষিপ্ত কিন্তু যথেষ্ট ব্যাখ্যা দেবে।

6. Citation লিখবে না।

7. শুধুমাত্র BOOK CONTEXT থেকে উত্তর দেবে।

8. Citation কখনো বাদ দেবে না।

9. Citation-এর পরে কোনো লেখা থাকবে না।

10. কোনো তথ্য বানিয়ে যোগ করবে না।

BOOK CONTEXT:

{context}
"""


_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT,
        ),
        (
            "human",
            "প্রশ্ন:\n{question}\n\n"
            "BOOK CONTEXT-এর ভিত্তিতে উত্তর দাও।",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Gemini LLM
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize Gemini once.
    """

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Set it inside .env"
        )

    logger.info(
        "Initializing Gemini model: %s",
        LLM_MODEL,
    )

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        max_retries=0,
    )


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(
    results: list[dict[str, Any]]
) -> str:
    """
    Convert retrieved passages into Gemini context.
    """

    if not results:
        return ""

    parts = []

    for index, result in enumerate(results, start=1):

        chapter = str(
            result.get("chapter_name")
            or "অজানা অধ্যায়"
        )

        chunk_id = str(
            result.get("chunk_id")
            or "unknown"
        )

        text = str(
            result.get("text")
            or ""
        ).strip()

        if not text:
            continue

        # Reduce Gemini token usage
        text = text[:1200]

        parts.append(
            f"[Passage {index} | অধ্যায়: {chapter} | chunk: {chunk_id}]\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Gemini invocation
# ---------------------------------------------------------------------------

def _invoke_llm(
    messages: list[BaseMessage]
) -> str:
    """
    Single Gemini call.

    Handles quota errors safely.
    """

    llm = get_llm()

    try:
        response = llm.invoke(messages)

        content = getattr(
            response,
            "content",
            response
        )

        if isinstance(content, list):
            content = "".join(
                str(item.get("text", item))
                if isinstance(item, dict)
                else str(item)
                for item in content
            )

        return str(content).strip()

    except Exception as exc:
        error_text = str(exc).lower()

        if (
            "429" in error_text
            or "resource exhausted" in error_text
            or "quota" in error_text
        ):
            logger.warning(
                "Gemini quota exceeded. Returning grounded refusal."
            )
            return REFUSAL_ANSWER

        raise

# ---------------------------------------------------------------------------
# Citation handling
# ---------------------------------------------------------------------------

def _extract_cited_chapters(
    answer: str,
    available_chapters: set[str],
) -> list[str]:
    """
    Extract valid chapter citations from Gemini output.
    """

    if not answer:
        return []

    matches = re.findall(
        r"সূত্র\s*:\s*([^\]\n]+)",
        answer,
        flags=re.UNICODE,
    )

    cited = []

    for item in matches:
        chapter = item.strip()

        if chapter in available_chapters:
            cited.append(chapter)

    return list(dict.fromkeys(cited))



def _remove_invalid_citation(
    answer: str,
) -> str:
    """
    Remove citation block.
    """

    return re.sub(
        r"\[সূত্র\s*:\s*[^\]]+\]",
        "",
        answer,
        flags=re.UNICODE,
    ).strip()



# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _sources_from_chapters(
    results: list[dict[str, Any]],
    chapters: list[str],
) -> list[dict[str, str]]:

    if not chapters:
        return []

    chapter_set = set(chapters)

    seen = set()
    sources = []

    for result in results:

        chapter = result.get(
            "chapter_name"
        )

        source_url = result.get(
            "source_url"
        )

        if chapter not in chapter_set:
            continue

        if not source_url:
            continue

        source_url = str(source_url)

        if source_url in seen:
            continue

        seen.add(source_url)

        sources.append(
            {
                "chapter_name": str(chapter),
                "source_url": source_url,
            }
        )

    return sources



# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize_retrieved(
    results: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    serialized = []

    for result in results:

        serialized.append(
            {
                "chunk_id": result.get(
                    "chunk_id"
                ),

                "chapter_name": result.get(
                    "chapter_name"
                ),

                "score": float(
                    result.get(
                        "_hybrid_score",
                        0.0,
                    )
                ),

                "rrf_score": float(
                    result.get(
                        "_rrf_score",
                        0.0,
                    )
                ),

                "text_preview": str(
                    result.get("text") or ""
                )[:200],
            }
        )

    return serialized



# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_context(
    results: list[dict[str, Any]],
    context: str,
):

    if results and not context.strip():

        raise RuntimeError(
            "Retrieved passages exist but context is empty."
        )



# ---------------------------------------------------------------------------
# Main RAG pipeline
# ---------------------------------------------------------------------------

def ask(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:

    question = str(
        question or ""
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )


    top_k = min(
        max(top_k, 1),
        MAX_TOP_K,
    )


    logger.info(
        "Question: %s",
        question,
    )


    # ---------------------------------------------------------------
    # 1. Retrieval
    # ---------------------------------------------------------------

    results = hybrid_search(
        question,
        k=top_k,
    )


    logger.info(
        "Retrieved %s passages",
        len(results),
    )


    # ---------------------------------------------------------------
    # 2. No evidence
    # ---------------------------------------------------------------

    if not results:

        return {
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "retrieved": [],
        }



    # ---------------------------------------------------------------
    # 3. Build context
    # ---------------------------------------------------------------

    context = format_context(
        results
    )


    _validate_context(
        results,
        context,
    )


    # ---------------------------------------------------------------
    # 4. Prepare Gemini prompt
    # ---------------------------------------------------------------

    messages = _prompt_template.format_messages(
        question=question,
        context=context,
    )


    rendered = "\n".join(
        str(message.content)
        for message in messages
    )


    if context not in rendered:

        raise RuntimeError(
            "BOOK CONTEXT missing from Gemini prompt."
        )



    # ---------------------------------------------------------------
    # 5. Generate answer
    # ---------------------------------------------------------------

    answer = _invoke_llm(
        messages
    )

    best_chapter = results[0].get("chapter_name")

    if best_chapter:
        answer = (
            answer.strip()
            + f"\n\n[সূত্র: {best_chapter}]"
        )

    if not answer:

        return {
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "retrieved": _serialize_retrieved(results),
        }



    # ---------------------------------------------------------------
    # 6. Grounded refusal
    # ---------------------------------------------------------------

    if REFUSAL_ANSWER in answer:

        return {
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "retrieved": _serialize_retrieved(results),
        }



    # ---------------------------------------------------------------
    # 7. Validate citations
    # ---------------------------------------------------------------

    available_chapters = {
        str(result.get("chapter_name"))
        for result in results
        if result.get("chapter_name")
    }


    cited_chapters = _extract_cited_chapters(
        answer,
        available_chapters,
    )


    if cited_chapters:

        sources = _sources_from_chapters(
            results,
            cited_chapters,
        )

    else:

        logger.warning(
            "No valid citation found."
        )

        clean_answer = _remove_invalid_citation(
            answer
        )

        answer = (
            clean_answer
            + "\n\n"
            + "দ্রষ্টব্য: উত্তরটির জন্য বৈধ অধ্যায়-উৎস শনাক্ত করা যায়নি।"
        )

        sources = []



    # ---------------------------------------------------------------
    # 8. Return
    # ---------------------------------------------------------------

    return {

        "answer": answer,

        "sources": sources,

        "retrieved": _serialize_retrieved(
            results
        ),

    }



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse
    import json


    parser = argparse.ArgumentParser(
        description="Ask grounded Debdas questions."
    )


    parser.add_argument(
        "question",
        nargs="?",
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )


    args = parser.parse_args()


    question = args.question


    if not question:

        question = input(
            "প্রশ্ন: "
        ).strip()



    result = ask(
        question,
        top_k=args.top_k,
    )


    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(
        result["answer"]
    )


    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)


    if result["sources"]:

        for source in result["sources"]:

            print(
                f"- {source['chapter_name']}: "
                f"{source['source_url']}"
            )

    else:

        print(
            "No validated sources."
        )


    print("\n" + "=" * 80)
    print("RETRIEVED")
    print("=" * 80)


    print(
        json.dumps(
            result["retrieved"],
            ensure_ascii=False,
            indent=2,
        )
    )