"""
tests/test_retrieval.py
=======================

Tests and evaluates the real RAG pipeline.

Default:
    pytest -q

This runs deterministic/offline tests and does NOT consume Gemini API
quota.

Live evaluation:
    RUN_LIVE_TESTS=1 python tests/test_retrieval.py

The live mode runs the 10 assignment questions against the real pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from src.config import (
    CHROMA_DIR,
    GEMINI_API_KEY,
    CHUNKS_PATH,
)
from src.lexical_retrieval import tokenize


NOT_FOUND_PHRASE = "এই তথ্যটি বইয়ে পাওয়া যায়নি."


TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "দেবদাস ও পার্বতীর মধ্যে ছোটবেলার সম্পর্ক কেমন ছিল?",
        "expected_keywords": ["দেবদাস", "পার্বতী"],
        "is_out_of_book": False,
    },
    {
        "id": 2,
        "question": "দেবদাসের পিতার সাথে তার সম্পর্ক কেমন ছিল?",
        "expected_keywords": ["পিতা"],
        "is_out_of_book": False,
    },
    {
        "id": 3,
        "question": (
            "দেবদাস কেন চিরদিনের মতো তার কলিকাতার বাসা "
            "ছেড়ে বাড়ি চলে গিয়েছিল?"
        ),
        "expected_keywords": ["বাসা", "বাড়ি"],
        "is_out_of_book": False,
    },
    {
        "id": 4,
        "question": "কলিকাতায় দেবদাসের ঘনিষ্ঠ বন্ধুর নাম কী ছিল?",
        "expected_keywords": ["চুনিলাল"],
        "is_out_of_book": False,
    },
    {
        "id": 5,
        "question": "পার্বতীর বিবাহ কেমন পরিবারে হয়েছিল?",
        "expected_keywords": ["বিবাহ", "জমিদার"],
        "is_out_of_book": False,
    },
    {
        "id": 6,
        "question": "চন্দ্রমুখী কে ছিলেন?",
        "expected_keywords": ["চন্দ্রমুখী"],
        "is_out_of_book": False,
    },
    {
        "id": 7,
        "question": "ষোড়শ পরিচ্ছেদে দেবদাসের শারীরিক অবস্থা কেমন ছিল?",
        "expected_keywords": ["দেবদাস"],
        "is_out_of_book": False,
    },
    {
        "id": 8,
        "question": "ধর্মদাস কে ছিলেন?",
        "expected_keywords": ["ধর্মদাস"],
        "is_out_of_book": False,
    },
    {
        "id": 9,
        "question": "উপন্যাসের শেষে দেবদাসের কী পরিণতি হয়েছিল?",
        "expected_keywords": ["দেবদাস"],
        "is_out_of_book": False,
    },
    {
        "id": 10,
        "question": "দেবদাস কোন স্মার্টফোন ব্র্যান্ড ব্যবহার করত?",
        "expected_keywords": [],
        "is_out_of_book": True,
    },
]


def pipeline_is_ready() -> bool:
    """Return whether the real pipeline appears ready."""

    return (
        CHUNKS_PATH.exists()
        and (CHROMA_DIR / "chroma.sqlite3").exists()
        and bool(GEMINI_API_KEY)
    )


def run_all_questions() -> list[dict]:
    """Run the complete live evaluation."""

    from src.rag_pipeline import ask

    results = []

    for question in TEST_QUESTIONS:
        outcome = ask(question["question"])

        results.append(
            {
                **question,
                "actual_answer": outcome["answer"],
                "actual_sources": outcome["sources"],
                "refused": outcome["refused"],
                "retrieved": outcome["retrieved"],
            }
        )

    return results


# ---------------------------------------------------------------------------
# Offline tests
# ---------------------------------------------------------------------------

def test_bengali_tokenizer_preserves_bengali_words():
    tokens = tokenize(
        "দেবদাস ও পার্বতীর সম্পর্ক কেমন ছিল?"
    )

    assert "দেবদাস" in tokens
    assert "পার্বতীর" in tokens


def test_bengali_tokenizer_removes_punctuation():
    tokens = tokenize(
        "দেবদাস, পার্বতী। চন্দ্রমুখী!"
    )

    assert "দেবদাস" in tokens
    assert "পার্বতী" in tokens
    assert "চন্দ্রমুখী" in tokens

    assert "," not in tokens
    assert "।" not in tokens
    assert "!" not in tokens


def test_test_dataset_contains_ten_questions():
    assert len(TEST_QUESTIONS) == 10


def test_test_dataset_contains_one_out_of_book_question():
    out_of_book = [
        q
        for q in TEST_QUESTIONS
        if q["is_out_of_book"]
    ]

    assert len(out_of_book) == 1


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

RUN_LIVE_TESTS = (
    os.getenv("RUN_LIVE_TESTS", "").lower()
    in {"1", "true", "yes"}
)


live_skip = pytest.mark.skipif(
    not RUN_LIVE_TESTS,
    reason=(
        "Live Gemini tests are disabled by default. "
        "Set RUN_LIVE_TESTS=1 to run them."
    ),
)


@live_skip
@pytest.mark.parametrize(
    "question",
    [
        q
        for q in TEST_QUESTIONS
        if not q["is_out_of_book"]
    ],
    ids=lambda q: f"q{q['id']}",
)
def test_live_in_book_question(question):
    """Check that a real in-book question gets an answer and source."""

    if not pipeline_is_ready():
        pytest.skip(
            "Pipeline is not ready."
        )

    from src.rag_pipeline import ask

    result = ask(question["question"])

    assert not result["refused"], (
        f"Question {question['id']} was refused: "
        f"{result['answer']!r}"
    )

    assert result["answer"].strip()

    assert result["sources"], (
        f"Question {question['id']} returned no sources."
    )


@live_skip
def test_live_out_of_book_question():
    """The deliberate out-of-book question must be refused."""

    if not pipeline_is_ready():
        pytest.skip(
            "Pipeline is not ready."
        )

    question = next(
        q
        for q in TEST_QUESTIONS
        if q["is_out_of_book"]
    )

    from src.rag_pipeline import ask

    result = ask(question["question"])

    assert result["refused"], (
        f"Expected refusal but received: "
        f"{result['answer']!r}"
    )

    assert result["answer"] == NOT_FOUND_PHRASE

    assert result["sources"] == []


def print_live_report(results: list[dict]) -> None:
    """Print a readable live evaluation report."""

    print()
    print("=" * 100)
    print("DEBDAS RAG — LIVE EVALUATION")
    print("=" * 100)

    for result in results:
        source_names = ", ".join(
            source["chapter_name"]
            for source in result["actual_sources"]
        )

        print()
        print(f"Q{result['id']}: {result['question']}")
        print(f"Refused: {result['refused']}")
        print(f"Sources: {source_names or '-'}")
        print(f"Answer: {result['actual_answer']}")

    print()
    print("=" * 100)


if __name__ == "__main__":
    if not RUN_LIVE_TESTS:
        print(
            "Live evaluation is disabled.\n\n"
            "Run:\n"
            "  RUN_LIVE_TESTS=1 python tests/test_retrieval.py"
        )
        sys.exit(0)

    if not pipeline_is_ready():
        print(
            "Pipeline is not ready.\n\n"
            "Run:\n"
            "  python src/crawler.py\n"
            "  python src/chunking.py\n"
            "  python src/vectordb.py\n\n"
            "and configure GEMINI_API_KEY in .env."
        )
        sys.exit(1)

    results = run_all_questions()

    print_live_report(results)