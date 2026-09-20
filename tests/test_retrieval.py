"""
tests/test_retrieval.py
=========================
The assignment's required 10 test questions about দেবদাস, implemented as
automated tests rather than a static table — so "does the chatbot still
answer correctly" can be checked with one command instead of manually
asking each question by hand every time something changes.

This file doubles as the bonus hit-rate evaluation harness (Phase 8):
run_all_questions() returns structured results that a comparison script
can score against expected_chapter to compute a hit-rate percentage.

A note on how questions 1-9's `expected_chapter` values were chosen:
------------------------------------------------------------------------
Questions 2, 3, 4, 7, and 8 are grounded in real excerpts confirmed while
researching this book's structure (the "Devdas leaves home for good"
scene and চুনিলাল's introduction both appear in সপ্তম পরিচ্ছেদ; Devdas's
declining health and ধর্মদাস's care appear in ষোড়শ পরিচ্ছেদ).
Questions 1, 5, 6, and 9 rely on the novel's well-documented general plot
(childhood romance, Parvati's arranged marriage, Chandramukhi's character,
Devdas's death) but their EXACT chapter number is a best-effort estimate,
since this was written without reading your specific crawled text.

>>> Before submitting, open your data/debdas_chunks.json and verify/adjust
>>> the `expected_chapter` values below against what your crawler actually
>>> found. This matters for the hit-rate bonus score, not for whether the
>>> chatbot itself works correctly.

Since Gemini's answers are generated text (not fixed strings), we can't
check for an exact match. Instead, each question lists `expected_keywords`
— Bangla terms a correct, book-grounded answer should contain — and we
check for their presence. This is standard practice for evaluating
generative RAG systems.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GEMINI_API_KEY, CHROMA_DIR, logger

# ---------------------------------------------------------------------------
# The 10 required test questions. Question 10 is the deliberate
# out-of-book question required by the assignment, testing refusal.
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    {
    "id": 1,
    "question": "দেবদাস ও পার্বতীর মধ্যে ছোটবেলার সম্পর্ক কেমন ছিল?",
    "expected_answer_summary": (
        "শৈশবে দেবদাস ও পার্বতীর মধ্যে ঘনিষ্ঠ কিশোর-বন্ধন ছিল; "
        "তারা একে অপরকে আপনজনের মতো জানত।"
    ),
    "expected_keywords": ["ছোটবেলায়", "অধিকার", "দখল"],
    "expected_chapter": "পঞ্চম পরিচ্ছেদ",
    "is_out_of_book": False,
    },
    {
        "id": 2,
        "question": "দেবদাসের পিতার সাথে তার সম্পর্ক কেমন ছিল?",
        "expected_answer_summary": (
            "পিতার সাথে দেবদাসের সম্পর্ক তিক্ত ছিল; "
            "পিতা তার প্রতি কঠোর ও বিরক্ত ছিলেন।"
        ),
        "expected_keywords": ["পিতা", "জ্বালাতন"],
        "expected_chapter": "সপ্তম পরিচ্ছেদ",
        "is_out_of_book": False,
    },
    {
        "id": 3,
        "question": "দেবদাস কেন চিরদিনের মতো তার কলিকাতার বাসা ছেড়ে বাড়ি চলে গিয়েছিল?",
        "expected_answer_summary": (
            "পিতার সাথে বাদানুবাদের পর দুঃখ পেয়ে সে বাসার সব জিনিসপত্র নিয়ে "
            "চিরতরে বাড়ি ফিরে যায়।"
        ),
        "expected_keywords": ["বাসা", "বাটী"],
        "expected_chapter": "সপ্তম পরিচ্ছেদ",
        "is_out_of_book": False,
    },
    {
        "id": 4,
        "question": "কলিকাতায় দেবদাসের ঘনিষ্ঠ বন্ধুর নাম কী ছিল?",
        "expected_answer_summary": "চুনিলাল।",
        "expected_keywords": ["চুনিলাল"],
        "expected_chapter": "সপ্তম পরিচ্ছেদ",
        "is_out_of_book": False,
    },
    {
        "id": 5,
        "question": "পার্বতীর বিবাহ কেমন পরিবারে হয়েছিল?",
        "expected_answer_summary": (
            "পার্বতীর বিবাহ হয়েছিল হাতীপোতা গ্রামের এক সচ্ছল জমিদারের সঙ্গে। "
            "পাত্রের বয়স চল্লিশের নিচে ছিল এবং তিনি আগে বিবাহিত ছিলেন।"
        ),
        "expected_keywords": ["জমিদার", "বয়স", "টাকাকড়ি"],
        "expected_chapter": "পঞ্চম পরিচ্ছেদ",
        "is_out_of_book": False,
    },
    {
        "id": 6,
        "question": "চন্দ্রমুখী কে ছিলেন?",
        "expected_answer_summary": (
            "চন্দ্রমুখী ছিলেন সেই নারী যাঁর সঙ্গে দেবদাসের পরিচয় হয় "
            "চুনিলালের মাধ্যমে; পরে তাঁর মনে দেবদাসের প্রতি মায়া জন্মায়।"
        ),
        "expected_keywords": ["মায়া"],
        "expected_chapter": "নবম পরিচ্ছেদ",
        "is_out_of_book": False,
    },
    {
        "id": 7,
        "question": "ষোড়শ পরিচ্ছেদে দেবদাসের শারীরিক অবস্থা কেমন ছিল?",
        "expected_answer_summary": (
            "জ্বরে আক্রান্ত হয়ে শয্যাশায়ী হয়ে পড়েছিল এবং সুস্থ হয়ে উঠতে পারছিল না।"
        ),
        "expected_keywords": ["জ্বর"],
        "expected_chapter": "ষোড়শ পরিচ্ছেদ",
        "is_out_of_book": False,
    },
    {
        "id": 8,
        "question": "ধর্মদাস কে ছিলেন?",
        "expected_answer_summary": (
            "ধর্মদাস ছিলেন দেবদাসের বাড়ির পুরনো ভৃত্য ও বিশ্বস্ত সঙ্গী।"
        ),
        "expected_keywords": ["ভৃত্য"],
        "expected_chapter": "প্রথম পরিচ্ছেদ",
        "is_out_of_book": False,
    },
    {
        "id": 9,
        "question": "উপন্যাসের শেষে দেবদাসের কী পরিণতি হয়েছিল?",
        "expected_answer_summary": (
            "দেবদাস পার্বতীর বাড়ির কাছে এসে অসুস্থ অবস্থায় মৃত্যুবরণ করে।"
        ),
        "expected_keywords": ["মৃত্যু", "মারা"],
        "expected_chapter": "ষোড়শ পরিচ্ছেদ",
        "is_out_of_book": False,
    },
    {
        "id": 10,
        "question": "দেবদাস কোন স্মার্টফোন ব্র্যান্ড ব্যবহার করত?",
        "expected_answer_summary": None,
        "expected_keywords": [],
        "expected_chapter": None,
        "is_out_of_book": True,
    },
]


def pipeline_is_ready() -> bool:
    """Checks whether the full pipeline (index + API key) is set up."""
    return (CHROMA_DIR / "chroma.sqlite3").exists() and bool(GEMINI_API_KEY)


skip_if_not_ready = pytest.mark.skipif(
    not pipeline_is_ready(),
    reason=(
        "Pipeline not fully set up — run src/crawler.py, src/chunking.py, "
        "src/vectordb.py, and set GEMINI_API_KEY in .env before running "
        "these tests."
    ),
)


def run_all_questions() -> list[dict]:
    """
    Runs every test question through the real RAG pipeline and returns
    structured results. Used both by the pytest tests below AND by the
    bonus hit-rate comparison script (Phase 8), so retrieval-quality
    scoring logic only needs to be written once.
    """
    from src.rag_pipeline import ask  # imported lazily so this module can
                                        # be imported even before the API
                                        # key is set, for tooling purposes

    results = []
    for q in TEST_QUESTIONS:
        logger.info(f"Running test question {q['id']}: {q['question']}")
        outcome = ask(q["question"])
        results.append({**q, "actual_answer": outcome["answer"],
                         "actual_sources": outcome["sources"],
                         "refused": outcome["refused"]})
    return results


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@skip_if_not_ready
@pytest.mark.parametrize("q", [q for q in TEST_QUESTIONS if not q["is_out_of_book"]], ids=lambda q: f"q{q['id']}")
def test_in_book_question_is_answered_and_cited(q):
    """
    For each of the 9 real in-book questions: the chatbot should NOT
    refuse, should produce a non-empty answer, should cite at least one
    source chapter, and that answer should contain at least one expected
    keyword (a loose but meaningful check on relevance, since we can't
    exact-match generative text).
    """
    from src.rag_pipeline import ask
    result = ask(q["question"])

    assert not result["refused"], (
        f"Q{q['id']} was incorrectly refused: {q['question']!r}"
    )
    assert result["answer"].strip(), f"Q{q['id']} produced an empty answer."
    assert result["sources"], f"Q{q['id']} produced no citations."

    if q["expected_keywords"]:
        found = any(kw in result["answer"] for kw in q["expected_keywords"])
        assert found, (
            f"Q{q['id']}: none of {q['expected_keywords']} found in "
            f"answer: {result['answer']!r}"
        )


@skip_if_not_ready
def test_out_of_book_question_is_refused():
    """
    The assignment's required refusal test: a deliberately unanswerable
    question must trigger the exact "not found in book" behavior, not a
    hallucinated answer.
    """
    from src.rag_pipeline import ask
    out_of_book_q = next(q for q in TEST_QUESTIONS if q["is_out_of_book"])
    result = ask(out_of_book_q["question"])

    assert result["refused"], (
        f"Out-of-book question was NOT refused — got: {result['answer']!r}"
    )
    assert result["sources"] == [], "Refused answer should carry no citations."


if __name__ == "__main__":
    # Standalone report mode: python tests/test_retrieval.py
    # Prints a readable pass/fail table for all 10 questions — this is
    # also the artifact you can screenshot/paste as your "10 test
    # questions + expected answers" deliverable.
    if not pipeline_is_ready():
        print(
            "Pipeline not fully set up. Run src/crawler.py, "
            "src/chunking.py, src/vectordb.py, and set GEMINI_API_KEY "
            "in .env first."
        )
        sys.exit(1)

    results = run_all_questions()
    print(f"\n{'#':<3} {'Refused?':<9} {'Sources':<30} Question")
    print("-" * 90)
    for r in results:
        chapters = ", ".join(s["chapter_name"] for s in r["actual_sources"]) or "-"
        print(f"{r['id']:<3} {str(r['refused']):<9} {chapters:<30} {r['question']}")
