"""
src/hybrid_retrieval.py
=======================

Hybrid retrieval for the Bangla Book RAG chatbot.

Combines:
    1. Dense semantic retrieval from Chroma + BGE-M3
    2. Lexical retrieval from BM25
    3. Reciprocal Rank Fusion (RRF)
    4. Lightweight Bengali query expansion
    5. Evidence-aware reranking
    6. Chapter-aware diversity

The implementation is intentionally compatible with the existing
project APIs:

    src.lexical_retrieval.search_lexical()
    src.vectordb.get_vectorstore()
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from typing import Any

from src.config import CHUNKS_PATH
from src.lexical_retrieval import search_lexical
from src.vectordb import get_vectorstore


# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------

DEFAULT_DENSE_K = 20
DEFAULT_LEXICAL_K = 20
DEFAULT_FINAL_K = 8

RRF_K = 60

DENSE_WEIGHT = 0.58
LEXICAL_WEIGHT = 0.42

EVIDENCE_BONUS = 0.08
CHAPTER_INTENT_BONUS = 0.12

MAX_RESULTS_PER_CHAPTER = 3


# ---------------------------------------------------------------------------
# Query / text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace for Bengali text matching."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Query intent detection
# ---------------------------------------------------------------------------

def detect_query_intent(question: str) -> str | None:
    """
    Detect a small number of retrieval intents.

    Important:
    - Explicit identity questions are handled separately.
    - Specific chapter/topic intents are checked afterward.
    - Broad words such as 'কেমন' must NOT turn a normal question into
      an identity query.
    """
    q = normalize_text(question)

    # Explicit identity / role / relationship questions.
    # Keep this deliberately narrow.
    if re.search(
        r"(কে\s+ছিল(?:েন)?|কে\s+ছিলেন|কে\s+ছিল|কে\s+ছিলো|"
        r"কে\s+ছিলে|কে\s+হিসেবে|কে\s+ছিল\s+|"
        r"কে\s+ছিলেন\s+|কী\s+ছিলেন|কী\s+ছিল|"
        r"কী\s+ভূমিকা|ভূমিকা\s+কি|ভূমিকা\s+কী|"
        r"পরিচয়|পরিচয়|সম্পর্ক\s+কি|সম্পর্ক\s+কী|"
        r"কার\s+সঙ্গে|কার\s+সাথে|কে\s+তার|কে\s+তাঁর)",
        q,
    ):
        return "identity"

    # Ending / death / final outcome.
    if re.search(
        r"(শেষ|শেষে|পরিণতি|মৃত্যু|মারা|মরণ|মরিয়া|মরে|"
        r"শেষ\s+পর্যন্ত|শেষ\s+অবস্থা|কী\s+হয়েছিল|কি\s+হয়েছিল)",
        q,
    ):
        return "ending"

    # Parvati's marriage / marriage-family questions.
    if re.search(
        r"(পার্ব্বতী|পার্বতী).{0,50}(বিবাহ|বিয়ে|বিয়ে|শ্বশুরবাড়ি|"
        r"শ্বশুরবাড়ি|পাত্র|স্বামী|সংসার)"
        r"|"
        r"(বিবাহ|বিয়ে|বিয়ে|শ্বশুরবাড়ি|শ্বশুরবাড়ি).{0,50}"
        r"(পার্ব্বতী|পার্বতী)",
        q,
    ):
        return "parvati_marriage"

    # Chandramukhi-specific questions.
    if "চন্দ্রমুখী" in q:
        return "chandramukhi"

    # Dharmadas-specific questions.
    if "ধর্মদাস" in q:
        return "dharmadas"

    # Friendship / Chunilal-related questions.
    if re.search(r"(চুনিলাল|চুনিবাবু|বন্ধু|বন্ধুত্ব)", q):
        return "friend"

    # Calcutta / Kolkata.
    if re.search(r"(কলিকাতা|কলকাতা)", q):
        return "calcutta"

    # Family / childhood.
    if re.search(r"(শৈশব|ছোটবেলা|বাল্য|পরিবার|পিতা|মাতা|জননী|বাবা|মা)", q):
        return "family"

    return "general"


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_query(question: str) -> list[str]:
    """
    Add a small number of Bengali paraphrases.

    Expansion is intentionally conservative so that it does not introduce
    unrelated retrieval terms.
    """
    q = normalize_text(question)
    queries = [question]

    expansions: dict[str, list[str]] = {
        "শেষ": ["পরিণতি", "মৃত্যু", "শেষ অবস্থা"],
        "পরিণতি": ["শেষ", "মৃত্যু", "শেষ অবস্থা"],
        "বিবাহ": ["বিয়ে", "বিয়ে", "শ্বশুরবাড়ি"],
        "বিয়ে": ["বিবাহ", "বিয়ে", "শ্বশুরবাড়ি"],
        "বিয়ে": ["বিবাহ", "বিয়ে", "শ্বশুরবাড়ি"],
        "পরিবার": ["বংশ", "ঘর", "সংসার"],
        "কে ছিলেন": ["পরিচয়", "ভূমিকা"],
        "কে ছিল": ["পরিচয়", "ভূমিকা"],
    }

    for trigger, additions in expansions.items():
        if trigger in q:
            queries.extend(
                f"{question} {addition}"
                for addition in additions
            )

    # Character-specific useful terms.
    if "চন্দ্রমুখী" in q:
        queries.extend(
            [
                "চন্দ্রমুখী পরিচয়",
                "চন্দ্রমুখী ভূমিকা",
                "চন্দ্রমুখী সম্পর্ক",
            ]
        )

    if "ধর্মদাস" in q:
        queries.extend(
            [
                "ধর্মদাস পরিচয়",
                "ধর্মদাস ভূমিকা",
                "ধর্মদাস দেবদাসের সঙ্গে",
            ]
        )

    return list(dict.fromkeys(queries))


# ---------------------------------------------------------------------------
# Character identity / role matching
# ---------------------------------------------------------------------------

KNOWN_CHARACTER_NAMES = (
    "চন্দ্রমুখী",
    "ধর্মদাস",
    "পার্বতী",
    "পার্ব্বতী",
    "দেবদাস",
    "চুনিলাল",
    "চুনিবাবু",
    "নীলকণ্ঠ",
    "ভুবনমোহন",
    "মহেন্দ্র",
)


# These are deliberately stricter than generic occupation/social words.
# For example, 'দাসী' should not count merely because it appears in a
# sentence containing 'চন্দ্রমুখী'.
STRICT_IDENTITY_TERMS = (
    "প্রজা",
    "জমিদার",
    "জমিদারের",
    "কন্যা",
    "পুত্র",
    "পিতা",
    "মাতা",
    "জননী",
    "স্ত্রী",
    "স্বামী",
    "ভৃত্য",
    "সেবক",
    "সেবিকা",
    "অনুচর",
    "সহচর",
    "সহচরী",
    "বন্ধু",
    "বন্ধুজন",
    "বাল্যবন্ধু",
    "পত্নী",
    "বিধবা",
    "গৃহিণী",
    "গৃহস্থ",
    "গৃহস্থের",
)


RELATIONSHIP_TERMS = (
    "সঙ্গে",
    "সাথে",
    "হাত দিয়া",
    "হাত দিয়ে",
    "নিয়ে",
    "লইয়া",
    "পাঠাইয়া",
    "পাঠিয়ে",
    "সেবা",
    "সেবাযত্ন",
    "সেবক",
    "ভৃত্য",
    "অনুচর",
    "সঙ্গী",
    "সহচর",
)


SPEECH_VERBS = (
    "বলিল",
    "বললেন",
    "বলিলেন",
    "বলিয়া",
    "বলতে",
    "কহিল",
    "কহিলেন",
    "কহিয়া",
    "কহতে",
    "জিজ্ঞাসা করিল",
    "জিজ্ঞাসা করলেন",
)


def extract_identity_target(question: str) -> str | None:
    """
    Extract the character whose identity/role is being requested.
    """
    q = normalize_text(question)

    # Prefer known names because Bengali questions can have many forms.
    for name in KNOWN_CHARACTER_NAMES:
        if name in q:
            return name

    return None


def _sentence_like_segments(text: str) -> list[str]:
    """
    Split Bengali prose into sentence-like segments.

    The source text is OCR/web-extracted prose, so sentence boundaries are
    not always perfect. We therefore also use local windows later.
    """
    text = normalize_text(text)

    if not text:
        return []

    segments = re.split(r"(?<=[।!?])\s+|[\n\r]+", text)

    return [segment.strip() for segment in segments if segment.strip()]


def _target_windows(
    text: str,
    target: str,
    window_chars: int = 180,
) -> list[str]:
    """
    Return local text windows around occurrences of the target.
    """
    normalized = normalize_text(text)
    target = normalize_text(target)

    if not normalized or not target:
        return []

    windows: list[str] = []

    for match in re.finditer(re.escape(target), normalized):
        start = max(0, match.start() - window_chars)
        end = min(len(normalized), match.end() + window_chars)
        windows.append(normalized[start:end])

    return windows


def _has_target_linked_role(sentence: str, target: str) -> bool:
    """
    Detect a role term only when it is locally attached to the target.

    This avoids false positives such as:

        চন্দ্রমুখী ... একজন দাসী ...
        ... জমিদার-গৃহিণী ...

    when the role actually belongs to another person.
    """
    sentence = normalize_text(sentence)
    target = normalize_text(target)

    if target not in sentence:
        return False

    # Direct apposition:
    #   ধর্মদাস একজন ভৃত্য
    #   চন্দ্রমুখী একজন প্রজা
    direct_pattern = (
        rf"{re.escape(target)}\s*(?:ছিল|ছিলেন|হয়|হয়|হলেন|হইল)?"
        rf".{{0,25}}?"
        rf"(?:একজন|একটি|একজনের)?\s*"
        rf"(?:{'|'.join(map(re.escape, STRICT_IDENTITY_TERMS))})"
    )

    if re.search(direct_pattern, sentence):
        return True

    # Reverse apposition:
    #   একজন প্রজা চন্দ্রমুখী
    reverse_pattern = (
        rf"(?:একজন|একটি)?\s*"
        rf"(?:{'|'.join(map(re.escape, STRICT_IDENTITY_TERMS))})"
        rf".{{0,25}}?"
        rf"{re.escape(target)}"
    )

    if re.search(reverse_pattern, sentence):
        return True

    return False


def _has_self_identification(sentence: str, target: str) -> bool:
    """
    Strong signal for statements where the character identifies themselves.

    Example:
        চন্দ্রমুখী কহিল, আমি আপনারই একজন প্রজা...
    """
    sentence = normalize_text(sentence)
    target = normalize_text(target)

    if target not in sentence or "আমি" not in sentence:
        return False

    # Character is speaking / introducing themselves.
    has_speech = any(verb in sentence for verb in SPEECH_VERBS)

    if not has_speech:
        return False

    # The role must occur reasonably close to the self-reference.
    for term in STRICT_IDENTITY_TERMS:
        if term not in sentence:
            continue

        target_pos = sentence.find(target)
        ami_pos = sentence.find("আমি")
        role_pos = sentence.find(term)

        if target_pos < 0 or ami_pos < 0 or role_pos < 0:
            continue

        if abs(ami_pos - role_pos) <= 100:
            return True

    return False


def _has_relationship_evidence(sentence: str, target: str) -> bool:
    """
    Detect useful contextual evidence about a character's role.

    This is intentionally weaker than formal identity evidence.
    """
    sentence = normalize_text(sentence)
    target = normalize_text(target)

    if target not in sentence:
        return False

    target_pos = sentence.find(target)

    for term in RELATIONSHIP_TERMS:
        term_pos = sentence.find(term)

        if term_pos < 0:
            continue

        # Require the relationship marker to be reasonably close to the
        # character mention.
        if abs(target_pos - term_pos) <= 120:
            return True

    return False


def identity_evidence_score(text: str, target: str) -> float:
    """
    Score identity/role evidence for one character.

    Score interpretation:
      0.15+  strong explicit identity/self-identification
      0.08   direct role/apposition evidence
      0.04   useful contextual relationship evidence
      0.00   no reliable identity evidence

    We intentionally do NOT count arbitrary role words in the same chunk.
    """
    if not text or not target:
        return 0.0

    sentences = _sentence_like_segments(text)

    strong = False
    direct = False
    relationship = False

    for sentence in sentences:
        if target not in sentence:
            continue

        if _has_self_identification(sentence, target):
            strong = True

        if _has_target_linked_role(sentence, target):
            direct = True

        if _has_relationship_evidence(sentence, target):
            relationship = True

    # Fallback to local windows because some source chunks have weak or
    # missing punctuation.
    if not strong or not direct or not relationship:
        for window in _target_windows(text, target):
            if not strong and _has_self_identification(window, target):
                strong = True

            if not direct and _has_target_linked_role(window, target):
                direct = True

            if not relationship and _has_relationship_evidence(window, target):
                relationship = True

    if strong:
        return 0.15

    if direct:
        return 0.085

    if relationship:
        return 0.04

    return 0.0


# ---------------------------------------------------------------------------
# Dense retrieval helpers
# ---------------------------------------------------------------------------

def _document_to_chunk(document: Any) -> dict:
    """
    Convert a LangChain Chroma Document into the chunk-dict representation
    used throughout the hybrid retriever.
    """
    metadata = dict(getattr(document, "metadata", {}) or {})

    return {
        "chunk_id": metadata.get("chunk_id", ""),
        "book_name": metadata.get("book_name", ""),
        "author": metadata.get("author", ""),
        "chapter_number": metadata.get("chapter_number", ""),
        "chapter_name": metadata.get("chapter_name", ""),
        "section": metadata.get("section", ""),
        "source_url": metadata.get("source_url", ""),
        "chunk_index_in_chapter": metadata.get(
            "chunk_index_in_chapter",
            "",
        ),
        "text": getattr(document, "page_content", ""),
    }


def _dense_search(
    question: str,
    k: int,
) -> list[tuple[dict, float]]:
    """
    Dense semantic retrieval from the persistent Chroma collection.
    """
    vectorstore = get_vectorstore()

    documents_with_scores = vectorstore.similarity_search_with_score(
        question,
        k=k,
    )

    results: list[tuple[dict, float]] = []

    for document, distance in documents_with_scores:
        chunk = _document_to_chunk(document)

        # Chroma's returned score is a distance. We only use it for
        # diagnostic purposes here; RRF is based on rank.
        results.append((chunk, float(distance)))

    return results


# ---------------------------------------------------------------------------
# Identity-focused lexical retrieval
# ---------------------------------------------------------------------------

def _identity_lexical_search(
    question: str,
    target: str,
    k: int = 20,
) -> list[tuple[dict, float]]:
    """
    Retrieve lexical candidates specifically containing the requested
    character.

    This gives identity questions another route to relevant chunks even
    when dense similarity under-ranks them.
    """
    lexical_results = search_lexical(question, k=max(k, 20))

    target = normalize_text(target)

    filtered: list[tuple[dict, float]] = []

    for chunk, score in lexical_results:
        text = normalize_text(chunk.get("text", ""))

        if target in text:
            filtered.append((chunk, float(score)))

    return filtered[:k]


# ---------------------------------------------------------------------------
# Chapter intent priors
# ---------------------------------------------------------------------------

def _chapter_intent_bonus(
    chunk: dict,
    intent: str | None,
) -> float:
    """
    Apply a small topic prior.

    This is deliberately weaker than direct retrieval evidence.
    """
    chapter_number = str(chunk.get("chapter_number", "")).strip()
    chapter_name = normalize_text(chunk.get("chapter_name", ""))

    if intent == "ending":
        if chapter_number == "16" or "ষোড়শ" in chapter_name:
            return CHAPTER_INTENT_BONUS

    elif intent == "chandramukhi":
        # Chandramukhi's main material is concentrated around chapters
        # 9, 13 and 15.
        if chapter_number in {"9", "13", "15"}:
            return CHAPTER_INTENT_BONUS

    elif intent == "dharmadas":
        if chapter_number in {"3", "4", "9", "11", "15"}:
            return CHAPTER_INTENT_BONUS

    elif intent == "parvati_marriage":
        # Early marriage discussion and later marriage context.
        if chapter_number in {"5", "7", "8", "14"}:
            return CHAPTER_INTENT_BONUS

    elif intent == "friend":
        if chapter_number in {"9", "10", "11", "13", "15"}:
            return CHAPTER_INTENT_BONUS

    elif intent == "calcutta":
        if chapter_number in {"4", "5", "6", "7", "8", "9", "10"}:
            return CHAPTER_INTENT_BONUS

    elif intent == "family":
        if chapter_number in {"1", "2", "3", "4", "5", "6", "7"}:
            return CHAPTER_INTENT_BONUS

    return 0.0


# ---------------------------------------------------------------------------
# Evidence scoring
# ---------------------------------------------------------------------------

def _general_evidence_score(
    chunk: dict,
    question: str,
) -> float:
    """
    Small generic evidence bonus for lexical overlap.

    This is intentionally modest and cannot overpower the RRF score.
    """
    text = normalize_text(chunk.get("text", ""))
    question_tokens = set(
        token
        for token in re.findall(r"[\w\u0980-\u09FF]+", normalize_text(question))
        if len(token) >= 2
    )

    if not text or not question_tokens:
        return 0.0

    matched = sum(1 for token in question_tokens if token in text)

    if matched >= 4:
        return EVIDENCE_BONUS

    if matched >= 2:
        return EVIDENCE_BONUS * 0.6

    if matched >= 1:
        return EVIDENCE_BONUS * 0.25

    return 0.0


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(
    question: str,
    k: int = DEFAULT_FINAL_K,
    dense_k: int = DEFAULT_DENSE_K,
    lexical_k: int = DEFAULT_LEXICAL_K,
) -> list[dict]:
    """
    Hybrid dense + lexical retrieval with lightweight intent-aware ranking.

    Returns a list of chunk dictionaries.
    """
    if not question or not question.strip():
        return []

    question = question.strip()

    intent = detect_query_intent(question)
    identity_target = (
        extract_identity_target(question)
        if intent == "identity"
        else None
    )

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    dense_results = _dense_search(
        question,
        k=max(dense_k, k),
    )

    # ------------------------------------------------------------------
    # Lexical retrieval
    # ------------------------------------------------------------------

    lexical_results = search_lexical(
        question,
        k=max(lexical_k, k),
    )

    # ------------------------------------------------------------------
    # Identity-specific lexical retrieval
    # ------------------------------------------------------------------

    identity_results: list[tuple[dict, float]] = []

    if identity_target:
        identity_results = _identity_lexical_search(
            question,
            identity_target,
            k=max(lexical_k, k),
        )

    # ------------------------------------------------------------------
    # Merge candidates
    # ------------------------------------------------------------------

    candidates: dict[str, dict] = {}

    def add_candidate(chunk: dict) -> str:
        chunk_id = str(chunk.get("chunk_id", ""))

        if not chunk_id:
            # Fallback key for malformed metadata.
            chunk_id = f"anonymous::{len(candidates)}"

        if chunk_id not in candidates:
            candidates[chunk_id] = dict(chunk)

        return chunk_id

    dense_rank: dict[str, int] = {}
    lexical_rank: dict[str, int] = {}
    identity_rank: dict[str, int] = {}

    for rank, (chunk, _distance) in enumerate(dense_results, start=1):
        chunk_id = add_candidate(chunk)
        dense_rank.setdefault(chunk_id, rank)

    for rank, (chunk, _score) in enumerate(lexical_results, start=1):
        chunk_id = add_candidate(chunk)
        lexical_rank.setdefault(chunk_id, rank)

    for rank, (chunk, _score) in enumerate(identity_results, start=1):
        chunk_id = add_candidate(chunk)
        identity_rank.setdefault(chunk_id, rank)

    # ------------------------------------------------------------------
    # Rank fusion
    # ------------------------------------------------------------------

    ranked: list[dict] = []

    for chunk_id, chunk in candidates.items():
        d_rank = dense_rank.get(chunk_id)
        l_rank = lexical_rank.get(chunk_id)
        i_rank = identity_rank.get(chunk_id)

        rrf_score = 0.0

        if d_rank is not None:
            rrf_score += DENSE_WEIGHT / (RRF_K + d_rank)

        if l_rank is not None:
            rrf_score += LEXICAL_WEIGHT / (RRF_K + l_rank)

        # Identity lexical retrieval is deliberately weak.
        # Its job is to keep character-containing evidence in contention,
        # not to dominate ordinary dense/lexical retrieval.
        if identity_target and i_rank is not None:
            rrf_score += 0.35 / (RRF_K + i_rank)

        # --------------------------------------------------------------
        # Evidence
        # --------------------------------------------------------------

        evidence_score = _general_evidence_score(
            chunk,
            question,
        )

        identity_score = 0.0

        if identity_target:
            identity_score = identity_evidence_score(
                chunk.get("text", ""),
                identity_target,
            )

        # --------------------------------------------------------------
        # Chapter intent
        # --------------------------------------------------------------

        chapter_bonus = 0.0

        # Identity questions should be driven primarily by evidence,
        # not chapter priors. This prevents unrelated chapter content
        # from being artificially promoted.
        if intent != "identity":
            chapter_bonus = _chapter_intent_bonus(
                chunk,
                intent,
            )

        # --------------------------------------------------------------
        # Final score
        # --------------------------------------------------------------

        final_score = (
            rrf_score
            + evidence_score
            + chapter_bonus
        )

        if identity_target:
            # Strong direct identity evidence should matter.
            #
            # 0.15 explicit self-identification
            # 0.085 direct role/apposition
            # 0.04 contextual relationship
            final_score += identity_score

        result = dict(chunk)

        result["_hybrid_score"] = float(final_score)
        result["_rrf_score"] = float(rrf_score)
        result["_evidence_score"] = float(evidence_score)
        result["_chapter_bonus"] = float(chapter_bonus)
        result["_identity_score"] = float(identity_score)
        result["_dense_rank"] = d_rank
        result["_lexical_rank"] = l_rank
        result["_identity_rank"] = i_rank

        ranked.append(result)

    # ------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------

    ranked.sort(
        key=lambda item: item["_hybrid_score"],
        reverse=True,
    )

    # ------------------------------------------------------------------
    # Chapter diversity
    # ------------------------------------------------------------------

    selected: list[dict] = []
    chapter_counts: dict[str, int] = {}

    for result in ranked:
        chapter = str(
            result.get("chapter_number")
            or result.get("chapter_name")
            or ""
        )

        count = chapter_counts.get(chapter, 0)

        if count >= MAX_RESULTS_PER_CHAPTER:
            continue

        selected.append(result)
        chapter_counts[chapter] = count + 1

        if len(selected) >= k:
            break

    # If diversity filtering removed too many results, fill remaining
    # slots from the ranked list.
    if len(selected) < k:
        selected_ids = {
            result.get("chunk_id")
            for result in selected
        }

        for result in ranked:
            if result.get("chunk_id") in selected_ids:
                continue

            selected.append(result)
            selected_ids.add(result.get("chunk_id"))

            if len(selected) >= k:
                break

    return selected[:k]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_chunk_count() -> int:
    """
    Return the number of source chunks.

    Useful for diagnostics/tests without loading Chroma embeddings.
    """
    if not CHUNKS_PATH.exists():
        return 0

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return len(chunks)