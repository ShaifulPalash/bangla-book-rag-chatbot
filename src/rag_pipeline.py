"""
src/rag_pipeline.py
===================
The core of the chatbot: retrieves relevant passages from the Chroma
vector store and asks Gemini to answer strictly from those passages.

Only this file makes an external API call to Gemini. Crawling, chunking,
embedding, and vector indexing run locally.
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from src.config import GEMINI_API_KEY, LLM_MODEL, TOP_K, logger
from src.hybrid_retrieval import hybrid_search 


NOT_FOUND_PHRASE = "এই তথ্যটি বইয়ে পাওয়া যায়নি।"


SYSTEM_PROMPT = f"""তুমি একজন সতর্ক সহায়ক, যে শুধুমাত্র "দেবদাস"
(শরৎচন্দ্র চট্টোপাধ্যায় রচিত) উপন্যাসের নিচে দেওয়া প্রসঙ্গ (Context)
থেকে প্রশ্নের উত্তর দাও।

কঠোর নিয়মাবলী:

1. শুধুমাত্র Context-এ দেওয়া তথ্য ব্যবহার করবে। Context-এর বাইরে থেকে
কোনো তথ্য, সাধারণ জ্ঞান, বা তোমার পূর্বজ্ঞান ব্যবহার করবে না।

2. Context-এর একাধিক অংশ একসাথে বিবেচনা করবে। কোনো একটি passage-এ
সম্পূর্ণ উত্তর না থাকলেও, একাধিক retrieved passage মিলিয়ে যদি প্রশ্নের
উত্তর সমর্থন করা যায়, তাহলে উত্তর দেবে।

3. Context-এ তথ্যটি সরাসরি অথবা একাধিক passage-এর সমন্বয়ে স্পষ্টভাবে
বোঝা গেলে উত্তর দেবে। শুধুমাত্র এই কারণে প্রত্যাখ্যান করবে না যে
প্রশ্নের উত্তরটি Context-এ হুবহু একই বাক্যে লেখা নেই।

4. Context-এ প্রশ্নের উত্তর দেওয়ার মতো পর্যাপ্ত তথ্য না থাকলে ঠিক এই
বাক্যটি লিখবে এবং আর কিছু লিখবে না:

"{NOT_FOUND_PHRASE}"

5. উত্তর সবসময় বাংলায় দেবে।

6. উত্তর সংক্ষিপ্ত কিন্তু অর্থপূর্ণ হবে এবং প্রশ্নের সরাসরি উত্তর দেবে।
প্রশ্নে যে বিষয়টি জানতে চাওয়া হয়েছে, সেটিই স্পষ্টভাবে উল্লেখ করবে।
শুধু সংশ্লিষ্ট ঘটনা বা পরোক্ষ তথ্য বর্ণনা করে থেমে যাবে না।

7. প্রশ্নটি যদি কোনো সম্পর্ক, কারণ, ব্যক্তি, ঘটনা, বৈশিষ্ট্য বা পরিস্থিতি
সম্পর্কে হয়, তাহলে Context-এর একাধিক passage প্রয়োজন হলে সেগুলো
একত্র করে প্রশ্নে চাওয়া নির্দিষ্ট তথ্যটি স্পষ্টভাবে প্রকাশ করবে।
তবে Context-এ সরাসরি সমর্থন নেই এমন কোনো ব্যাখ্যা বা দাবি যোগ করবে না।

8. উত্তরের শেষে শুধুমাত্র যে অধ্যায়গুলো থেকে উত্তরের তথ্য নেওয়া হয়েছে,
সেগুলোর নাম এই ফরম্যাটে উল্লেখ করবে:

(সূত্র: <অধ্যায়ের নাম>)

9. কোনো তথ্য অনুমান করবে না এবং Context-এর বাইরে থেকে চরিত্র, ঘটনা,
সম্পর্ক বা ব্যাখ্যা যোগ করবে না।

10. বইয়ের পুরোনো বাংলা বানান বা ভাষারীতি থাকলেও তার অর্থ বুঝে উত্তর
দিতে পারো, কিন্তু Context-এর তথ্য পরিবর্তন করবে না।

Context:
{{context}}
"""


_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Lazily initialize the Gemini LLM.

    Gemini is initialized only when a question is actually asked, so
    crawling, chunking, embedding, and vector database creation can run
    without requiring a Gemini API key.
    """
    global _llm

    if _llm is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file "
                "(see .env.example) before asking questions."
            )

        logger.info(f"Initializing Gemini LLM: {LLM_MODEL}")

        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
        )

    return _llm


def format_context(docs_with_scores: list[tuple]) -> str:
    """
    Convert retrieved documents into a single context block for Gemini.

    Each passage is labeled with its number and chapter name so the model
    can clearly identify where the information came from.
    """
    parts = []

    for index, (doc, _score) in enumerate(docs_with_scores, start=1):
        chapter = doc.metadata.get("chapter_name", "অজানা অধ্যায়")

        parts.append(
            f"[Passage {index} | অধ্যায়: {chapter}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _invoke_llm(messages) -> str:
    """
    Send the prompt to Gemini with automatic retry handling.

    Retries help recover from temporary API failures such as HTTP 503.
    """
    llm = get_llm()
    chain = llm | StrOutputParser()

    return chain.invoke(messages)


def ask(question: str, top_k: int = None) -> dict:
    """
    Retrieve relevant passages and generate a grounded answer.

    Parameters
    ----------
    question:
        User's question about the book.

    top_k:
        Number of passages to retrieve. If omitted, the value from
        config.py is used.

    Returns
    -------
    dict
        Contains the generated answer, source chapters, refusal status,
        and retrieved passage information.
    """
    k = top_k or TOP_K

    results = hybrid_search(question, k=k)

    logger.info(f"Question: {question!r}")

    for doc, score in results:
        preview = doc.page_content[:60].replace("\n", " ")

        logger.info(
    "Retrieved chunk: rrf_score=%.4f | chapter=%s | preview=%s",
    score,
    doc.metadata.get("chapter_name"),
    doc.page_content[:120].replace("\n", " "),
    )

    context = format_context(results)

    messages = _prompt_template.format_messages(
        context=context,
        question=question,
    )

    answer = _invoke_llm(messages).strip()

    # Treat the response as a refusal only when the model returns the
    # exact required refusal sentence.
    refused = answer == NOT_FOUND_PHRASE

    if refused:
        sources = []
        logger.info("  -> Model reported: answer not found in book.")

    else:
        seen = set()
        sources = []

        for doc, _score in results:
            chapter_name = doc.metadata.get("chapter_name")

            if chapter_name not in seen:
                seen.add(chapter_name)

                sources.append(
                    {
                        "chapter_name": chapter_name,
                        "source_url": doc.metadata.get("source_url"),
                    }
                )

        logger.info(
            f"  -> Answered, citing {len(sources)} retrieved chapter(s)."
        )

    return {
        "answer": answer,
        "sources": sources,
        "refused": refused,
        "retrieved": [
            {
                "chapter_name": doc.metadata.get("chapter_name"),
                "score": float(score),
                "text_preview": doc.page_content[:100],
            }
            for doc, score in results
        ],
    }


if __name__ == "__main__":
    # Quick manual smoke test from the command line:
    #
    # python -m src.rag_pipeline
    #
    # Or:
    #
    # python -m src.rag_pipeline "দেবদাস ও পার্বতীর সম্পর্ক কেমন ছিল?"

    import sys

    question = (
        " ".join(sys.argv[1:])
        or "দেবদাস ও পার্বতীর সম্পর্ক কেমন ছিল?"
    )

    result = ask(question)

    print("\nপ্রশ্ন:", question)
    print("উত্তর:", result["answer"])
    print("সূত্র:", result["sources"])