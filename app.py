"""
app.py
======

Streamlit interface for the Bangla Book RAG chatbot.
"""

from __future__ import annotations

import streamlit as st

from src.config import (
    BOOK_AUTHOR,
    BOOK_INDEX_URL,
    BOOK_TITLE,
    CHROMA_DIR,
    GEMINI_API_KEY,
    VECTORSTORE_MANIFEST_PATH,
    CHUNKS_PATH,
    logger,
)


st.set_page_config(
    page_title="Bangla Book RAG Chatbot",
    page_icon="📖",
    layout="centered",
)


def check_setup() -> list[str]:
    """Return human-readable setup problems."""

    problems: list[str] = []

    if not CHUNKS_PATH.exists():
        problems.append(
            "📄 Chunked book data not found. Run "
            "`python src/crawler.py` and then "
            "`python src/chunking.py`."
        )

    chroma_db_file = CHROMA_DIR / "chroma.sqlite3"

    if (
        not chroma_db_file.exists()
        or not VECTORSTORE_MANIFEST_PATH.exists()
    ):
        problems.append(
            "🗂️ Verified vector index not found. Run "
            "`python src/vectordb.py`."
        )

    if not GEMINI_API_KEY:
        problems.append(
            "🔑 GEMINI_API_KEY is not configured. "
            "Create `.env` from `.env.example` and add your Gemini API key."
        )

    return problems


def friendly_error_message(exc: Exception) -> str:
    """Convert common API errors into useful user-facing messages."""

    message = str(exc).upper()

    if "401" in message or "403" in message or "API KEY" in message:
        return (
            "দুঃখিত, Gemini API key সংক্রান্ত সমস্যা হয়েছে। "
            "আপনার `.env`-এর GEMINI_API_KEY পরীক্ষা করুন।"
        )

    if (
        "429" in message
        or "RESOURCE_EXHAUSTED" in message
        or "RATE LIMIT" in message
    ):
        return (
            "দুঃখিত, Gemini API-এর rate limit বা quota বর্তমানে "
            "শেষ হয়ে গেছে। কিছুক্ষণ পরে আবার চেষ্টা করুন।"
        )

    if any(
        code in message
        for code in ("500", "502", "503", "504")
    ):
        return (
            "দুঃখিত, Gemini পরিষেবায় সাময়িক সমস্যা হয়েছে। "
            "কিছুক্ষণ পরে আবার চেষ্টা করুন।"
        )

    return (
        "দুঃখিত, উত্তর তৈরি করার সময় একটি সমস্যা হয়েছে। "
        "logs/app.log দেখুন।"
    )


setup_problems = check_setup()


with st.sidebar:
    st.header("📖 বই সম্পর্কে")

    st.markdown(
        f"**শিরোনাম:** {BOOK_TITLE}"
    )

    st.markdown(
        f"**লেখক:** {BOOK_AUTHOR}"
    )

    st.markdown(
        f"[উইকিসংকলনে দেখুন ↗]({BOOK_INDEX_URL})"
    )

    st.divider()

    st.caption(
        "এই চ্যাটবট শুধুমাত্র দেবদাস উপন্যাসের retrieved context-এর "
        "উপর ভিত্তি করে উত্তর দেয়।"
    )

    st.divider()

    if st.button("🗑️ নতুন কথোপকথন শুরু করুন"):
        st.session_state.messages = []
        st.rerun()


st.title("📖 Bangla Book RAG Chatbot")
st.caption(
    "Retrieval-Augmented Generation (RAG) ভিত্তিক বাংলা বই চ্যাটবট"
)


if setup_problems:
    st.warning(
        "চ্যাটবট শুরু করার আগে নিচের ধাপগুলো সম্পূর্ণ করুন:"
    )

    for problem in setup_problems:
        st.markdown(f"- {problem}")

    st.stop()


from src.rag_pipeline import ask  # noqa: E402


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 সূত্র (Sources)"):
                for source in message["sources"]:
                    chapter_name = source["chapter_name"]
                    source_url = source["source_url"]

                    if source_url:
                        st.markdown(
                            f"- **{chapter_name}** — "
                            f"[উৎস দেখুন ↗]({source_url})"
                        )
                    else:
                        st.markdown(
                            f"- **{chapter_name}**"
                        )


question = st.chat_input(
    "বইটি সম্পর্কে একটি প্রশ্ন করুন..."
)


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("বইটি খুঁজে দেখছি..."):
            try:
                result = ask(question)

                answer = result["answer"]
                sources = result["sources"]

            except Exception as exc:
                logger.exception(
                    "Error answering question %r",
                    question,
                )

                answer = friendly_error_message(exc)
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📚 সূত্র (Sources)"):
                for source in sources:
                    chapter_name = source["chapter_name"]
                    source_url = source["source_url"]

                    if source_url:
                        st.markdown(
                            f"- **{chapter_name}** — "
                            f"[উৎস দেখুন ↗]({source_url})"
                        )
                    else:
                        st.markdown(
                            f"- **{chapter_name}**"
                        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )