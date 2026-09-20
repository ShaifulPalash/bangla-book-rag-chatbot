"""
app.py
=======
The Streamlit chat interface for the দেবদাস RAG chatbot. This is the file
you actually run to talk to the chatbot:

    streamlit run app.py

What this file does, in plain English:
-----------------------------------------
1. Checks that the earlier pipeline steps have actually been run (crawled
   data exists, vector index is built, API key is set) and shows a clear,
   friendly message instead of crashing if something's missing.
2. Keeps track of the conversation using Streamlit's `session_state` —
   Streamlit re-runs this whole script from top to bottom every time you
   interact with the page, so without session_state, your chat history
   would disappear after every single message.
3. For each question, calls rag_pipeline.ask() and displays the answer
   plus its chapter citation(s), or a clear "not found in book" message.
"""

import streamlit as st

from src.config import BOOK_TITLE, BOOK_AUTHOR, BOOK_INDEX_URL, GEMINI_API_KEY, CHUNKS_PATH, CHROMA_DIR, logger

st.set_page_config(
    page_title="Bangla Book RAG Chatbot",
    page_icon="📖",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Pre-flight checks: catch a not-yet-built pipeline BEFORE trying to chat,
# and explain exactly which command to run to fix it. Without this, a
# missing vector index or API key would surface as a confusing crash deep
# inside rag_pipeline.py instead of a clear instruction here.
# ---------------------------------------------------------------------------
def check_setup() -> list[str]:
    problems = []
    if not CHUNKS_PATH.exists():
        problems.append(
            "📄 Book data not found. Run `python src/crawler.py` then "
            "`python src/chunking.py` first."
        )
    # Chroma creates a "chroma.sqlite3" file when it actually builds an
    # index. Just checking "is the folder non-empty" isn't reliable — the
    # folder always contains a .gitkeep placeholder file (so git tracks
    # the otherwise-empty directory), which would make this check think
    # a real index exists even on a completely fresh clone.
    chroma_db_file = CHROMA_DIR / "chroma.sqlite3"
    if not chroma_db_file.exists():
        problems.append(
            "🗂️ Vector index not found. Run `python src/vectordb.py` first "
            "(this embeds the book with bge-m3 — may take a few minutes)."
        )
    if not GEMINI_API_KEY:
        problems.append(
            "🔑 GEMINI_API_KEY not set. Copy `.env.example` to `.env` and "
            "add your free key from https://aistudio.google.com/apikey"
        )
    return problems


setup_problems = check_setup()

# ---------------------------------------------------------------------------
# Sidebar: book info + a way to start a fresh conversation.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📖 বই সম্পর্কে")
    st.markdown(f"**শিরোনাম:** {BOOK_TITLE}")
    st.markdown(f"**লেখক:** {BOOK_AUTHOR}")
    st.markdown(f"[উইকিসংকলনে দেখুন ↗]({BOOK_INDEX_URL})")
    st.divider()
    st.caption(
        "এই চ্যাটবট শুধুমাত্র উপরের বইয়ের তথ্যের উপর ভিত্তি করে উত্তর দেয়, "
        "এবং প্রতিটি উত্তরের সাথে সংশ্লিষ্ট অধ্যায় উল্লেখ করে।"
    )
    st.divider()
    if st.button("🗑️ নতুন কথোপকথন শুরু করুন"):
        st.session_state.messages = []
        st.rerun()

st.title("📖 Bangla Book RAG Chatbot")
st.caption("একটি Retrieval-Augmented Generation (RAG) চালিত জ্ঞানভিত্তিক চ্যাটবট")

# ---------------------------------------------------------------------------
# If the pipeline hasn't been fully set up yet, show what's missing and
# stop here — no point rendering a chat box that will just error out.
# ---------------------------------------------------------------------------
if setup_problems:
    st.warning("চ্যাটবট শুরু করার আগে নিচের ধাপগুলো সম্পূর্ণ করুন:")
    for problem in setup_problems:
        st.markdown(f"- {problem}")
    st.stop()

# Importing rag_pipeline is deferred until after the setup check passes,
# so an incomplete setup shows our friendly message instead of an import-
# time crash from inside rag_pipeline.py (e.g. a missing API key raising
# an error while loading the Gemini client).
from src.rag_pipeline import ask  # noqa: E402

# ---------------------------------------------------------------------------
# Chat history, kept in session_state so it survives Streamlit's rerun-on-
# every-interaction behavior. Each entry is a dict: role, content, and
# (for assistant messages) the sources used.
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render the full conversation so far on every rerun.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 সূত্র (Sources)"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['chapter_name']}** — [লিঙ্ক ↗]({src['source_url']})")

# ---------------------------------------------------------------------------
# The actual chat input. st.chat_input returns None until the user submits
# a message, so this block only runs when there's something new to answer.
# ---------------------------------------------------------------------------
question = st.chat_input("বইটি সম্পর্কে একটি প্রশ্ন করুন...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("বইটি খুঁজে দেখছি..."):
            try:
                result = ask(question)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                # A real error (rate limit exhausted after retries, auth
                # failure, etc.) — log the full detail for debugging, but
                # show the user a short, non-technical message.
                logger.error(f"Error answering question '{question}': {e}")
                answer = (
                    "দুঃখিত, একটি সমস্যা হয়েছে। কিছুক্ষণ পর আবার চেষ্টা করুন। "
                    f"(বিস্তারিত লগে দেখুন: logs/app.log)"
                )
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("📚 সূত্র (Sources)"):
                for src in sources:
                    st.markdown(f"- **{src['chapter_name']}** — [লিঙ্ক ↗]({src['source_url']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
