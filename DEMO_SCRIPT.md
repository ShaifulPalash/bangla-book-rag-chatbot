# Demo Video Script — দেবদাস RAG চ্যাটবট

A checklist for recording the required 3–5 minute demo video. Follow the
three parts in order; approximate timings add up to ~4 minutes, leaving
room to breathe.

**Before you hit record:**
- [ ] `chroma_db/chroma.sqlite3` exists (vector index is built)
- [ ] `.env` has your real `GEMINI_API_KEY`
- [ ] Run `streamlit run app.py` and confirm it loads with **no setup
      warnings** — if you see any, fix them before recording
- [ ] Close other noisy browser tabs/notifications
- [ ] Have this checklist open on a second screen or printed, so you're
      not improvising on camera

---

## Part 1 — Pipeline (≈45 seconds)

**Goal:** briefly prove the RAG pipeline is real and running, not a
hardcoded demo.

- [ ] Show the terminal running `python src/crawler.py` (can be sped up /
      cut to the final summary log line if you already ran it earlier —
      just show that it *has* run and what it logged)
- [ ] Show `data/debdas_chunks.json` briefly — scroll past a couple of
      chunks so the viewer sees real Bengali text with metadata fields
      (chapter_name, source_url)
- [ ] Show the terminal output of `python src/vectordb.py` (or its final
      log line: `Collection now contains N chunks`)
- [ ] Launch the app: `streamlit run app.py` — show it opening cleanly in
      the browser with no setup warnings

**Say out loud (approximate):** *"This chatbot is built with a full RAG
pipeline — it crawls all 16 chapters of দেবদাস from Bengali Wikisource,
splits them into chunks, embeds them locally with a multilingual model
called bge-m3, and stores them in a Chroma vector database. Let's see it
answer some questions."*

---

## Part 2 — Questions (≈2 minutes, at least 5 questions)

Ask at least 5 real, in-book questions, live, one at a time. Suggested
picks from the required 10 test questions (mix a couple of straightforward
ones with a couple of more specific ones, so the variety is visible):

- [ ] দেবদাস ও পার্বতীর মধ্যে ছোটবেলার সম্পর্ক কেমন ছিল?
- [ ] কলিকাতায় দেবদাসের ঘনিষ্ঠ বন্ধুর নাম কী ছিল?
- [ ] চন্দ্রমুখী কে ছিলেন?
- [ ] ধর্মদাস কে ছিলেন?
- [ ] উপন্যাসের শেষে দেবদাসের কী পরিণতি হয়েছিল?

For **at least one** of these, click open the "📚 সূত্র (Sources)"
expander on camera and point out the chapter name and the clickable
Wikisource link — this is the part that directly shows the assignment's
"cite the relevant chapter/section" requirement working.

**Say out loud (approximate):** *"Notice every answer comes with a
citation showing exactly which chapter it was pulled from — you can click
through to the source on Wikisource."*

---

## Part 3 — No-Answer Case (≈30 seconds, at least 1 question)

- [ ] Ask a deliberately out-of-book question live — e.g. *"দেবদাস কোন
      স্মার্টফোন ব্র্যান্ড ব্যবহার করত?"* (test question #10)
- [ ] Let the chatbot respond on camera
- [ ] Point out that it clearly says the answer isn't in the book,
      **instead of inventing something** — and that no source citation
      appears alongside the refusal

**Say out loud (approximate):** *"This is the core rule this chatbot
follows — it never makes things up. If the book doesn't contain the
answer, it says so directly, instead of hallucinating."*

---

## Wrap-up (≈15 seconds, optional but nice)

- [ ] One sentence on the tech stack: *"Everything here runs on free
      tools — local embeddings with bge-m3, Chroma as the vector
      database, and Gemini's free tier for the final answer."*

---

## After recording

- [ ] Upload the video (YouTube unlisted, Google Drive, or similar)
- [ ] Paste the link into the README's [Demo Video](README.md#-demo-video) section
- [ ] Save at least 3 screenshots into `screenshots/` matching the
      filenames the README expects:
  - `pipeline_running.png` — terminal/logs from Part 1
  - `sample_answer.png` — a chat answer with the sources expander open
  - `no_answer_case.png` — the refusal response from Part 3
