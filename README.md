<div align="center">

# 📖 Bangla Book RAG Chatbot

### বাংলা বইভিত্তিক Retrieval-Augmented Generation চ্যাটবট

A Bengali Knowledge-Base Chatbot Powered by Retrieval-Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square\&logo=python\&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat-square\&logo=langchain\&logoColor=white)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/VectorDB-Chroma-FF6F61?style=flat-square)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%203.6%20Flash-4285F4?style=flat-square\&logo=googlegemini\&logoColor=white)](https://ai.google.dev/)
[![Hugging Face](https://img.shields.io/badge/Embeddings-Hugging%20Face-FFD21E?style=flat-square\&logo=huggingface\&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square\&logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#-license)

**Ask questions about the Bengali novel *দেবদাস* by শরৎচন্দ্র চট্টোপাধ্যায় and receive answers grounded in the book with chapter-level citations.**

</div>

---

## 📑 Table of Contents

* [Overview](#-overview)
* [Key Features](#-key-features)
* [Book Information](#-book-information)
* [System Architecture](#-system-architecture)
* [How the RAG Pipeline Works](#-how-the-rag-pipeline-works)
* [Project Structure](#-project-structure)
* [Technology Stack](#-technology-stack)
* [Setup & Installation](#-setup--installation)
* [Configuration](#-configuration)
* [Running the Project](#-running-the-project)
* [Technical Details](#-technical-details)

  * [Book Ingestion](#book-ingestion)
  * [Chunking Strategy](#chunking-strategy)
  * [Embedding Model](#embedding-model)
  * [Vector Database](#vector-database)
  * [Retriever & LLM](#retriever--llm)
  * [Prompt & Grounding](#prompt--grounding)
  * [Logging & Error Handling](#logging--error-handling)
* [Testing](#-testing)
* [Screenshots](#-screenshots)
* [Demo Video](#-demo-video)
* [Retrieval Comparison Experiment](#-retrieval-comparison-experiment)
* [Future Improvements](#-future-improvements)
* [License](#-license)

---

## 📖 Overview

This project implements a **Knowledge Base Chatbot** using a Retrieval-Augmented Generation (RAG) architecture.

The chatbot retrieves relevant passages from a Bengali prose book and uses **Google Gemini** to generate answers grounded exclusively in the retrieved book content.

The retrieval layer uses a **hybrid search architecture** that combines:

- **BGE-M3** dense semantic retrieval
- **BM25** lexical retrieval
- **Weighted Reciprocal Rank Fusion (RRF)**
- Configurable **Top-K** passage selection

This approach combines semantic understanding with exact lexical matching, which is particularly useful for Bengali text, character names, and explicit terms appearing in the source material.

The application provides a simple **Streamlit chat interface** where users can ask questions in Bengali.

---

## 🎯 Key Features

- 📚 Bengali book knowledge base
- 🕷️ Automated crawling from Bengali Wikisource
- 🧹 HTML cleaning and text preprocessing
- ✂️ Chapter-aware text chunking
- 🧠 Local **BAAI/bge-m3** embeddings
- 🔎 Hybrid retrieval using:
  - BGE-M3 dense search
  - BM25 lexical search
- 🔀 Weighted Reciprocal Rank Fusion (RRF)
- 📌 Configurable Top-K retrieval
- 🤖 Google Gemini for grounded answer generation
- 🛡️ Refuses questions when the requested information is not supported by the book
- 📖 Chapter-based source attribution
- 💾 Persistent ChromaDB vector store
- 💬 Streamlit-based conversational UI
- 🔁 Retry handling for transient LLM/API failures
- 🧪 Offline/unit testing without consuming Gemini API quota
- 🌐 Optional live end-to-end Gemini evaluation
- 📝 Application logging

---

## 🏗️ System Architecture

```text
                    Bengali Book
                         │
                         ▼
              Bengali Wikisource
                         │
                         ▼
                    Web Crawler
                         │
                         ▼
                 HTML Cleaning
                         │
                         ▼
               Chapter-aware Chunking
                         │
                         ▼
                 ┌───────────────┐
                 │ Book Chunks   │
                 └───────────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      BGE-M3 Embeddings          BM25 Index
             │                       │
             ▼                       ▼
      Dense Retrieval          Lexical Retrieval
             │                       │
             └───────────┬───────────┘
                         ▼
                 Weighted RRF Fusion
                         │
                         ▼
                     Top-K Passages
                         │
                         ▼
                  Grounded Prompt
                         │
                         ▼
                  Google Gemini
                         │
                         ▼
                  Bengali Answer
                         │
                         ▼
                 Chapter Citations
````

---

## 🔄 How the RAG Pipeline Works

### 1. Data Collection

The crawler retrieves the book and its chapter subpages from Bengali Wikisource.

The collected content is stored locally as structured JSON data.

```text
data/raw_debdas.json
```

---

### 2. Text Cleaning

HTML content is processed to remove irrelevant elements such as:

* Navigation elements
* Scripts
* Styles
* Page controls
* Other non-book content

The result is clean Bengali prose suitable for downstream processing.

---

### 3. Text Chunking

The cleaned book text is divided into smaller chunks while preserving important metadata.

Each chunk contains information such as:

* Book name
* Author
* Chapter number
* Chapter name
* Section
* Source URL
* Chunk index
* Text content

The processed chunks are stored in:

```text
data/debdas_chunks.json
```

---

### 4. Dense Retrieval

The project uses:

**`BAAI/bge-m3`**

to generate local embeddings for the book chunks.

Dense retrieval captures **semantic similarity**, allowing the system to retrieve relevant passages even when the wording of the question differs from the wording in the book.

The embeddings are stored in a persistent **ChromaDB** vector database.

---

### 5. Lexical Retrieval

The project also uses **BM25** for lexical retrieval.

BM25 is useful when the query contains:

* Character names
* Specific Bengali terms
* Explicit phrases
* Words that appear directly in the source text

Bengali text is normalized and tokenized using Unicode-aware preprocessing before BM25 indexing.

---

### 6. Hybrid Retrieval

Dense and lexical retrieval are combined using **Weighted Reciprocal Rank Fusion (RRF)**.

```text
Hybrid Retrieval
├── BGE-M3 dense retrieval
└── BM25 lexical retrieval
        ↓
Weighted RRF
        ↓
Top-K passages
```

This allows the system to benefit from both semantic and exact lexical matching.

The current default configuration retrieves:

```text
TOP_K = 6
```

---

### 7. Grounded Answer Generation

The retrieved passages are inserted into a structured prompt and sent to Google Gemini.

The model is instructed to:

1. Answer only from the provided book context.
2. Combine information from multiple retrieved passages when necessary.
3. Respond in Bengali.
4. Mention relevant chapter names.
5. Avoid inventing information.
6. Return a predefined refusal message when the answer is not supported by the book.

The refusal phrase is:

```text
এই তথ্যটি বইয়ে পাওয়া যায়নি।
```

---

### 8. Source Attribution

For supported questions, the application displays the relevant chapter names associated with the retrieved context.

This provides users with a transparent indication of where the answer was grounded.

---

## 📚 Book Information

The current knowledge base uses:

| Property        | Value                   |
| --------------- | ----------------------- |
| Book            | দেবদাস                  |
| Author          | শরৎচন্দ্র চট্টোপাধ্যায় |
| Source          | Bengali Wikisource      |
| Chapters        | 16                      |
| Language        | Bengali                 |
| Retrieval       | Hybrid Dense + BM25     |
| Embedding Model | BAAI/bge-m3             |
| Vector Database | ChromaDB                |
| LLM             | Google Gemini           |

---

## 🛠️ Technology Stack

| Component         | Technology                      |
| ----------------- | ------------------------------- |
| Language          | Python 3.12+                    |
| Data Source       | Bengali Wikisource              |
| Web Crawling      | Requests + BeautifulSoup        |
| Text Splitting    | LangChain Text Splitters        |
| Embeddings        | BAAI/bge-m3                     |
| Dense Retrieval   | ChromaDB                        |
| Lexical Retrieval | BM25                            |
| Hybrid Fusion     | Weighted Reciprocal Rank Fusion |
| LLM               | Google Gemini                   |
| RAG Framework     | LangChain                       |
| UI                | Streamlit                       |
| Testing           | Pytest                          |
| Configuration     | python-dotenv                   |
| Logging           | Python logging                  |

---

## 📁 Project Structure

```text
bangla-book-rag-chatbot/
│
├── data/
│   ├── raw_debdas.json
│   └── debdas_chunks.json
│
├── chroma_db/
│   ├── chroma.sqlite3
│   └── ...
│
├── docs/
│   └── test_results.md
│
├── logs/
│   └── app.log
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── crawler.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectordb.py
│   ├── lexical_retrieval.py
│   ├── hybrid_retrieval.py
│   └── rag_pipeline.py
│
├── tests/
│   └── test_retrieval.py
│
├── .env
├── .env.example
├── .gitignore
├── app.py
├── DEMO_SCRIPT.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ShaifulPalash/bangla-book-rag-chatbot.git
cd bangla-book-rag-chatbot
```

---

### 2. Create a Virtual Environment

Create a dedicated Python virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Verify Python:

```bash
python --version
```

Python 3.12+ is recommended.

---

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Open the file:

```bash
nano .env
```

Add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Do not commit `.env` to Git.

The `.gitignore` file already excludes it from version control.

---

## 🔑 API Key

The application requires a Google Gemini API key for answer generation.

The API key is used only for the LLM generation stage. The embedding model runs locally using `BAAI/bge-m3`.

This design reduces dependency on external embedding APIs and helps minimize API usage.

---

## 🔧 Configuration

Application settings are managed in:

```text
src/config.py
```

The current default retrieval configuration includes:

```python
TOP_K = int(os.getenv("TOP_K", "6"))
```

This means the system retrieves **6 passages by default** after hybrid retrieval and RRF fusion.

You can change the value through the environment:

```env
TOP_K=6
```

Increasing `TOP_K` may provide more context to the LLM, while a smaller value can reduce the amount of retrieved context.

---

## ▶️ Running the Project

The project should be run in the following order when building the knowledge base from scratch.

### Step 1 — Crawl the Book

Run:

```bash
python src/crawler.py
```

This retrieves the book chapters and saves the raw content to:

```text
data/raw_debdas.json
```

---

### Step 2 — Create Chunks

Run:

```bash
python src/chunking.py
```

This processes the raw book content and creates chapter-aware text chunks.

Output:

```text
data/debdas_chunks.json
```

---

### Step 3 — Build the Vector Database

Run:

```bash
python src/vectordb.py
```

This:

* Loads the processed chunks
* Generates local BGE-M3 embeddings
* Stores the embeddings in ChromaDB
* Creates the persistent vector database

Output:

```text
chroma_db/
```

The BM25 lexical index is built automatically at runtime from:

```text
data/debdas_chunks.json
```

---

### Step 4 — Start the Streamlit Application

Run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal.

Open that URL in your browser to use the chatbot.

---

## 🔄 Complete Rebuild

If you want to rebuild the complete knowledge base from scratch:

```bash
python src/crawler.py
python src/chunking.py
python src/vectordb.py
streamlit run app.py
```

---

## 💬 Example Questions

The chatbot is designed to answer questions that can be supported by the book.

Examples:

```text
দেবদাসের শৈশব কেমন ছিল?

দেবদাস ও পার্বতীর সম্পর্ক কেমন ছিল?

দেবদাসের পিতার সাথে তার সম্পর্ক কেমন ছিল?

চন্দ্রমুখী দেবদাসের জীবনে কী ভূমিকা পালন করেছিল?
```

For information that cannot be supported by the book, the chatbot should respond:

```text
এই তথ্যটি বইয়ে পাওয়া যায়নি।
```

---

## 🧪 Testing

The project separates **offline/unit testing** from **live Gemini evaluation**.

This prevents routine test execution from unnecessarily consuming Gemini API quota.

### Run Offline Tests

Run:

```bash
pytest -q
```

These tests are intended to run without making live Gemini API requests.

They can be used during normal development and CI workflows.

---

### Check Test Collection

To verify which tests are being collected:

```bash
pytest -q --collect-only
```

---

### Run Live Gemini Evaluation

For actual end-to-end evaluation using the Gemini API:

```bash
RUN_LIVE_TESTS=1 python tests/test_retrieval.py
```

This performs live evaluation using the configured Gemini API key.

Live evaluation requires:

* A valid `GOOGLE_API_KEY`
* Internet connectivity
* Available Gemini API quota
* Access to the configured Gemini model

Because live tests consume API quota, they should be run intentionally rather than on every local test execution.

---

## 📊 Evaluation Dataset

The retrieval evaluation contains questions designed to test both supported and unsupported queries.

The test set includes:

* In-book questions
* Character-related questions
* Relationship-based questions
* Chapter-specific questions
* An out-of-book question

For supported questions, the evaluation checks that:

* The chatbot does not refuse the question.
* A non-empty answer is generated.
* Relevant sources are returned.
* Expected Bengali keywords are present.

For unsupported questions, the evaluation checks that the chatbot returns the predefined refusal response.

Detailed test information is documented in:

```text
docs/test_results.md
```

---

## 🔍 Retrieval Design

### Dense Retrieval

Dense retrieval uses BGE-M3 embeddings to identify semantically related passages.

Advantages:

* Handles semantic similarity
* Works when query wording differs from source wording
* Useful for conceptual questions

---

### BM25 Retrieval

BM25 performs lexical matching between the question and book chunks.

Advantages:

* Strong exact-term matching
* Useful for names and explicit terms
* Helps retrieve passages containing important Bengali words

---

### Weighted RRF

The results from both retrievers are combined using Weighted Reciprocal Rank Fusion.

Conceptually:

```text
RRF Score =
    Dense Weight / (RRF_K + Dense Rank)
    +
    Lexical Weight / (RRF_K + Lexical Rank)
```

The current configuration uses:

```text
RRF_K = 60

Dense Weight   = 1.0
Lexical Weight = 2.0
```

The higher lexical weight gives BM25 a stronger contribution to the final ranking.

---

## 🧠 Why Hybrid Retrieval?

A Bengali knowledge base benefits from combining semantic and lexical retrieval.

For example, a question may contain an important character name or Bengali term that appears explicitly in the source. BM25 can strongly identify such passages, while dense retrieval can capture relevant passages even when the wording differs.

The hybrid approach therefore provides two complementary retrieval signals:

```text
Semantic Understanding
        +
Exact Lexical Matching
        ↓
Hybrid Retrieval
        ↓
Weighted RRF
        ↓
Improved Candidate Ranking
```

---

## 🛡️ Grounding and Hallucination Control

The chatbot is designed to keep generated answers grounded in the retrieved book content.

The generation prompt instructs the model to:

* Use only the supplied context.
* Avoid unsupported claims.
* Combine multiple passages when necessary.
* Answer in Bengali.
* Cite chapter names.
* Refuse unsupported questions.

The system uses the following refusal response:

```text
এই তথ্যটি বইয়ে পাওয়া যায়নি।
```

This provides a simple mechanism for handling questions outside the knowledge base.

---

## 🔐 Security Considerations

### API Keys

Never commit API keys to Git.

The `.env` file should remain local:

```text
.env
```

Only `.env.example` should be committed.

---

### Local Embeddings

BGE-M3 embeddings are generated locally rather than through an external embedding API.

This helps reduce exposure of book content to external embedding services and minimizes API usage.

---

### Persistent Data

The following generated files are environment-specific and should generally not be committed:

```text
.env
.venv/
chroma_db/chroma.sqlite3
logs/app.log
data/*.json
```

The repository's `.gitignore` is configured accordingly.

---

## 📝 Logging

Application logs are written to:

```text
logs/app.log
```

Logging can help with debugging:

* Retrieval behavior
* RAG pipeline execution
* API errors
* Runtime issues

---

## 🚀 Performance Considerations

The project uses several techniques to reduce unnecessary computation:

* Local embeddings
* Persistent ChromaDB storage
* Cached BM25 index construction
* Configurable Top-K retrieval
* Limited dense and lexical candidate sets
* Optional live evaluation rather than mandatory API-based testing

The first embedding/database build may take longer because BGE-M3 must process the complete knowledge base.

---

## 🧩 Design Principles

The project follows several practical RAG engineering principles:

1. **Separate data ingestion from retrieval.**
2. **Keep embeddings local where practical.**
3. **Use hybrid retrieval for complementary search signals.**
4. **Preserve source metadata throughout the pipeline.**
5. **Ground LLM responses in retrieved context.**
6. **Provide source attribution.**
7. **Handle unsupported questions explicitly.**
8. **Separate offline tests from API-dependent evaluation.**
9. **Keep secrets outside version control.**
10. **Use persistent vector storage for repeatable application runs.**

---

## 🔮 Future Improvements

Potential improvements include:

* Add a dedicated reranking stage after hybrid retrieval
* Evaluate different BM25 and RRF weight combinations
* Add retrieval metrics such as Recall@K and MRR
* Expand the evaluation dataset
* Add automated retrieval benchmarking
* Improve Bengali-specific tokenization
* Add conversation history management
* Add multilingual query support
* Introduce metadata-aware filtering
* Improve citation granularity down to individual passages
* Add Docker-based deployment
* Add CI-based automated testing and linting

---

## 📌 Limitations

The chatbot is intentionally limited to the content available in its knowledge base.

Therefore:

* It cannot reliably answer questions unrelated to the book.
* Answer quality depends on retrieval quality.
* Gemini API availability and quota affect live evaluation.
* Local BGE-M3 inference can require significant CPU/RAM resources.
* Bengali tokenization can be more challenging than English tokenization.
* Retrieved chapter sources indicate the supporting retrieval context but do not necessarily mean every listed chapter was directly used in the final generated answer.

---

## 📄 License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

* [Bengali Wikisource](https://bn.wikisource.org/) for the source text
* [BAAI](https://huggingface.co/BAAI) for the BGE-M3 embedding model
* [LangChain](https://www.langchain.com/) for the RAG framework
* [Chroma](https://www.trychroma.com/) for vector storage
* [Google Gemini](https://ai.google.dev/) for LLM-based answer generation
* [Streamlit](https://streamlit.io/) for the interactive user interface

---

## 👨‍💻 Author

**Shaiful Palash**

GitHub: [@ShaifulPalash](https://github.com/ShaifulPalash)

---

> **A production-oriented Bengali RAG implementation combining semantic retrieval, lexical retrieval, and grounded LLM generation.**

