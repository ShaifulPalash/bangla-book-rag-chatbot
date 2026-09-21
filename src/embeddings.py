"""
src/embeddings.py
===================
Wraps the local, multilingual BAAI/bge-m3 embedding model so it can be
plugged directly into LangChain's Chroma vector store (Phase 4) and
retriever (Phase 5).

Why bge-m3, and why local instead of an API:
-----------------------------------------------
- Multilingual/Bengali support: bge-m3 is trained on 100+ languages and
  ranks strongly on multilingual retrieval benchmarks, including
  low-resource languages like Bengali — unlike a default English-only
  embedding model, which would produce poor-quality vectors for Bangla
  text and directly violate the assignment's multilingual requirement.
- Running LOCALLY (via sentence-transformers, wrapped here through
  LangChain's HuggingFaceEmbeddings) means embedding the entire book —
  hundreds of chunks, plus every future query — costs zero API quota and
  can never hit a rate limit. This is the main lever that keeps this
  project safe on a free tier: the highest-volume step in the whole
  pipeline (indexing every chunk) never touches an external API.
- The ONLY external API call anywhere in this project is the LLM answer-
  generation step in rag_pipeline.py (Phase 5), which happens once per
  user question — a much lower, much safer volume for a free tier.

First-run note: the model (~2GB) downloads automatically from Hugging
Face the first time this is used, then is cached locally
(~/.cache/huggingface) for every run after that. The first run will be
slow; later runs are fast.
"""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL, logger

# Module-level cache so we only ever load the (large) model once per
# process, no matter how many times get_embedding_model() is called.



@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    logger.info(
        f"Loading local embedding model '{EMBEDDING_MODEL}' "
        f"(first run downloads ~2GB from Hugging Face and may take "
        f"a few minutes; cached after that) ..."
    )

    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model loaded successfully.")

    return model


def self_test():
    """
    A quick, standalone sanity check: embeds one Bengali sentence and
    reports the resulting vector's dimensionality. Useful to run in
    isolation to confirm the model downloaded and works, before running
    the full (slower) vector store build in vectordb.py.

    Run with: python src/embeddings.py
    """
    model = get_embedding_model()
    sample_text = "দেবদাস ও পার্বতী শৈশবের বন্ধু ছিল।"
    vector = model.embed_query(sample_text)
    logger.info(
        f"Self-test embedding produced a {len(vector)}-dimensional vector "
        f"for the sample sentence. (bge-m3 should report 1024.)"
    )
    return vector


if __name__ == "__main__":
    self_test()
