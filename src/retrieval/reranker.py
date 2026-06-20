# src/retrieval/reranker.py
# Cross-encoder re-ranking: top-20 → top-5

from FlagEmbedding import FlagReranker
from loguru import logger

# Loaded once globally
_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = FlagReranker(
            "BAAI/bge-reranker-base",
            use_fp16=True
        )

    return _reranker


def rerank(query: str, docs: list, top_n: int = 5) -> list:
    """
    Re-rank docs using cross-encoder.
    Returns top_n documents with scores.
    """

    if not docs:
        return []

    reranker = get_reranker()

    pairs = [
        [query, d["text"]]
        for d in docs
    ]

    scores = reranker.compute_score(pairs)

    if not isinstance(scores, list):
        scores = [scores]

    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(
        docs,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    logger.debug(
        f"Reranked {len(docs)} docs → top {top_n}"
    )

    return ranked[:top_n]

if __name__ == "__main__":
    docs = [
        {"text": "Apple invested heavily in AI infrastructure."},
        {"text": "JPMorgan reported quarterly earnings."},
        {"text": "Apple expanded machine learning research."}
    ]

    results = rerank(
        "What are Apple's AI investments?",
        docs
    )

    for r in results:
        print(r["rerank_score"])
        print(r["text"])
        print("-" * 50)