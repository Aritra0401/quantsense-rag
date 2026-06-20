# src/retrieval/hybrid_merger.py
# Reciprocal Rank Fusion of BM25 + Dense results
import pickle, numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from .hyde import hyde_embed
def load_indexes():
    '''Load BM25 index and ChromaDB collection.'''
    with open('data/processed/bm25_index.pkl', 'rb') as f:
        bm25_data = pickle.load(f)
    client = chromadb.PersistentClient(path='.chroma')
    col = client.get_collection('quantsense')
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return bm25_data, col, model
def rrf_fusion(ranked_lists: list, k: int = 60) -> dict:
    '''Reciprocal Rank Fusion across multiple ranked lists.'''
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
def hybrid_search(
    query: str,
    bm25_data: dict,
    col,
    model: SentenceTransformer,
    top_k: int = 20,
    use_hyde: bool = True,
):
    bm25 = bm25_data["bm25"]
    doc_ids = bm25_data["doc_ids"]

    # ---------------- BM25 Retrieval ----------------
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)

    bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_top_ids = [doc_ids[i] for i in bm25_top_idx]

    # ---------------- Dense Retrieval ----------------
    hyp_text = None

    if use_hyde:
        q_emb, hyp_text = hyde_embed(query, model)
    else:
        q_emb = model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

    dense_res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    dense_ids = dense_res["ids"][0]
    dense_docs = dense_res["documents"][0]
    dense_metas = dense_res["metadatas"][0]

    # ---------------- RRF Fusion ----------------
    fused = rrf_fusion([bm25_top_ids, dense_ids])

    top_fused_ids = list(fused.keys())[:top_k]

    # Build document objects
    id_to_doc = {
        did: {
            "text": dt,
            "metadata": dm,
            "rrf_score": fused.get(did, 0)
        }
        for did, dt, dm in zip(
            dense_ids,
            dense_docs,
            dense_metas
        )
    }

    results = [
        id_to_doc[did]
        for did in top_fused_ids
        if did in id_to_doc
    ]

    # ---------------- Company-Aware Boosting ----------------
    query_lower = query.lower()

    if "apple" in query_lower or "aapl" in query_lower:
        results.sort(
            key=lambda d: (
                d["metadata"].get("ticker", "").lower() == "aapl",
                d["rrf_score"]
            ),
            reverse=True
        )

    elif "jpm" in query_lower or "jpmorgan" in query_lower:
        results.sort(
            key=lambda d: (
                d["metadata"].get("ticker", "").lower() == "jpm",
                d["rrf_score"]
            ),
            reverse=True
        )

    elif "infosys" in query_lower or "infy" in query_lower:
        results.sort(
            key=lambda d: (
                d["metadata"].get("ticker", "").lower() == "infy",
                d["rrf_score"]
            ),
            reverse=True
        )

    elif "reliance" in query_lower:
        results.sort(
            key=lambda d: (
                "reliance" in d["metadata"].get("ticker", "").lower(),
                d["rrf_score"]
            ),
            reverse=True
        )

    return results, hyp_text