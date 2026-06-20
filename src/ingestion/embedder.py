# src/ingestion/embedder.py
# Embeds all chunks into ChromaDB + builds BM25 index
import os, json, pickle
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from loguru import logger
EMBED_MODEL = 'BAAI/bge-small-en-v1.5' # Free, strong, fast
CHROMA_DIR = '.chroma'
def load_all_chunks(chunk_dir='data/processed/chunks') -> list:
    all_docs = []
    for fname in os.listdir(chunk_dir):
        if fname.endswith('.json'):
            with open(f'{chunk_dir}/{fname}') as f:
                all_docs.extend(json.load(f))
    return all_docs
def build_vector_index(docs: list):
    '''Build ChromaDB collection from chunks.'''
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Delete existing collection if rebuilding
    try:
        client.delete_collection('quantsense')
    except:
        pass
    col = client.create_collection('quantsense',
        metadata={'hnsw:space': 'cosine'})
    # Batch embed (32 at a time)
    BATCH = 32
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i+BATCH]
        texts = [d['text'] for d in batch]
        ids = [d['id'] for d in batch]
        metas = [d['metadata'] for d in batch]
        embs = model.encode(texts, normalize_embeddings=True).tolist()
        col.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
        logger.info(f'Embedded batch {i//BATCH + 1}')
    logger.info(f'ChromaDB: {col.count()} docs indexed')
    return col
def build_bm25_index(docs: list):
    '''Build BM25 index and save to disk.'''
    corpus = [d['text'].lower().split() for d in docs]
    doc_ids = [d['id'] for d in docs]
    bm25 = BM25Okapi(corpus)
    with open('data/processed/bm25_index.pkl', 'wb') as f:
        pickle.dump({'bm25': bm25, 'doc_ids': doc_ids}, f)
    logger.info('BM25 index saved.')
    return bm25, doc_ids
if __name__ == '__main__':
    docs = load_all_chunks()
    build_vector_index(docs)
    build_bm25_index(docs)
    logger.info('Indexing complete!')