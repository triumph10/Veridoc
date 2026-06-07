from rank_bm25 import BM25Okapi
import numpy as np

def build_bm25_index(chunks: list[dict]):
    
    tokenized = [chunk["chunk"].split() for chunk in chunks]
    
    bm25 = BM25Okapi(tokenized)
    
    return bm25, chunks

def search(query, bm25, chunks, top_k = 5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


