from rank_bm25 import BM25Okapi

def build_bm25_index(chunks: list[dict]):
    texts = [chunk["chunk"] for chunk in chunks]

    tokenized = [chunk["chunk"].split() for chunk in chunks]
    
    bm25 = BM250kapi(tokenized)
    
    return bm25, texts


    