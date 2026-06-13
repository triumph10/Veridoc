from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    passages = [chunk["chunk"] for chunk in chunks]
    scores = model.predict([(query, passage) for passage in passages])
    
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in ranked[:top_k]]
