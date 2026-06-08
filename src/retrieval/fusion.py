def reciprocal_rank_fusion(dense_results: list[dict], sparse_results: list[dict], top_k: int = 5) -> list[dict]:
    scores = {}
    
    for rank, chunk in enumerate(dense_results):
        scores[chunk["chunk"]] = scores.get(chunk["chunk"], 0) + 1/(rank + 60)
    
    for rank, chunk in enumerate(sparse_results):
        scores[chunk["chunk"]] = scores.get(chunk["chunk"], 0) + 1/(rank + 60)
    
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    all_chunks = dense_results + sparse_results
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk["chunk"] not in seen:
            seen.add(chunk["chunk"])
            unique_chunks.append(chunk)
    
    result = []
    for text, score in sorted_chunks[:top_k]:
        for chunk in unique_chunks:
            if chunk["chunk"] == text:
                result.append(chunk)
                break
    
    return result