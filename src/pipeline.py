from src.ingestion.pipeline import ingest_documents
from src.retrieval.dense import build_index, search as dense_search
from src.retrieval.sparse import build_bm25_index, search as sparse_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.generation.prompt import build_prompt
from src.generation.llm import generate


def run_pipeline(query: str, docs_folder: str = "docs") -> dict:
    #ingest
    chunks = ingest_documents(docs_folder)
    
    #build indexes
    index, chunks = build_index(chunks)
    bm25, chunks = build_bm25_index(chunks)
    
    #retrieve
    dense_results = dense_search(query, index, chunks)
    sparse_results = sparse_search(query, bm25, chunks)
    
    #fuse
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    
    #rerank 
    reranked = rerank(query, fused)
    
    #generate
    prompt = build_prompt(query, reranked)
    answer = generate(prompt)
    
    return {
        "answer": answer,
        "sources": reranked
    }
    