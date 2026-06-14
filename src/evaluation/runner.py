from src.ingestion.pipeline import ingest_documents
from src.retrieval.dense import build_index, search as dense_search
from src.retrieval.sparse import build_bm25_index, search as sparse_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.generation.prompt import build_prompt
from src.generation.llm import generate
from src.evaluation.dataset import generate_test_dataset
from src.evaluation.metrics import evaluate_pipeline
from src.evaluation.logger import log_experiment


def run_evaluation(docs_folder: str = "docs", num_questions: int = 2):
    print("Ingesting documents...")
    chunks = ingest_documents(docs_folder)
    
    print("Building indexes...")
    index, chunks = build_index(chunks)
    bm25, chunks = build_bm25_index(chunks)
    
    print("Generating test dataset...")
    dataset = generate_test_dataset(chunks, num_questions)
    
    print("Running pipeline on test questions...")
    test_results = []
    for item in dataset:
        query = item["question"]
        
        dense_results = dense_search(query, index, chunks)
        sparse_results = sparse_search(query, bm25, chunks)
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        reranked = rerank(query, fused)
        prompt = build_prompt(query, reranked)
        answer = generate(prompt)
        
        test_results.append({
            "question": query,
            "answer": answer,
            "context": item["context"]
        })
    
    print("Evaluating with RAGAS...")
    metrics = evaluate_pipeline(test_results)
    
    params = {
        "chunk_threshold": 0.5,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "mistral",
        "num_questions": num_questions,
        "top_k": 5
    }
    
    log_experiment("baseline_run", params, metrics)
    
    print("Done. Results:")
    print(metrics)


if __name__ == "__main__":
    run_evaluation()