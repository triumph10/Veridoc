from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset


def evaluate_pipeline(test_results: list[dict]) -> dict:
    data = {
        "question": [r["question"] for r in test_results],
        "answer": [r["answer"] for r in test_results],
        "contexts": [[r["context"]] for r in test_results],
        "ground_truth": [r["context"] for r in test_results]
    }
    
    dataset = Dataset.from_dict(data)
    
    ollama_llm = Ollama(model="mistral")
    wrapped_llm = LangchainLLMWrapper(ollama_llm)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings
    )
    
    return results