from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset


def evaluate_pipeline(test_results: list[dict]) -> dict:
    """Run RAGAS evaluation on pipeline results."""
    data = {
        "question": [r["question"] for r in test_results],
        "answer": [r["answer"] for r in test_results],
        "contexts": [[r["context"]] for r in test_results],
        "ground_truth": [r["context"] for r in test_results]
    }
    
    dataset = Dataset.from_dict(data)
    
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ]
    )
    
    return results