# Ablation Report

## Baseline Configuration
- chunk_threshold: 0.5
- embedding_model: all-MiniLM-L6-v2
- llm_model: mistral
- top_k: 5
- num_questions: 3

## Results

| Config | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|--------|-------------|-----------------|----------------|-------------------|
| baseline (hybrid + reranker) | 1.00 | 0.89 | 1.00 | 1.00 |