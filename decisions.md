# Architectural Decisions

Every major choice made in veridoc: what was picked, what was rejected, and why.

---

## Ingestion: semantic chunking vs fixed-size

Chose: semantic chunking using cosine similarity between sentence embeddings
Rejected: fixed-size (500 chars), recursive character splitting

Why: Fixed-size splits at character count regardless of meaning, breaks 
sentences mid-thought and destroys context. Semantic chunking splits only 
when cosine similarity between consecutive sentences drops below 0.5, 
indicating a topic boundary. Tradeoff: slower at ingest, requires a model 
to run, naive period splitting creates small fragments like "Prof." 
acceptable for now.

---

## Retrieval: hybrid RRF vs dense-only

Chose: FAISS dense + BM25 sparse combined with Reciprocal Rank Fusion
Rejected: dense-only FAISS

Why: Dense retrieval finds semantically similar chunks but fails on exact 
keywords. BM25 finds keyword matches but misses paraphrased meaning. RRF 
combines both by scoring chunks based on rank position in each list. 
Constant 60 prevents top results from dominating. Hybrid consistently 
outperforms either retriever alone.

---

## FAISS index: IndexFlatL2 vs approximate indexes

Chose: IndexFlatL2 - exact search
Rejected: IndexIVFFlat - approximate search

Why: IndexFlatL2 does exact nearest neighbour search with no approximation. 
For small to medium datasets this is the right choice, accuracy over speed. 
IndexIVFFlat trades accuracy for speed and only makes sense at millions of 
vectors.

---

## Reranker: cross-encoder vs no reranker

Chose: cross-encoder/ms-marco-TinyBERT-L-2-v2
Rejected: no reranker, Cohere rerank API

Why: Embedding models encode query and chunk separately then compare vectors. 
Cross-encoder reads query and chunk together in one pass, significantly more 
accurate relevance scoring. Cohere API rejected, adds cost and external 
dependency. Local cross-encoder runs on CPU under 200ms for 5 candidates.

---

## Generation: local Ollama + Groq fallback vs OpenAI API

Chose: Ollama with Mistral-7B locally, Groq API for production
Rejected: OpenAI GPT-4 API

Why: Local LLM means zero cost, unlimited ablation runs, no data privacy 
concerns, reproducible results. Groq added as a fast cloud fallback in same 
code, one environment variable switches between them. This is a real 
engineering pattern: local for dev, cloud for prod. OpenAI rejected, cost 
and external dependency for a system where the RAG architecture is being 
evaluated, not the LLM quality.

---

## Prompt design: context-only with source attribution

Why: Jinja2 template explicitly instructs the LLM to answer only from 
provided context. Each chunk labeled with source filename and page number 
makes answers traceable. This is the primary hallucination prevention 
mechanism, the LLM cannot invent information not in the retrieved chunks.

---

## Pipeline design: end to end single function

Why: Single function connects ingest → dense retrieve → sparse retrieve → 
RRF fusion → rerank → generate. Returns both answer and source chunks so 
the caller always knows where the answer came from. Traceability is a core 
design principle. Known limitation: ingestion happens at query time future 
improvement is to persist FAISS and BM25 indexes to disk.

---

## Evaluation: RAGAS + MLflow

Why: RAGAS measures the right things, faithfulness catches hallucination, 
context recall tells you if the retriever is failing, answer relevance tells 
you if the LLM is going off topic, context precision tells you if retrieved 
chunks are noisy. MLflow logs every run with parameters so configurations 
can be compared systematically. This turns the project from a demo into an 
engineering artifact with measurable, reproducible results.