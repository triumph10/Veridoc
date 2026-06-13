# Architectural Decisions

Every major choice made in this project.

## [Updated as the project is built]

Starting with the loader.py which is the loader of the document from the local device.

After looking at the output of the loader.py we are getting some amount of reducdancy which needs to be fixed so we create cleaner.py which normalizes the white spaces and newlines("\n").

Text is completely clean now, but before passing it to the LLM as a prompt we cannot pass it in multiple amounts so, here we need to do chunking. Instead of fixed-sized chunking we observe in various project i will be doing semantic chunking. As fixed sized leaves out on the relevent contexts most often. 

Done with the ingestion phase, started with Retrieval.

Working on dense.py, here i chose IndexFlat2 over the other FAISS index types because it the exact search and no approximation. For a small or medium dataset it is a better choice. There were other indexes like IndexIVFFlat which trade accuracy for speed. 

## Chunking: semantic vs fixed-size

Chose: semantic chunking using cosine similarity
Rejected: fixed-size (500 chars), recursive character splitting

Why: Fixed-size splits at character count regardless of meaning - breaks 
sentences mid-thought. Semantic chunking splits only when cosine similarity 
between consecutive sentences drops below 0.5, meaning the topic has shifted.

Tradeoff: slower at ingest, requires sentence-transformers model to run, 
naive period splitting creates small fragments like "Prof." - acceptable 
for now.

## Retrieval: hybrid RRF vs dense-only

Chose: hybrid retrieval — FAISS dense + BM25 sparse combined with RRF
Rejected: dense-only

Why: Dense retrieval finds semantically similar chunks but fails on exact 
keywords. BM25 finds keyword matches but misses paraphrased meaning. RRF 
combines both by scoring chunks based on rank position in each list — 
chunks appearing high in both lists score highest. Constant 60 prevents 
top results from dominating.

## Reranker: cross-encoder vs no reranker

Chose: ms-marco-MiniLM-L-6-v2 cross-encoder
Rejected: no reranker, Cohere rerank API

Why: Embedding models encode query and chunk separately then compare vectors.
Cross-encoder reads query and chunk together in one pass — much more accurate
relevance scoring. MiniLM is small enough to run on CPU under 200ms for 5 
chunks. Cohere API rejected — adds cost and external dependency. Without 
reranker faithfulness scores drop significantly based on ablation results.

## Generation: local Ollama vs OpenAI API

Chose: Ollama with Mistral-7B running locally
Rejected: OpenAI GPT-4 API

Why: Local LLM means zero API cost, unlimited ablation runs, no data privacy 
concerns, and reproducible results with fixed seed. Quality tradeoff is real 
— Mistral-7B is weaker than GPT-4 — but acceptable for a portfolio project 
where the RAG architecture is being evaluated, not the LLM itself.

## Prompt design: context-only with source attribution

Why: Jinja2 template explicitly instructs the LLM to answer only from provided 
context. Each chunk labeled with source filename and page number makes answers 
traceable and verifiable. This is the primary hallucination prevention mechanism.

prompt.py takes your query and the retrieved chunks and builds a structured prompt using a Jinja2 template. The template tells the LLM to answer only from the provided context — this is what prevents hallucination. Each chunk is labeled with its source and page number so the answer is traceable.
llm.py sends that prompt to Ollama running locally on your machine. Ollama hosts Mistral-7B and exposes it as a simple HTTP API on port 11434. We send the prompt, get the answer back as a string.
qa.j2 is the actual prompt template. The {{ context }} and {{ query }} are Jinja2 placeholders that get filled in at runtime.


## End to end pipeline design

The pipeline connects every module in a single function: ingest → dense 
retrieve → sparse retrieve → RRF fusion → rerank → generate.

Ingestion happens at query time for now — not ideal for production where 
you would pre-build the index once and save it to disk. Acceptable for 
portfolio stage. Future improvement: persist FAISS index and BM25 index 
to disk so ingest only runs once.

The pipeline returns both the answer and the source chunks so the caller 
always knows where the answer came from. Traceability is a core design 
principle — every answer is grounded and attributable.