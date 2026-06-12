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