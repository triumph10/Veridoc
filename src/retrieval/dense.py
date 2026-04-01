from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_index(chunks: list[dict]) -> tuple[faiss.IndexFlatL2, list[dict]]:
    """Embed chunks and store in FAISS index."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [chunk["chunk"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, chunks