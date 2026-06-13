from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

def build_index(chunks : list[dict]) -> tuple[faiss.IndexFlatL2, list[dict]]:
    """embed chunks and store in faiss index"""
    model = SentenceTransformer('all-MiniLM-L6-v2') #creates an embedding model using pre trained model
    texts = [chunk["chunk"] for chunk in chunks] #extracts the text from the chunks
    embeddings = model.encode(texts,  show_progress_bar = True) #creates embeddings for the text
    embeddings = np.array(embeddings).astype('float32') #converts the embeddings to a numpy array of type float32
    dimension = embeddings.shape[1] #gets the dimension of the embeddings
    index = faiss.IndexFlatL2(dimension) #creates a faiss index for the embeddings
    index.add(embeddings)
    return index, chunks

def search(query : str, index: faiss.IndexFlatL2, chunks : list[dict], top_k: int = 5) -> list[dict]:
    """Search FAISS index for most similar chunks to query."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    distances, indices= index.search(query_embedding, top_k) #searches the index for the top_k most similar chunks to the query embedding
    
    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])
    return results

