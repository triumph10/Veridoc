from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

"""
split the cleaned texxt in sentences
embed the sentences using sentence transformer
calc consine similarity btwn sentences
similarty above a threshold are grouped together

"""

model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text: str,threshold: float=0.5) -> list[str]:
    """Splits the input text into chunks based on cosine similarity of sentence embeddings."""
    
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    embeddings = model.encode(sentences)