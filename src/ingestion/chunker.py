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
    
    chunks = []
    current_chunk = [sentences[0]]  #initialize
    
    for i in range(1,len(sentences)): # compare each sentence with prev
        sim = cosine_similarity(embeddings[i-1].reshape(1,-1),
                                embeddings[i].reshape(1,-1))[0][0]
        
        if sim < threshold: # if similarity is low, start a new chunk
            chunks.append(" ".join(current_chunk)+ ".") 
            current_chunk = [sentences[i]]
            
        else: # if similarity is high, add to current chunk
            current_chunk.append(sentences[i])
        
    chunks.append(". ".join(current_chunk) + ".")
    return chunks