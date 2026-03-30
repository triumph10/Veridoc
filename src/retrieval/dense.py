from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Dense retriever using sentence embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def index(self, docs: list[str]):
        """
        Encode and store document embeddings.
        """
        if not docs:
            raise ValueError("No documents provided for indexing.")

        self.documents = docs
        self.embeddings = self.model.encode(docs, show_progress_bar=True)

    def search(self, query: str, top_k: int = 5):
        """
        Retrieve top_k most similar documents for a query.
        """
        if self.embeddings is None:
            raise ValueError("No index found. Call index() first.")

        query_embedding = self.model.encode([query])

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            {
                "document": self.documents[i],
                "score": float(similarities[i])
            }
            for i in top_indices
        ]

        return results