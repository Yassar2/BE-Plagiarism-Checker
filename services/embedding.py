# services/embedding.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingService:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = None
        self.model_name = model_name

    def load_model(self):
        """Lazy load untuk menghindari long startup time di Railway."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def encode(self, text):
        model = self.load_model()
        return model.encode([text])[0]

    def semantic_similarity(self, query_embedding, corpus_embeddings_subset):
        return cosine_similarity([query_embedding], corpus_embeddings_subset).flatten()
