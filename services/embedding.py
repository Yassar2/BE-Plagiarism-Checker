# services/embedding.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingService:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        return self.model.encode([text])[0]

    def semantic_similarity(self, query_embedding, corpus_embeddings_subset):
        return cosine_similarity([query_embedding], corpus_embeddings_subset).flatten()
