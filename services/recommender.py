# services/recommender.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from services.preprocessing import jaccard_similarity

class RecommenderService:
    def __init__(
        self,
        tfidf_vectorizer,
        tfidf_matrix,
        corpus_texts,
        corpus_embeddings,
        embedding_service,
        weight_lexical=0.6,
        weight_structure=0.25,
        weight_semantic=0.15,
        top_k_tfidf=20
    ):
        self.vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.corpus_texts = corpus_texts
        self.corpus_embeddings = corpus_embeddings.astype("float32")
        self.embedding_service = embedding_service

        self.weight_lexical = weight_lexical
        self.weight_structure = weight_structure
        self.weight_semantic = weight_semantic
        self.top_k_tfidf = top_k_tfidf

    def tfidf_search(self, query):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        idx_sorted = scores.argsort()[::-1][:self.top_k_tfidf]
        return idx_sorted, scores[idx_sorted]

    def compute_semantic(self, query, candidate_indices):
        q_emb = self.embedding_service.encode(query)
        subset = self.corpus_embeddings[candidate_indices]
        return self.embedding_service.semantic_similarity(q_emb, subset)

    def compute_structure(self, query, candidate_indices):
        return np.array([
            jaccard_similarity(query, self.corpus_texts[i])
            for i in candidate_indices
        ], dtype="float32")

    def search(self, query, top_k=10):
        idx_top, lexical_scores = self.tfidf_search(query)
        semantic_scores = self.compute_semantic(query, idx_top)
        structure_scores = self.compute_structure(query, idx_top)

        final_scores = (
            lexical_scores * self.weight_lexical +
            structure_scores * self.weight_structure +
            semantic_scores * self.weight_semantic
        )

        top_indices = final_scores.argsort()[::-1][:top_k]

        return {
            "candidate_indices": idx_top,
            "lexical": lexical_scores,
            "semantic": semantic_scores,
            "structure": structure_scores,
            "final": final_scores,
            "top_indices": top_indices
        }
