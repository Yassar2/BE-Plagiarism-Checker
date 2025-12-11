import joblib, numpy as np, pandas as pd, os

BASE_DIR = "backend"
MODELS_DIR = os.path.join(BASE_DIR, "models")

tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf.pkl"))
tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"))
df = pd.read_csv(os.path.join(MODELS_DIR, "corpus_clean.csv")).fillna("")
corpus_embeddings = np.load(os.path.join(MODELS_DIR, "corpus_embeddings.npy"))

print("✅ Semua model bisa di-load")
