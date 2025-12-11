# backend/app.py
import os
import sys
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

# ====== PATH FIX ======
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# ====== IMPORT SESUAI STRUKTUR FOLDER ======
from backend.services.embedding import EmbeddingService
from backend.services.recommender import RecommenderService
from backend.services.preprocessing import jaccard_similarity
from backend.utils.helpers import format_results

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TOP_K_TFIDF = int(os.environ.get("TOP_K_TFIDF", 20))
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", 10))
FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")  # ganti ke domain Vercel pas production

app = Flask(__name__)
# Fix CORS
if FRONTEND_URL == "*":
    CORS(app)
else:
    CORS(app, resources={r"/predict": {"origins": [FRONTEND_URL]}})

# ================= GLOBAL VARIABLES =================
tfidf_vectorizer = None
tfidf_matrix = None
df = None
corpus_texts = None
corpus_embeddings = None
embedding_service = None
recommender = None

# ================== HELPER =================
def load_models():
    global tfidf_vectorizer, tfidf_matrix, df, corpus_texts, corpus_embeddings, embedding_service, recommender
    if recommender is not None:
        return recommender
    try:
        print("Loading TFIDF vectorizer...")
        tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf.pkl"))
        print("TFIDF vocabulary size:", len(tfidf_vectorizer.vocabulary_))
        
        print("Loading TFIDF matrix...")
        tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"))
        
        print("Loading corpus CSV...")
        df = pd.read_csv(os.path.join(MODELS_DIR, "corpus_clean.csv")).fillna("")
        if "abstract" not in df.columns:
            raise ValueError("Kolom 'abstract' tidak ditemukan dalam corpus_clean.csv")
        corpus_texts = df["abstract"].astype(str).tolist()
        
        print("Loading embeddings...")
        corpus_embeddings = np.load(os.path.join(MODELS_DIR, "corpus_embeddings.npy"))
        
        embedding_service = EmbeddingService()
        recommender = RecommenderService(
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            corpus_texts=corpus_texts,
            corpus_embeddings=corpus_embeddings,
            embedding_service=embedding_service,
            top_k_tfidf=TOP_K_TFIDF
        )
        print("✅ Model & services loaded successfully.")
        return recommender
    except Exception:
        print("❌ Gagal load model:")
        traceback.print_exc()
        recommender = None
        return None

# ================== PARSE INPUT ==================
def _parse_request_for_text(req):
    if req.files and "file" in req.files:
        file = req.files["file"]
        if file.filename:
            from backend.services import preprocessing
            return preprocessing.extract_text_from_file(file)
        raise ValueError("File tidak valid.")
    if req.is_json:
        data = req.get_json(silent=True) or {}
        txt = data.get("text") or data.get("query") or data.get("q")
        if txt:
            return txt
    txt = req.form.get("text") or req.form.get("query") or req.form.get("q")
    if txt:
        return txt
    raw = req.get_data(as_text=True)
    if raw.strip():
        return raw
    raise ValueError("Tidak menemukan teks atau file. Kirim JSON {'text': ...}")

# ================== ROUTES ==================
@app.before_request
def log_request_info():
    print(f"Incoming {request.method} {request.path}")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": recommender is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    global recommender
    if recommender is None:
        recommender = load_models()
    if recommender is None:
        return jsonify({"error": "Server tidak siap (model gagal dimuat)."}), 500
    try:
        top_k = request.args.get("top_k")
        if request.is_json:
            top_k = request.get_json(silent=True).get("top_k", top_k)
        top_k = int(top_k) if top_k else DEFAULT_TOP_K

        query_text = _parse_request_for_text(request)
        if not query_text.strip():
            return jsonify({"error": "Teks kosong"}), 400

        result = recommender.search(query_text, top_k=top_k)
        formatted = format_results(result, df)

        MAX_ABSTRACT_LEN = int(os.environ.get("MAX_ABSTRACT_LEN", 2000))
        for r in formatted:
            if "abstract" in r and len(r["abstract"]) > MAX_ABSTRACT_LEN:
                r["abstract"] = r["abstract"][:MAX_ABSTRACT_LEN] + "..."

        return jsonify(formatted)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        if os.environ.get("DEBUG") == "1":
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
        return jsonify({"error": "Internal server error"}), 500

# ================= LOCAL RUN ==================
if __name__ == "__main__":
    # Auto-load model saat start
    recommender = load_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
