from sentence_transformers import SentenceTransformer
import numpy as np

# LOAD MODEL
model = SentenceTransformer("all-MiniLM-L6-v2")
# =========================================================
# ENCODE FUNCTION FOR QUERY ROUTING
# =========================================================

def encode(texts):

    if isinstance(texts, str):
        texts = [texts]

    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    return embeddings