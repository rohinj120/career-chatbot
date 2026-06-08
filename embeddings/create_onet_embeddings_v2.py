import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/onet/onet_semantic_jobs.csv"
INDEX_SAVE = "indexes/onet/faiss_index.bin"
METADATA_SAVE = "metadata/onet/metadata.pkl"
os.makedirs("indexes/onet", exist_ok=True)
os.makedirs("metadata/onet", exist_ok=True)

_NAN = {"nan", "none", "null", "n/a", "na", ""}

def _safe(value) -> str:
    s = str(value).strip()
    return "" if s.lower() in _NAN else s

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

print("Loading O*NET data...")
df = pd.read_csv(DATA_PATH).fillna("")
print(f"Loaded {len(df)} occupations.")

documents = []
for _, row in df.iterrows():
    documents.append({
        "text": _safe(row.get("semantic_text", "")),
        "title": _safe(row.get("title", "")),
        "description": _safe(row.get("description", "")),
        "skills": _safe(row.get("skills", "")),
        "abilities": _safe(row.get("abilities", "")),
        "knowledge": _safe(row.get("knowledge", "")),
        "tasks": _safe(row.get("tasks", "")),
        "work_activities": _safe(row.get("work_activities", "")),
        "work_styles": _safe(row.get("work_styles", "")),
        "related_occupations": _safe(row.get("related_occupations", "")),
        "type": "occupation",
    })

texts = [doc["text"] for doc in documents]

print("Generating embeddings...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=64,
    normalize_embeddings=True,
)
embeddings = embeddings.astype(np.float32)
print("Embeddings generated.")

print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors (dim={dimension}).")

print("Saving...")
faiss.write_index(index, INDEX_SAVE)
with open(METADATA_SAVE, "wb") as f:
    pickle.dump(documents, f)
print("O*NET embedding pipeline complete.")
print(f"  Index    → {INDEX_SAVE}")
print(f"  Metadata → {METADATA_SAVE}")
