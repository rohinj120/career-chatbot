"""
Build a lightweight router index from both ESCO and ONET titles/descriptions.
This compact index is used by the semantic router to pick which full index to search.
"""
import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_SAVE = "indexes/router/faiss_index.bin"
METADATA_SAVE = "metadata/router/metadata.pkl"
os.makedirs("indexes/router", exist_ok=True)
os.makedirs("metadata/router", exist_ok=True)

_NAN = {"nan", "none", "null", "n/a", "na", ""}

def _safe(value) -> str:
    s = str(value).strip()
    return "" if s.lower() in _NAN else s

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

documents = []

# ESCO — load from existing metadata
print("Loading ESCO entries...")
with open("embeddings/esco/metadata.pkl", "rb") as f:
    esco_raw = pickle.load(f)
for entry in esco_raw:
    if isinstance(entry, dict):
        text = entry.get("text", "")
    else:
        text = str(entry)
    if _safe(text):
        documents.append({"text": _safe(text), "source": "ESCO"})
print(f"  {len([d for d in documents if d['source'] == 'ESCO'])} ESCO entries.")

# ONET — load titles + descriptions from CSV
print("Loading O*NET entries...")
df = pd.read_csv("data/onet/onet_semantic_jobs.csv").fillna("")
onet_start = len(documents)
for _, row in df.iterrows():
    title = _safe(row.get("title", ""))
    desc = _safe(row.get("description", ""))
    text = f"{title}. {desc}".strip(". ") if desc else title
    if text:
        documents.append({"text": text, "source": "ONET"})
print(f"  {len(documents) - onet_start} O*NET entries.")
print(f"Total router documents: {len(documents)}")

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
print("Router embedding pipeline complete.")
print(f"  Index    → {INDEX_SAVE}")
print(f"  Metadata → {METADATA_SAVE}")
