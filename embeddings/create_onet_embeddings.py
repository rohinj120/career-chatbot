import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# PATHS
DATA_PATH = "data/onet"
SAVE_PATH = "embeddings/onet"
os.makedirs(SAVE_PATH, exist_ok=True)

# LOAD MODEL
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

# LOAD DATA
print("Loading O*NET semantic jobs data...")
df = pd.read_csv(f"{DATA_PATH}/onet_semantic_jobs.csv")
# Replace all NaN/None cells with empty string before any str() conversion
df = df.fillna("")
print(f"Loaded {len(df)} occupations.")


def _safe(value) -> str:
    """Convert a cell value to a clean string, never 'nan'."""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none", "null", "n/a", "na") else s


# PREPARE DOCUMENTS
print("Preparing documents...")
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
print(f"Total documents prepared: {len(documents)}")

# GENERATE EMBEDDINGS
print("Generating embeddings...")
texts = [doc["text"] for doc in documents]
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=64,
)
embeddings = embeddings.astype(np.float32)
print("Embeddings generated successfully!")

# CREATE FAISS INDEX
print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors (dim={dimension}).")

# SAVE FILES
print("Saving FAISS index...")
faiss.write_index(index, f"{SAVE_PATH}/faiss_index.bin")
print("Saving metadata...")
with open(f"{SAVE_PATH}/metadata.pkl", "wb") as f:
    pickle.dump(documents, f)
print("\nO*NET embedding pipeline completed successfully!")
print(f"Total embeddings stored: {len(documents)}")
