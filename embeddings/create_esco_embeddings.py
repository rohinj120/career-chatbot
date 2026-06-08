import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

METADATA_SRC = "embeddings/esco/metadata.pkl"
INDEX_SAVE = "indexes/esco/faiss_index.bin"
METADATA_SAVE = "metadata/esco/metadata.pkl"
os.makedirs("indexes/esco", exist_ok=True)
os.makedirs("metadata/esco", exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

print("Loading ESCO metadata...")
with open(METADATA_SRC, "rb") as f:
    raw = pickle.load(f)
print(f"Loaded {len(raw)} ESCO entries.")

documents = []
for entry in raw:
    if isinstance(entry, dict):
        documents.append(entry)
    else:
        documents.append({"text": str(entry)})

texts = [doc.get("text", "") for doc in documents]

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
print("ESCO embedding pipeline complete.")
print(f"  Index    → {INDEX_SAVE}")
print(f"  Metadata → {METADATA_SAVE}")
