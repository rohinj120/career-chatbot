import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# LOAD MODEL
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# LOAD DATASETS
skills_df = pd.read_csv("data/skills.csv")
occupation_df = pd.read_csv("data/occupations.csv")
knowledge_df = pd.read_csv("data/knowledge.csv")
abilities_df = pd.read_csv("data/abilities.csv")

# CREATE RICH TEXT CHUNKS
# VERY IMPORTANT:
# Instead of embedding only titles,
# create rich semantic chunks.
# SKILLS CHUNKS
skills_chunks = []
for _, row in skills_df.iterrows():
    text = f"""
    Skill Name: {row.get('preferredLabel', '')}
    Description:
    {row.get('description', '')}
    Alternative Labels:
    {row.get('altLabels', '')}
    Skill Type:
    {row.get('skillType', '')}
    """
    skills_chunks.append(text)

# OCCUPATION CHUNKS
occupation_chunks = []
for _, row in occupation_df.iterrows():
    text = f"""
    Occupation: {row.get('preferredLabel', '')}
    Description:
    {row.get('description', '')}
    Alternative Labels:
    {row.get('altLabels', '')}
    Occupation Type:
    {row.get('occupationType', '')}
    """
    occupation_chunks.append(text)

# KNOWLEDGE CHUNKS
knowledge_chunks = []
for _, row in knowledge_df.iterrows():
    text = f"""
    Knowledge Area: {row.get('preferredLabel', '')}
    Description:
    {row.get('description', '')}
    Alternative Labels:
    {row.get('altLabels', '')}
    """
    knowledge_chunks.append(text)

# ABILITIES CHUNKS
abilities_chunks = []
for _, row in abilities_df.iterrows():
    text = f"""
    Ability: {row.get('preferredLabel', '')}
    Description:
    {row.get('description', '')}
    Alternative Labels:
    {row.get('altLabels', '')}
    """
    abilities_chunks.append(text)

# CREATE EMBEDDINGS
print("Creating embeddings...")
skills_embeddings = model.encode(
    skills_chunks,
    show_progress_bar=True
)
occupation_embeddings = model.encode(
    occupation_chunks,
    show_progress_bar=True
)
knowledge_embeddings = model.encode(
    knowledge_chunks,
    show_progress_bar=True
)
abilities_embeddings = model.encode(
    abilities_chunks,
    show_progress_bar=True
)

# CONVERT TO FLOAT32
skills_embeddings = np.array(
    skills_embeddings
).astype("float32")
occupation_embeddings = np.array(
    occupation_embeddings
).astype("float32")
knowledge_embeddings = np.array(
    knowledge_embeddings
).astype("float32")
abilities_embeddings = np.array(
    abilities_embeddings
).astype("float32")

# CREATE FAISS INDEXES
dimension = skills_embeddings.shape[1]

# Skills Index
skills_index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(skills_embeddings)
skills_index.add(skills_embeddings)

# Occupation Index
occupation_index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(occupation_embeddings)
occupation_index.add(occupation_embeddings)

# Knowledge Index
knowledge_index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(knowledge_embeddings)
knowledge_index.add(knowledge_embeddings)

# Abilities Index
abilities_index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(abilities_embeddings)
abilities_index.add(abilities_embeddings)

# SAVE INDEXES
faiss.write_index(
    skills_index,
    "faiss_indexes/skills.index"
)
faiss.write_index(
    occupation_index,
    "faiss_indexes/occupations.index"
)
faiss.write_index(
    knowledge_index,
    "faiss_indexes/knowledge.index"
)
faiss.write_index(
    abilities_index,
    "faiss_indexes/abilities.index"
)

# SAVE TEXT CHUNKS
with open("embeddings/skills_chunks.pkl", "wb") as f:
    pickle.dump(skills_chunks, f)
with open("embeddings/occupation_chunks.pkl", "wb") as f:
    pickle.dump(occupation_chunks, f)
with open("embeddings/knowledge_chunks.pkl", "wb") as f:
    pickle.dump(knowledge_chunks, f)
with open("embeddings/abilities_chunks.pkl", "wb") as f:
    pickle.dump(abilities_chunks, f)
print("\nEmbeddings and FAISS indexes created successfully!")