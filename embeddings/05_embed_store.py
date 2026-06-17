"""
05_embed_store.py
-----------------
PURPOSE:
  This is the final step in the pipeline. It takes all the flat JSONL files
  created by scripts 01-04, embeds each record into a vector using a
  sentence-transformer model, and stores them in a local Qdrant vector database.

HOW EMBEDDINGS WORK:
  A sentence like "Occupation: Software Developer | Skill: Programming | Score: 4.5"
  gets converted into a list of 1024 numbers (a "vector") that captures its meaning.
  Two semantically similar sentences will have vectors that are close together
  in this 1024-dimensional space. This lets us do semantic search:
  "find jobs similar to data scientist" without exact keyword matching.

THE MODEL: BAAI/bge-large-en-v1.5
  - Open source, from Beijing Academy of AI, hosted on HuggingFace
  - One of the best English embedding models available
  - Output: 1024-dimensional vectors
  - Requires a special prefix for retrieval use: "Represent this sentence for searching..."

THE DATABASE: Qdrant (local)
  - Vector database stored as files on disk (no server needed)
  - One "collection" per domain (like a table in SQL)
  - Each "point" = one embedded record (vector + original data as payload)
  - Search uses cosine similarity: vectors pointing in the same direction = similar meaning

RESUME LOGIC:
  Embedding 500k+ records takes hours. If the script stops, it checks how many
  vectors already exist in each collection and resumes from where it left off.

Collections created:
  onet_occupations, onet_skills, onet_abilities, onet_knowledge,
  onet_work_activities, onet_work_styles, onet_work_values, onet_interests,
  onet_tasks, onet_technology_skills, onet_tools_used, onet_alternate_titles,
  onet_education,
  esco_occupations, esco_occupation_skills, esco_skills,
  canada_job_postings,
  bls_series
"""

import json
import os
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Use all available CPU cores for faster matrix operations
torch.set_num_threads(os.cpu_count() or 4)


# ── CONFIG ────────────────────────────────────────────────────────────────────

# Folder containing all the flat JSONL files (output from scripts 01-04)
DATA_DIR = Path(__file__).parent / "data"

# Folder where Qdrant saves its vector database files
QDRANT_PATH = str(Path(__file__).parent / "qdrant_storage")

# The HuggingFace model name to download and use for embeddings
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# How many records to embed at once. Larger = faster but uses more RAM.
# 32 is a safe choice for CPU-only machines.
BATCH_SIZE = 32

# The number of dimensions in each vector (fixed by the model)
VECTOR_SIZE = 1024

# Truncate text to 256 tokens before embedding.
# BGE-large supports up to 512 tokens but longer = much slower on CPU.
MAX_SEQ_LEN = 256

# BGE models are trained to expect this prefix on documents being indexed.
# Without it, retrieval quality drops. This is a quirk of BGE specifically.
BGE_PREFIX = "Represent this sentence for searching relevant passages: "


# ── COLLECTION DEFINITIONS ────────────────────────────────────────────────────
# This list defines all 17 collections we want to create.
# Each entry is a tuple of:
#   (path_to_jsonl_file, collection_name, text_function)
#
# The text_function takes one record (dict) and returns the string to embed.
# _join() builds the string by combining multiple fields with " | " separators.
# Fields that are empty/null/NaN are automatically skipped.

def _join(*parts, sep=" | "):
    """
    Combine multiple field values into one string, skipping empty/null ones.

    Example:
        _join("Occupation: Software Developer", "Skill: Python", None, "Score: 4.5")
        → "Occupation: Software Developer | Skill: Python | Score: 4.5"
    """
    return sep.join(str(p) for p in parts if p and str(p).strip() not in ("", "nan", "None"))


COLLECTIONS = [
    # ── O*NET COLLECTIONS ──────────────────────────────────────────────────

    # onet_occupations: master list of US occupations
    # Text: "Occupation: Software Developers | Develop, create, and modify..."
    (
        DATA_DIR / "onet/occupations.jsonl",
        "onet_occupations",
        lambda r: _join(
            f"Occupation: {r.get('title')}",
            r.get("description"),
        ),
    ),

    # onet_skills: skill importance/level scores per occupation
    # Text: "Occupation: Software Developers | Skill: Programming | Level | 4.5"
    (
        DATA_DIR / "onet/skills.jsonl",
        "onet_skills",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Skill: {r.get('element_name')}",
            r.get("element_description"),
            f"Scale: {r.get('scale_name')}",
            f"Score: {r.get('data_value')}",
        ),
    ),

    # onet_abilities: cognitive and physical ability scores per occupation
    (
        DATA_DIR / "onet/abilities.jsonl",
        "onet_abilities",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Ability: {r.get('element_name')}",
            r.get("element_description"),
            f"Scale: {r.get('scale_name')}",
            f"Score: {r.get('data_value')}",
        ),
    ),

    # onet_knowledge: subject knowledge scores per occupation
    (
        DATA_DIR / "onet/knowledge.jsonl",
        "onet_knowledge",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Knowledge: {r.get('element_name')}",
            r.get("element_description"),
            f"Scale: {r.get('scale_name')}",
            f"Score: {r.get('data_value')}",
        ),
    ),

    # onet_work_activities: what workers actually do on the job
    (
        DATA_DIR / "onet/work_activities.jsonl",
        "onet_work_activities",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Work Activity: {r.get('element_name')}",
            r.get("element_description"),
            f"Scale: {r.get('scale_name')}",
            f"Score: {r.get('data_value')}",
        ),
    ),

    # onet_work_styles: personality traits and work habits scores
    (
        DATA_DIR / "onet/work_styles.jsonl",
        "onet_work_styles",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Work Style: {r.get('element_name')}",
            r.get("element_description"),
            f"Score: {r.get('data_value')}",
        ),
    ),

    # onet_work_values: what workers value (achievement, independence, etc.)
    (
        DATA_DIR / "onet/work_values.jsonl",
        "onet_work_values",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Work Value: {r.get('element_name')}",
            r.get("element_description"),
            f"Score: {r.get('data_value')}",
        ),
    ),

    # onet_interests: RIASEC interest profile scores
    (
        DATA_DIR / "onet/interests.jsonl",
        "onet_interests",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Interest: {r.get('element_name')}",
            r.get("element_description"),
            f"Score: {r.get('data_value')}",
        ),
    ),

    # onet_tasks: specific day-to-day tasks performed in each occupation
    (
        DATA_DIR / "onet/tasks.jsonl",
        "onet_tasks",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Task: {r.get('task')}",
            f"Type: {r.get('task_type')}",
        ),
    ),

    # onet_technology_skills: software used in each occupation
    (
        DATA_DIR / "onet/technology_skills.jsonl",
        "onet_technology_skills",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Technology: {r.get('example')}",
            f"Category: {r.get('commodity_title')}",
            f"Class: {r.get('class_title')}",
            f"Hot Technology: {r.get('hot_technology')}",
            f"In Demand: {r.get('in_demand')}",
        ),
    ),

    # onet_tools_used: physical tools used in each occupation
    (
        DATA_DIR / "onet/tools_used.jsonl",
        "onet_tools_used",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Tool: {r.get('example')}",
            f"Category: {r.get('commodity_title')}",
            f"Class: {r.get('class_title')}",
        ),
    ),

    # onet_alternate_titles: other job titles that map to this occupation
    (
        DATA_DIR / "onet/alternate_titles.jsonl",
        "onet_alternate_titles",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Alternate Title: {r.get('alternate_title')}",
            r.get("short_title"),
        ),
    ),

    # onet_education: education/training/experience requirements
    (
        DATA_DIR / "onet/education_training_experience.jsonl",
        "onet_education",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Education/Training: {r.get('element_name')}",
            r.get("element_description"),
            f"Scale: {r.get('scale_name')}",
            f"Score: {r.get('data_value')}",
        ),
    ),

    # ── ESCO COLLECTIONS ───────────────────────────────────────────────────

    # esco_occupations: European occupations with ISCO group context
    (
        DATA_DIR / "esco/occupations.jsonl",
        "esco_occupations",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            r.get("definition"),
            r.get("description"),
            f"ISCO Group: {r.get('iscoGroupLabel')}",
            r.get("scopeNote"),
        ),
    ),

    # esco_occupation_skills: which skills each European occupation requires
    (
        DATA_DIR / "esco/occupation_skills.jsonl",
        "esco_occupation_skills",
        lambda r: _join(
            f"Occupation: {r.get('occupation_title')}",
            f"Skill: {r.get('skill_title')}",
            f"Relation: {r.get('relation_type')}",  # "essential" or "optional"
            f"Type: {r.get('skillType')}",
            r.get("skill_description"),
            r.get("skill_definition"),
        ),
    ),

    # esco_skills: standalone European skill definitions
    (
        DATA_DIR / "esco/skills.jsonl",
        "esco_skills",
        lambda r: _join(
            f"Skill: {r.get('skill_title')}",
            f"Type: {r.get('skillType')}",
            f"Group: {r.get('skillGroupLabel')}",
            r.get("description"),
            r.get("definition"),
            r.get("scopeNote"),
        ),
    ),

    # ── CANADA COLLECTION ──────────────────────────────────────────────────

    # canada_job_postings: real Canadian job postings with salary/location/education
    (
        DATA_DIR / "canada/job_postings.jsonl",
        "canada_job_postings",
        lambda r: _join(
            f"Job Title: {r.get('job_title')}",
            f"NOC 2016: {r.get('noc_2016_code_name')}",
            f"NOC 2021: {r.get('noc21_code_name')}",
            f"Province: {r.get('province_territory')}",
            f"City: {r.get('city')}",
            f"Employment Type: {r.get('employment_type')}",
            f"Employment Term: {r.get('employment_term')}",
            f"Salary: {r.get('salary_condition_detail')}",
            f"Education: {r.get('education_los')}",
            f"Experience: {r.get('experience_level')}",
        ),
    ),

    # ── BLS COLLECTION ─────────────────────────────────────────────────────

    # bls_series: US labor statistics time-series data points
    (
        DATA_DIR / "bls/series.jsonl",
        "bls_series",
        lambda r: _join(
            f"Series: {r.get('series_name')}",
            f"Date: {r.get('date')}",
            f"Period: {r.get('month')} {r.get('year')}",
            f"Value: {r.get('value')}",
        ),
    ),
]


# ── LOGGING ──────────────────────────────────────────────────────────────────
# Write progress to both the terminal and a log file so we can check it later.
_log_path = Path(__file__).parent / "embed_progress.log"
_log_file = open(_log_path, "w", encoding="utf-8", buffering=1)  # buffering=1 = line-buffered


def log(msg: str):
    """Print to terminal and write to log file simultaneously."""
    print(msg, flush=True)
    _log_file.write(msg + "\n")
    _log_file.flush()


# ── INITIALIZE MODEL AND DATABASE ────────────────────────────────────────────

log(f"Loading model: {MODEL_NAME}")
# SentenceTransformer downloads the model from HuggingFace on first run,
# then caches it locally for future runs.
model = SentenceTransformer(MODEL_NAME, truncate_dim=None)
model.max_seq_length = MAX_SEQ_LEN  # limit input length for speed
log(f"Model loaded. max_seq_length={model.max_seq_length}")

# Connect to (or create) the local Qdrant database folder
client = QdrantClient(path=QDRANT_PATH)
log(f"Qdrant connected at: {QDRANT_PATH}")
log(f"CPU threads: {torch.get_num_threads()}")


def collection_count(name: str) -> int:
    """Return how many vectors are already stored in a collection. Returns -1 if missing."""
    try:
        return client.get_collection(name).points_count or 0
    except Exception:
        return -1


def ensure_collection(name: str):
    """Create the collection in Qdrant if it doesn't already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,        # 1024 dimensions (matches the BGE model)
                distance=Distance.COSINE # cosine similarity for semantic search
            ),
        )


def embed_and_store(jsonl_path: Path, collection_name: str, text_fn):
    """
    Main function: reads a JSONL file, embeds each record, stores in Qdrant.

    Steps:
      1. Read all records from the JSONL file
      2. Check if we already embedded some (resume logic)
      3. For each batch: build text string → embed → store in Qdrant
      4. Log progress (record count, speed, ETA)
    """
    if not jsonl_path.exists():
        log(f"  SKIP {collection_name}: file not found")
        return

    log(f"\n{'='*60}")
    log(f"Processing: {collection_name}  source: {jsonl_path.name}")

    # Read all records from the JSONL file into memory
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))  # parse each line as JSON

    total = len(records)
    log(f"  Records: {total}")
    if not records:
        return

    ensure_collection(collection_name)

    # RESUME LOGIC: check how many vectors are already stored
    # If interrupted, we skip ahead instead of starting over
    start_idx = collection_count(collection_name)
    if start_idx == total:
        log(f"  DONE (already has {start_idx} vectors) — skipping")
        return
    if start_idx > 0:
        log(f"  Resuming from row {start_idx}/{total}")

    # Build the text strings for only the records we haven't embedded yet
    texts = []
    for r in records[start_idx:]:
        try:
            t = text_fn(r)   # call the lambda to build the text for this record
        except Exception:
            t = ""
        # Prepend the BGE prefix — required for this model to work well
        texts.append(BGE_PREFIX + t if t else BGE_PREFIX + "no data")

    # BATCH EMBEDDING LOOP
    # We process BATCH_SIZE records at a time to avoid running out of memory.
    # After each batch we immediately save to Qdrant (so progress is preserved).
    t0 = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]

        # embed(): converts text strings → numpy arrays of shape (batch_size, 1024)
        # normalize_embeddings=True: scales each vector to length 1 (required for cosine similarity)
        vecs = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)

        # Build Qdrant PointStruct objects — each point has:
        #   id: unique integer (the record's global position in the dataset)
        #   vector: the 1024-float embedding
        #   payload: the original record data (stored alongside for retrieval)
        points = [
            PointStruct(
                id=start_idx + i + j,
                vector=vecs[j].tolist(),
                payload={
                    # NaN values in JSON cause errors, so convert them to None
                    k: (None if (isinstance(v, float) and v != v) else v)
                    for k, v in records[start_idx + i + j].items()
                },
            )
            for j in range(len(batch_texts))
        ]

        # Save this batch to Qdrant (immediately persisted to disk)
        client.upsert(collection_name=collection_name, points=points)

        # Log progress with speed and estimated time remaining
        done = i + len(batch_texts)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = len(texts) - done
        eta = remaining / rate if rate > 0 else 0
        log(f"  [{start_idx + done}/{total}]  {rate:.1f}/s  ETA {eta/60:.1f}min")

    log(f"  Done: {total} vectors in '{collection_name}' ({time.time()-t0:.1f}s)")


# ── RUN ALL COLLECTIONS ───────────────────────────────────────────────────────
# Loop through all 17 (jsonl_path, collection_name, text_fn) tuples and process each one.

total_start = time.time()
for jsonl_path, collection_name, text_fn in COLLECTIONS:
    embed_and_store(jsonl_path, collection_name, text_fn)

log(f"\nAll done. Total time: {(time.time() - total_start) / 3600:.2f} hours")
_log_file.close()

print(f"\n{'='*60}")
print(f"All done. Total time: {(time.time() - total_start) / 60:.1f} min")
