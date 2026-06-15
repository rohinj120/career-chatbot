import logging
import math
import os
import pickle
import sys
import warnings
import faiss
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logger = logging.getLogger(__name__)

index = faiss.read_index("indexes/noc/faiss_index.bin")
with open("metadata/noc/metadata.pkl", "rb") as f:
    _raw_metadata = pickle.load(f)

_NAN_STRINGS = {"nan", "none", "null", "n/a", "na", "not available", ""}

def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in _NAN_STRINGS else s

def _clean_entry(entry: dict) -> dict:
    cleaned = {}
    for k, v in entry.items():
        if isinstance(v, list):
            cleaned[k] = [_clean(x) for x in v if _clean(x)]
        else:
            cleaned[k] = _clean(v)
    return cleaned

metadata = [
    _clean_entry(e) if isinstance(e, dict) else {"text": _clean(e)}
    for e in _raw_metadata
]

def _build_rich_content(entry: dict) -> str:
    parts = []
    for field in ("description", "category", "subcategory", "similarity_group"):
        value = entry.get(field, "")
        if isinstance(value, list):
            value = ", ".join(v for v in value if v)
        if not value:
            continue
        parts.append(f"{field.replace('_', ' ').title()}:\n{value}")
    if not parts:
        return entry.get("text", "") or ""
    return "\n\n".join(parts)

def search_noc(
    query: str,
    top_k: int = 1,
    embedding: np.ndarray | None = None,
) -> list[dict]:
    if embedding is None:
        from core.embedder import encode
        embedding = encode(query)
    distances, indices = index.search(embedding, top_k)
    logger.debug("NOC top-1 score=%.4f", float(distances[0][0]))
    seen, results = set(), []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        entry = metadata[idx]
        title = entry.get("title", "")
        if not title or title in seen:
            continue
        seen.add(title)
        content = _build_rich_content(entry)
        if not content:
            continue
        results.append({
            "title": title,
            "content": content,
            "source": "NOC",
            "score": float(distances[0][rank]),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    logger.debug("NOC returned %d results", len(results))
    return results
