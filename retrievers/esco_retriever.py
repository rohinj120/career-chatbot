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
index = faiss.read_index("indexes/esco/faiss_index.bin")
with open("metadata/esco/metadata.pkl", "rb") as f:
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
_RICH_FIELDS = [
    "description", "skills", "abilities", "knowledge",
    "tasks", "work_activities", "work_styles", "related_occupations",
]
def _extract_title(entry: dict) -> str:
    title = entry.get("title", "")
    if title:
        return title
    text = entry.get("text", "")
    if text.lower().startswith("occupation:"):
        return text[len("occupation:"):].strip()
    return text
def _build_rich_content(entry: dict) -> str:
    parts = []
    for field in _RICH_FIELDS:
        value = entry.get(field, "")
        if isinstance(value, list):
            value = ", ".join(v for v in value if v)
        if not value:
            continue
        parts.append(f"{field.replace('_', ' ').title()}:\n{value}")
    if not parts:
        return entry.get("text", "") or ""
    return "\n\n".join(parts)
def search_esco(
    query: str,
    top_k: int = 1,
    embedding: np.ndarray | None = None,
) -> list[dict]:
    """
    Search the ESCO FAISS index for *query*.
    Parameters
    ----------
    query     : Natural-language question from the user.
    top_k     : Number of FAISS candidates to consider.
    embedding : Pre-computed (1, dim) float32 vector from the semantic router.
                When supplied the model is NOT called again â€” pass this from
                route_query() to avoid a redundant encode() call per turn.
    """
    if embedding is None:
        from core.embedder import encode
        embedding = encode(query)
    distances, indices = index.search(embedding, top_k)
    logger.debug("ESCO  top-1 L2=%.4f", float(distances[0][0]))
    seen, results = set(), []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        entry = metadata[idx]
        title = _extract_title(entry)
        if not title or title in seen:
            continue
        seen.add(title)
        content = _build_rich_content(entry)
        if not content:
            continue
        results.append({
            "title": title,
            "content": content,
            "source": "ESCO",
            "score": float(distances[0][rank]),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    logger.debug("ESCO  returned %d results", len(results))
    for i, r in enumerate(results[:3], 1):
        logger.debug("  ESCO chunk %d: %s  (score=%.4f)", i, r["title"], r["score"])
    return results
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    q = input("\nAsk an ESCO-related question: ")
    for i, r in enumerate(search_esco(q), 1):
        print(f"\nResult {i}: {r['title']}  (L2={r['score']:.4f})")
        print("-" * 50)
        print(r["content"])