"""
Canada occupational dataset retriever.

Uses:
  - FAISS index : vector_db/descriptors_index.faiss
  - Metadata    : embeddings/descriptors_metadata.pkl
  - Model       : BAAI/bge-small-en-v1.5  (same as ESCO / O*NET retrievers)

The retriever is intentionally flexible about metadata field names so it
works whether the Canada pickle stores dicts with 'title', 'noc_code',
'description', 'skills', etc. — or a flat 'text' string.
"""

import logging
import math
import os
import pickle
import sys
import warnings
from pathlib import Path

import faiss
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)

# ── Paths (relative to project root) ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_INDEX_PATH = PROJECT_ROOT / "vector_db" / "descriptors_index.faiss"
_META_PATH = PROJECT_ROOT / "embeddings" / "descriptors_metadata.pkl"

# ── Lazy-load so import doesn't crash when files are absent ───────────────────
_index = None
_metadata: list[dict] = []
_load_attempted = False


def _load() -> bool:
    """Load FAISS index and metadata once. Returns True on success."""
    global _index, _metadata, _load_attempted
    if _load_attempted:
        return _index is not None
    _load_attempted = True

    if not _INDEX_PATH.exists():
        logger.warning("Canada FAISS index not found: %s", _INDEX_PATH)
        return False
    if not _META_PATH.exists():
        logger.warning("Canada metadata not found: %s", _META_PATH)
        return False

    try:
        _index = faiss.read_index(str(_INDEX_PATH))
        logger.info("Canada FAISS index loaded (%d vectors, dim=%d)", _index.ntotal, _index.d)
    except Exception as exc:
        logger.error("Failed to load Canada FAISS index: %s", exc)
        return False

    try:
        with _META_PATH.open("rb") as f:
            raw = pickle.load(f)
        _metadata = [_clean_entry(e) if isinstance(e, dict) else {"text": _clean(e)} for e in raw]
        logger.info("Canada metadata loaded (%d entries)", len(_metadata))
    except Exception as exc:
        logger.error("Failed to load Canada metadata: %s", exc)
        _index = None
        return False

    return True


# ── Cleaning helpers (same pattern as ESCO / ONET retrievers) ─────────────────
_NAN_STRINGS = {"nan", "none", "null", "n/a", "na", "not available", ""}


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in _NAN_STRINGS else s


def _clean_entry(entry: dict) -> dict:
    cleaned: dict = {}
    for k, v in entry.items():
        if isinstance(v, list):
            cleaned[k] = [_clean(x) for x in v if _clean(x)]
        else:
            cleaned[k] = _clean(v)
    return cleaned


# ── Field extraction ───────────────────────────────────────────────────────────
# Canada NOC data may use various field names — list in priority order.
_TITLE_FIELDS = ["title", "occupation_title", "noc_title", "job_title", "name"]
_RICH_FIELDS = [
    "description",
    "category",
    "subcategory",
    "similarity_group",
    "main_duties",
    "employment_requirements",
    "skills",
    "abilities",
    "knowledge",
    "tasks",
    "work_activities",
    "example_titles",
    "noc_code",
]


def _extract_title(entry: dict) -> str:
    for field in _TITLE_FIELDS:
        val = entry.get(field, "")
        if val:
            return val
    # Fall back: if there is a 'text' field that starts with a known prefix
    text = entry.get("text", "")
    if text.lower().startswith("occupation:"):
        return text[len("occupation:"):].strip()
    return text[:80] if text else ""


def _build_rich_content(entry: dict) -> str:
    parts: list[str] = []
    for field in _RICH_FIELDS:
        value = entry.get(field, "")
        if isinstance(value, list):
            value = ", ".join(v for v in value if v)
        if not value:
            continue
        label = field.replace("_", " ").title()
        parts.append(f"{label}:\n{value}")
    if not parts:
        return entry.get("text", "") or ""
    return "\n\n".join(parts)


# ── Public search API ──────────────────────────────────────────────────────────

def search_canada(
    query: str,
    top_k: int = 1,
    embedding: np.ndarray | None = None,
) -> list[dict]:
    """
    Search the Canada occupational FAISS index.

    Parameters
    ----------
    query     : Natural-language question from the user.
    top_k     : Number of top results to return.
    embedding : Pre-computed (1, dim) float32 embedding from the router.
                When provided the model is NOT called again.

    Returns
    -------
    List of result dicts with keys: title, content, source, score.
    Returns [] if the index is unavailable.
    """
    if not _load():
        logger.warning("Canada retriever unavailable — skipping.")
        return []

    if embedding is None:
        from core.embedder import encode
        embedding = encode(query)

    # Normalise in-place (safe: we only use the embedding for search here)
    query_vec = embedding.copy()
    faiss.normalize_L2(query_vec)

    distances, indices = _index.search(query_vec, top_k)
    logger.debug("Canada top-1 score=%.4f", float(distances[0][0]))

    seen: set[str] = set()
    results: list[dict] = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(_metadata):
            continue
        entry = _metadata[idx]
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
            "source": "CANADA",
            "score": float(distances[0][rank]),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info("Canada retrieval executed; returned %d results", len(results))
    for i, result in enumerate(results[:3], 1):
        logger.debug("  CANADA chunk %d: %s  (score=%.4f)", i, result["title"], result["score"])
    return results


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    q = input("\nAsk a Canada career question: ")
    for i, r in enumerate(search_canada(q), 1):
        print(f"\nResult {i}: {r['title']}  (score={r['score']:.4f})")
        print("-" * 60)
        print(r["content"])
