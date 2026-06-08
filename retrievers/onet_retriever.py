import logging
import math
import os
import pickle
import sys
import warnings
import faiss
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"]="false"
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logger = logging.getLogger(__name__)

#Load FAISS Index
index = faiss.read_index("indexes/onet/faiss_index.bin")
logger.info("O*NET FAISS index loaded successfully")

#Load Metadata
with open("metadata/onet/metadata.pkl", "rb") as f:
    _raw_metadata = pickle.load(f)

#Clean Helpers
_NAN_STRINGS = {
     "nan",
    "none",
    "null",
    "n/a",
    "na",
    "not available",
    "",
}
def _clean(value) -> str:
    if value is None:
        return""
    if isinstance(value, float) and math.isnan(value):
        return""
    s = str(value).strip()
    return "" if s.lower() in _NAN_STRINGS else s

def _clean_entry(entry: dict) -> dict:
    cleaned = {}
    for k,v in entry.items():
        if isinstance(v, list):
            cleaned[k] = [_clean(x) for x in v if _clean(x)]
        else:
            cleaned[k] = _clean(v)

    return cleaned
metadata = [
    _clean_entry(e) if isinstance(e, dict) else {"text": _clean(e)}
    for e in _raw_metadata
]

#Important Fields
_RICH_FIELDS = [
    "description",
    "skills",
    "abilities",
    "knowledge",
    "tasks",
    "work_activities",
    "work_styles",
    "related_occupations",
]

def _build_rich_content(entry: dict) -> str:
    parts = []
    # Append education data from lookup table when available
    title = entry.get("title", "")
    if title:
        try:
            import sys, os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from llm.education_data import lookup_education, format_education_response
            edu = lookup_education(title)
            if edu:
                parts.append(format_education_response(title, edu))
        except Exception:
            pass
    for field in _RICH_FIELDS:
        value = entry.get(field, "")
        if isinstance(value, list):
            value = ", ".join(v for v in value if v)
        if value:
            parts.append(f"{field.replace('_', ' ').title()}: {value}")
    if parts:
        return "\n\n".join(parts)
    return entry.get("text", "") or "No details available."
    
#Main Search Function
def search_onet(
        query: str,
    top_k: int = 1,
    embedding: np.ndarray | None = None,
) -> list[dict]:
    """
    Search the O*NET FAISS index for a query.
    """

    #Generate query embedding
    if embedding is None:
        from core.embedder import encode
        embedding = encode(query)
    # Normalize query embedding
    faiss.normalize_L2(embedding)

    #Search FAISS index
    distances, indices = index.search(embedding, top_k)
    logger.debug(
        "ONET top-1 distance = %.4f",
        float(distances[0][0])
    )
    seen = set()
    results = []

    #Process results
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        entry = metadata[idx]
        title = entry.get("title", "")
        if not title:
            continue
        if title in seen:
            continue
        seen.add(title)
        content = _build_rich_content(entry)
        if not content or content == "No details available.":
            continue
        similarity = float(distances[0][rank])
        results.append({
            "title": title,
            "content": content,
            "source": "ONET",
            "score": similarity,
        })

    #Sort results
    results = sorted(
        results,
        key = lambda x: x["score"],
        reverse = True
    )
    logger.debug(
        "O*NET returned %d results",
        len(results)
    )
    #Debug top results
    for i, r in enumerate(results[:3], 1):
        logger.debug(
            "O*NET results %d: %s (score = %.4f)",
            i,
            r["title"],
            r["score"]
        )
    return results
#Direct title lookup
def get_onet_by_title(titles: list[str]) -> list[dict]:
    by_title = {}
    normalized_titles = {
        t.lower() for t in titles
    }
    for entry in metadata:
        title = _clean(entry.get("title", ""))
        if not title:
            continue
        if title.lower() not in normalized_titles:
            continue
        content = _build_rich_content(entry)
        if not content or content == "No details available":
            continue
        by_title[title.lower()] = {
            "title": title,
            "content": content,
            "source": "ONET",
            "score": 1.0,
            "direct_match": True,
        }
    return [
        by_title[title.lower()]
        for title in titles if title.lower() in by_title
    ]

def get_related_occupations_by_title(title: str) -> tuple[list[str], str, float]:
    """
    Look up related occupations for a given O*NET occupation title.

    Returns (related_list, matched_title, confidence) where confidence is in [0, 1].
    Logs extraction details to aid debugging of mis-matches.
    """
    title_lower = title.lower().strip()

    # Score each metadata entry by token overlap with the requested title
    title_tokens = set(re.findall(r"[a-z0-9]+", title_lower))
    best_entry = None
    best_score = 0.0
    best_title = ""

    for entry in metadata:
        entry_title = _clean(entry.get("title", ""))
        if not entry_title:
            continue
        et_lower = entry_title.lower()
        et_tokens = set(re.findall(r"[a-z0-9]+", et_lower))
        if not et_tokens:
            continue
        overlap = len(title_tokens & et_tokens) / max(len(title_tokens), len(et_tokens))
        if overlap > best_score:
            best_score = overlap
            best_entry = entry
            best_title = entry_title

    logger.info(
        "[RELATED_OCC] extracted=%r matched=%r score=%.3f",
        title, best_title, best_score,
    )

    if best_entry is None or best_score < 0.3:
        return [], best_title, best_score

    raw = _clean(best_entry.get("related_occupations", ""))
    related = [r.strip() for r in re.split(r";|,\s*(?=[A-Z])", raw) if r.strip()] if raw else []
    return related, best_title, best_score


import re


#Test mode
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(message)s"
    )
    q = input("\nAsk a career question: ")
    results = search_onet(q)
    for i, r in enumerate(results, 1):
        print(f"\nResult {i}: {r['title']}")
        print(f"Similarity Score: {r['score']:.4f}")
        print("-" * 60)
        print(r["content"])
