import logging
import os
import sys
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
logger = logging.getLogger(__name__)
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
QDRANT_PATH = r"c:\Users\rohin jain\Desktop\canada cleaning\Canada\qdrant_local_db"
COLLECTION_NAME = "canada_jobs"
_client = QdrantClient(path=QDRANT_PATH)
_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
def search_noc(
    query: str,
    top_k: int=1,
    embedding=None,
) -> list[dict]:
    query_vector = _model.encode(query, normalize_embeddings=True).tolist()
    hits = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    seen, results = set(), []
    for hit in hits:
        payload = hit.payload or {}
        title = str(payload.get("job_title") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        parts=[]
        for field in ("original_title", "province", "city", "employment_type", "salary_detail", "education", "experience"):
            value = payload.get(field, "")
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value if v)
            if value:
                parts.append(f"{field.replace('_', ' ').title()}:\n{value}")
        content = "\n\n".join(parts) if parts else str(payload)
        results.append({
            "title": title,
            "content": content,
            "source": "NOC",
            "score": hit.score,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    logger.debug("NOC returned %d results", len(results))
    return results