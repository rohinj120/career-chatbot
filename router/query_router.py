"""
Dynamic semantic router — fully embedding-similarity-based.

Pipeline
--------
User Query
  → encode()              (one model call, result reused by retrievers)
  → search every FAISS index for top-1 hit
  → convert L2 distance → similarity score  [0, 1]
  → rank sources by score
  → select best source; add runner-up if gap ≤ CLOSE_SCORE_THRESHOLD
  → return (selected_sources, all_scores, query_embedding)

No hardcoded intents, labels, or keyword rules anywhere.
To add a new dataset: add one entry to INDEX_REGISTRY — nothing else changes.
"""

import logging
import os
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.embedder import encode

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Index Registry ─────────────────────────────────────────────────────────────
# Maps a human-readable source name → (faiss_index_path, metadata_path).
# Paths are resolved from the project root.
INDEX_REGISTRY: dict[str, tuple[str, str]] = {
    "ESCO": (
        str(PROJECT_ROOT / "indexes" / "esco" / "faiss_index.bin"),
        str(PROJECT_ROOT / "metadata" / "esco" / "metadata.pkl"),
    ),
    "ONET": (
        str(PROJECT_ROOT / "indexes" / "onet" / "faiss_index.bin"),
        str(PROJECT_ROOT / "metadata" / "onet" / "metadata.pkl"),
    ),
    # Canada NOC occupational dataset
    "CANADA": (
        str(PROJECT_ROOT / "vector_db" / "descriptors_index.faiss"),
        str(PROJECT_ROOT / "embeddings" / "descriptors_metadata.pkl"),
    ),
}

# ── Routing thresholds (tune these without touching any other code) ────────────
# Any source whose score is within this gap of the best score is also selected.
# Raise this to include more sources on overlapping queries.
CLOSE_SCORE_THRESHOLD: float = 0.15

# Warn the user when the best match scores below this — results may be off-topic.
MIN_CONFIDENCE: float = 0.20

CANADA_QUERY_TERMS = ("canada", "canadian", "noc")
CANADA_HINT_THRESHOLD: float = 0.45

# ── Lazy-loaded FAISS indexes ──────────────────────────────────────────────────
_loaded_indexes: dict[str, faiss.Index] = {}


def _load_indexes() -> dict[str, faiss.Index]:
    """Load every registered FAISS index once; return cached dict thereafter."""
    if _loaded_indexes:
        return _loaded_indexes
    for name, (idx_path, _) in INDEX_REGISTRY.items():
        if not os.path.exists(idx_path):
            logger.warning("Index not found, skipping: %s (%s)", name, idx_path)
            continue
        _loaded_indexes[name] = faiss.read_index(idx_path)
        logger.debug("Loaded FAISS index: %s  (%d vectors)", name, _loaded_indexes[name].ntotal)
    return _loaded_indexes


def _l2_to_similarity(distance: float) -> float:
    # IndexFlatIP with normalized vectors returns cosine similarity directly [0, 1]
    return float(distance)


def _targets_canada(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in CANADA_QUERY_TERMS)


# ── Public API ─────────────────────────────────────────────────────────────────

def route_query(
    query: str,
) -> tuple[list[str], dict[str, float], np.ndarray]:
    """
    Select the most semantically relevant FAISS index(es) for *query*.

    Returns
    -------
    selected_sources : list[str]
        Ordered list of source names to retrieve from (best score first).
        Contains 1 source normally; 2 when scores are within CLOSE_SCORE_THRESHOLD.
    scores : dict[str, float]
        Similarity score in [0, 1] for every registered index.
    query_embedding : np.ndarray  shape (1, dim), dtype float32
        Pre-computed embedding — pass directly to retrievers so the model
        is called only once per user turn.
    """
    query_embedding = encode(query)
    indexes = _load_indexes()

    if not indexes:
        logger.error("No FAISS indexes could be loaded. Check INDEX_REGISTRY paths.")
        return [], {}, query_embedding

    # ── Score every index ──────────────────────────────────────────────────────
    scores: dict[str, float] = {}
    for name, idx in indexes.items():
        distances, _ = idx.search(query_embedding, 1)
        scores[name] = _l2_to_similarity(float(distances[0][0]))

    # ── Rank best → worst ──────────────────────────────────────────────────────
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = sorted_scores[0]

    # ── Debug log: similarity scoreboard ──────────────────────────────────────
    logger.info("┌─ Semantic Router ────────────────────────────────")
    logger.info("│  Query : %r", query[:80])
    logger.info("│")
    for name, sim in sorted_scores:
        bar = "█" * round(sim * 20)
        marker = " ◄ best" if name == best_name else ""
        logger.info("│  %-6s  %.4f  |%-20s|%s", name, sim, bar, marker)
    logger.info("└──────────────────────────────────────────────────")

    # ── Confidence warning ─────────────────────────────────────────────────────
    if best_score < MIN_CONFIDENCE:
        logger.warning(
            "Low-confidence retrieval — best score %.4f is below threshold %.2f. "
            "The query may be outside the knowledge base.",
            best_score,
            MIN_CONFIDENCE,
        )

    # ── Source selection ───────────────────────────────────────────────────────
    # Include every source whose score is within CLOSE_SCORE_THRESHOLD of the best.
    selected = [best_name]
    for name, score in sorted_scores[1:]:
        gap = best_score - score
        if gap <= CLOSE_SCORE_THRESHOLD:
            selected.append(name)
            logger.info(
                "Added source %s  (gap=%.4f ≤ threshold=%.2f)",
                name, gap, CLOSE_SCORE_THRESHOLD,
            )
        else:
            logger.info(
                "Excluded source %s  (gap=%.4f > threshold=%.2f)",
                name, gap, CLOSE_SCORE_THRESHOLD,
            )

    # Canada geography hint: force-add CANADA even if gap is large when query targets it.
    canada_score = scores.get("CANADA")
    if (
        canada_score is not None
        and "CANADA" not in selected
        and _targets_canada(query)
        and canada_score >= CANADA_HINT_THRESHOLD
    ):
        selected.append("CANADA")
        logger.info(
            "Added CANADA source due to query geography hint (score=%.4f >= %.2f)",
            canada_score, CANADA_HINT_THRESHOLD,
        )

    logger.info("Final selected sources: %s", selected)
    return selected, scores, query_embedding


def build_context(results: list) -> str:
    """Format a list of retriever result dicts into a readable context string."""
    if not results:
        return "No relevant information found."

    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Unknown")
        source = r.get("source", "")
        content = r.get("content", "")
        score = r.get("score")
        if isinstance(content, list):
            content = ", ".join(str(c) for c in content)
        score_line = f"Similarity: {score:.4f}\n\n" if score is not None else ""
        block = (
            f"RESULT {i}\n\n"
            f"Occupation:\n{title}\n\n"
            f"Source:\n{source}\n\n"
            f"{score_line}"
            f"Full Information:\n{content}\n\n"
            f"{'=' * 36}"
        )
        parts.append(block)
    return "\n\n".join(parts)
