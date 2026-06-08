import logging
import os
import re
import sys
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

LOG_LEVEL = os.getenv("CAREER_RAG_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)-8s %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

from llm.generate_response import generate_response, build_related_occupations_response
from retrievers.esco_retriever import search_esco
from retrievers.onet_retriever import get_onet_by_title, get_related_occupations_by_title, search_onet
from router.query_router import route_query

# Synonym map: normalised user phrase → exact O*NET occupation title
OCCUPATION_SYNONYMS: dict[str, str] = {
    "software development": "Software Developers",
    "software engineering": "Software Developers",
    "software developer": "Software Developers",
    "software engineer": "Software Developers",
    "cybersecurity": "Information Security Analysts",
    "cyber security": "Information Security Analysts",
    "information security": "Information Security Analysts",
    "data science": "Data Scientists",
    "data scientist": "Data Scientists",
    "cloud engineering": "Computer Systems Engineers/Architects",
    "cloud engineer": "Computer Systems Engineers/Architects",
    "cloud computing": "Computer Network Architects",
    "cloud architect": "Computer Network Architects",
    "machine learning": "Computer and Information Research Scientists",
    "ml engineering": "Computer and Information Research Scientists",
    "artificial intelligence": "Computer and Information Research Scientists",
    "web development": "Web Developers",
    "web developer": "Web Developers",
    "data engineering": "Database Architects",
    "data engineer": "Database Architects",
    "devops": "Software Developers",
    "network engineering": "Computer Network Architects",
    "network engineer": "Computer Network Architects",
}

# Phrases that unambiguously signal a "related occupations" query
_RELATED_INTENT_PHRASES = [
    "related occupation", "related career", "similar career", "similar job",
    "similar occupation", "careers related to", "jobs related to",
    "occupations related to", "closely related", "alternative career",
    "career alternative", "career transition", "similar role", "related role",
    "similar to", "related to",
]

# Regex patterns to extract the target phrase from related-occupations queries
_RELATED_OCC_PATTERNS = [
    r"occupations?\s+(?:are\s+)?(?:closely\s+)?related\s+to\s+(.+?)[\?\.]*$",
    r"careers?\s+(?:are\s+)?(?:closely\s+)?related\s+to\s+(.+?)[\?\.]*$",
    r"jobs?\s+(?:are\s+)?(?:closely\s+)?related\s+to\s+(.+?)[\?\.]*$",
    r"(?:careers?|occupations?|jobs?|roles?)\s+(?:are\s+)?similar\s+to\s+(.+?)[\?\.]*$",
    r"similar\s+(?:careers?|occupations?|jobs?|roles?)\s+(?:to|as)\s+(.+?)[\?\.]*$",
    r"(?:careers?|occupations?|jobs?|roles?)\s+related\s+to\s+(.+?)[\?\.]*$",
    r"alternative\s+(?:careers?|occupations?|jobs?)\s+(?:for|to)\s+(.+?)[\?\.]*$",
    r"career\s+(?:alternatives?|transitions?)\s+(?:from|for|to)\s+(.+?)[\?\.]*$",
    r"(?:show|find|list)\s+(?:careers?|occupations?|jobs?|roles?)\s+similar\s+to\s+(.+?)[\?\.]*$",
]

_CONFIDENCE_THRESHOLD = 0.45  # minimum token-overlap to trust a metadata match


def _detect_related_occupations_intent(query: str) -> bool:
    q = query.lower().strip()
    return any(p in q for p in _RELATED_INTENT_PHRASES)


def _extract_related_occ_target(query: str) -> str:
    q = query.strip()
    for pattern in _RELATED_OCC_PATTERNS:
        m = re.search(pattern, q, re.I)
        if m:
            return m.group(1).strip().lower().rstrip("?.! ")
    return ""


def _resolve_onet_title(phrase: str) -> str:
    """Map a free-form phrase to the best O*NET occupation title."""
    phrase_lower = phrase.lower().strip()
    # Exact synonym hit
    if phrase_lower in OCCUPATION_SYNONYMS:
        return OCCUPATION_SYNONYMS[phrase_lower]
    # Partial synonym hit
    for key, title in OCCUPATION_SYNONYMS.items():
        if key in phrase_lower or phrase_lower in key:
            return title
    # Return title-cased phrase as a best-effort guess for fuzzy matching downstream
    return phrase.title()


def _handle_related_occupations(query: str) -> str:
    phrase = _extract_related_occ_target(query)
    if not phrase:
        # Couldn't parse a target — fall through to normal pipeline
        return ""

    resolved_title = _resolve_onet_title(phrase)
    related, matched_title, confidence = get_related_occupations_by_title(resolved_title)

    logging.getLogger(__name__).info(
        "[RELATED_OCC] phrase=%r resolved=%r matched=%r confidence=%.3f",
        phrase, resolved_title, matched_title, confidence,
    )

    if confidence < _CONFIDENCE_THRESHOLD or not related:
        suggestions = ["Computer Systems Engineers/Architects", "Computer Network Architects",
                       "Software Developers", "Network and Computer Systems Administrators"]
        return (
            f"I couldn't confidently identify an occupation matching '{phrase}'. "
            f"Did you mean one of these?\n"
            + "\n".join(f"- {s}" for s in suggestions)
        )

    return build_related_occupations_response(matched_title, related)


def _extract_role_from_query(query: str) -> str:
    q = " ".join(query.strip().split())
    patterns = [
        r"what does (?:an?|the)\s+(.+?)\s+do\??$",
        r"tell me about\s+(.+?)\s+(?:career|careers)\??$",
        r"about\s+(.+?)\s+(?:career|careers)\??$",
    ]
    for pattern in patterns:
        m = re.search(pattern, q, flags=re.I)
        if m:
            return m.group(1).strip(" ?.")
    return ""


def _role_title_variants(role: str) -> list[str]:
    if not role:
        return []
    base = " ".join(role.split()).strip()
    variants = [base]
    if not base.lower().endswith("s"):
        variants.append(base + "s")
    return variants


def _norm_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"a", "an", "the", "and", "or", "of", "in", "to", "for"}
    return {t for t in tokens if t not in stop}


def _pick_best_role_match(role: str, candidates: list[dict]) -> list[dict]:
    role_tokens = _norm_tokens(role)
    if not role_tokens:
        return candidates[:1] if candidates else []
    scored: list[tuple[float, dict]] = []
    for c in candidates:
        title = (c.get("title") or "").strip()
        if not title:
            continue
        title_tokens = _norm_tokens(title)
        if not title_tokens:
            continue
        overlap = len(role_tokens & title_tokens) / max(1, len(role_tokens))
        if overlap <= 0:
            continue
        scored.append((overlap, c))
    if not scored:
        return []
    scored.sort(key=lambda x: (x[0], float(x[1].get("score", 0) or 0)), reverse=True)
    return [scored[0][1]]


def run_pipeline(query: str) -> str:
    if _detect_related_occupations_intent(query):
        answer = _handle_related_occupations(query)
        if answer:
            return answer

    role = _extract_role_from_query(query)
    if role:
        direct = get_onet_by_title(_role_title_variants(role))
        if direct:
            return generate_response(query, direct)
        # Role intent fallback: keep retrieval occupation-focused (ONET only).
        role_candidates = search_onet(role, top_k=1)
        best_role = _pick_best_role_match(role, role_candidates)
        if best_role:
            return generate_response(query, best_role)

    selected_sources, scores, query_embedding = route_query(query)
    if not selected_sources:
        return "No relevant career information was found for your query."
    logger.info("Selected sources: %s", selected_sources)
    logger.info("Retrieval scores: %s", {k: round(v, 4) for k, v in scores.items()})

    results: list[dict] = []
    source_handlers = {
        "ESCO": search_esco,
        "ONET": search_onet,
    }
    for source_name in selected_sources:
        handler = source_handlers.get(source_name)
        if handler is None:
            continue
        source_results = handler(query, embedding=query_embedding)
        logger.info("%s retrieval executed; returned %d results", source_name, len(source_results))
        results.extend(source_results)

    seen_titles: set[str] = set()
    unique: list[dict] = []
    for r in results:
        title = (r.get("title") or "").strip().lower()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        unique.append(r)

    if role and unique:
        exact = [r for r in unique if (r.get("title", "").strip().lower() == role.lower())]
        if exact:
            unique = exact

    # Keep at least 1 result per selected source so multi-source responses are complete.
    context_results: list[dict] = []
    seen_sources: set[str] = set()
    for r in unique:
        src = (r.get("source") or "").upper()
        if src not in seen_sources:
            context_results.append(r)
            seen_sources.add(src)
    # Fill remaining slots (up to 6 total) with extra results in score order.
    for r in unique:
        if len(context_results) >= 6:
            break
        if r not in context_results:
            context_results.append(r)

    logger.info(
        "Unique retrieval results: %d | sources=%s",
        len(unique),
        [r.get("source", "") for r in unique[:5]],
    )
    logger.info(
        "Context chunks passed to generator: %s",
        [f"{r.get('source','')}:{r.get('title','')}" for r in context_results],
    )
    return generate_response(query, context_results)


if __name__ == "__main__":
    print("\n==============================")
    print("   Career RAG System")
    print("==============================")
    while True:
        question = input("\nAsk Career Question (or type 'exit'): ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        print("\nGenerating answer...")
        answer = run_pipeline(question)
        print(f"\nAnswer: {answer}")
