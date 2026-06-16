import math
import os
import logging
import re
from dotenv import load_dotenv
from langfuse import Langfuse
from llm.education_data import lookup_education, format_education_response

load_dotenv()
_langfuse = Langfuse()
logger = logging.getLogger(__name__)
DEBUG_GENERATION = os.getenv("RAG_DEBUG_GENERATION", "0").lower() in {"1", "true", "yes", "on"}

_NAN_STRINGS = {"nan", "none", "null", "n/a", "na", "not available", ""}

# ---------------------------------------------------------------------------
# Model loader (lazy — loads on first call)
# ---------------------------------------------------------------------------

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from transformers import pipeline

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype = torch.float32

    logger.info("Loading %s on CPU (dtype=%s) — this takes ~30–60 s the first time", model_id, dtype)

    _pipeline = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    logger.info("Qwen model ready.")
    return _pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in _NAN_STRINGS else s


def _l2_to_similarity(distance) -> float | None:
    if distance is None:
        return None
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return None
    return 1.0 / (1.0 + d)


_EDUCATION_KEYWORDS = {
    "education", "degree", "qualification", "qualifications",
    "certification", "certifications", "certificate", "certificates",
    "study", "studies", "college", "university",
    "bachelor", "bachelor's", "masters", "master's", "phd", "doctorate",
    "academic", "training", "license", "licensing",
    "required education", "required degree", "educational requirement",
}


def _detect_education_intent(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _EDUCATION_KEYWORDS)


def _extract_occupation_from_query(question: str, chunks: list[dict]) -> str:
    patterns = [
        r"(?:for|become|becoming|as a|as an)\s+(?:a\s+|an\s+)?([a-zA-Z\s]+?)(?:\?|$|\.|,)",
        r"(?:education|degree|qualifications?|certifications?|training)\s+(?:required|needed|common|useful)\s+for\s+(?:a\s+|an\s+)?([a-zA-Z\s]+?)(?:\?|$|\.|,)",
        r"(?:do|does)\s+(?:a\s+|an\s+)?([a-zA-Z\s]+?)\s+need",
        r"(?:what\s+degree|what\s+education)\s+is\s+(?:common|required|needed)\s+among\s+([a-zA-Z\s]+?)(?:\?|$|\.|,)",
    ]
    for pat in patterns:
        m = re.search(pat, question, re.I)
        if m:
            candidate = m.group(1).strip().rstrip("s?.,")
            if len(candidate) > 3:
                return candidate
    if chunks:
        return chunks[0].get("title", "")
    return ""


def _build_education_response(question: str, chunks: list[dict]) -> str | None:
    occupation = _extract_occupation_from_query(question, chunks)
    edu_data = lookup_education(occupation) if occupation else None
    if edu_data:
        return format_education_response(occupation.title(), edu_data)
    for chunk in chunks:
        title = _clean(chunk.get("title", ""))
        edu_data = lookup_education(title)
        if edu_data:
            return format_education_response(title, edu_data)
    return None


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(results: list[dict]) -> tuple[str, list[dict]]:
    """Convert raw retrieval results into a plain-text context block."""
    parts: list[str] = []
    usable_chunks: list[dict] = []

    for r in results[:6]:
        if not isinstance(r, dict):
            text = _clean(str(r))
            if text:
                parts.append(text)
                usable_chunks.append({"title": "", "source": "", "score": None, "chars": len(text)})
            continue

        title = _clean(r.get("title", ""))
        source = _clean(r.get("source", ""))
        score = r.get("score")
        content = r.get("content", "")

        if isinstance(content, list):
            content = ", ".join(_clean(c) for c in content if _clean(c))
        else:
            content = _clean(content)

        if not content:
            continue

        header = f"[{source}] {title}" if source else title
        parts.append(f"{header}\n{content}" if header else content)
        usable_chunks.append({
            "title": title,
            "source": source,
            "score": score,
            "similarity": _l2_to_similarity(score),
            "chars": len(content),
        })

    if DEBUG_GENERATION:
        for i, c in enumerate(usable_chunks, 1):
            sim = c.get("similarity")
            sim_txt = f"{sim:.4f}" if sim is not None else "n/a"
            logger.debug(
                "[GEN-DEBUG] %d. source=%s title=%r sim=%s chars=%d",
                i, c.get("source", "?"), c.get("title", "")[:80], sim_txt, c.get("chars", 0),
            )
        logger.debug("[GEN-DEBUG] Chunk count: %d", len(usable_chunks))

    return "\n\n---\n\n".join(parts), usable_chunks


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a knowledgeable career counsellor assistant. \
Answer the user's career question using ONLY the information provided in the context below.

Format your answer using proper markdown:
- Use ## for main section headings
- Use ### for sub-headings
- Use bullet points (-) for lists
- Leave a blank line between sections
- Do NOT write everything in one paragraph

If the context does not contain enough information to answer the question, say so honestly."""


def _call_llm(question: str, context: str) -> str:
    pipe = _get_pipeline()

    user_message = f"""Context information from career databases:

{context}

---

Question: {question}

Please provide a detailed, well-structured answer based on the context above."""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    output = pipe(
        messages,
        max_new_tokens=200,
        do_sample=False,           # greedy — deterministic and faster
        temperature=None,          # must be None when do_sample=False
        top_p=None,                # same
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,    # return only the generated portion
    )

    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        text = generated[-1]["content"].strip()
    else:
        text = generated.strip()

    # Convert semicolon-separated lists into bullet points
    if text.count(";") >= 3 and "\n" not in text[:200]:
        match = re.search(r"^(.+?(?:includes?|are|:))\s+(.+)$", text, re.S | re.I)
        if match:
            intro = match.group(1).strip().rstrip(":") + ":"
            body = match.group(2).strip()
            items = [i.strip().rstrip(".") for i in re.split(r";|(?<=\w)\.\s+(?=[A-Z])", body) if i.strip()]
            text = intro + "\n\n" + "\n".join(f"- {item}" for item in items if item)

    text = re.sub(r"\s*(#{1,3})\s+", r"\n\n\1 ", text)
    text = re.sub(r"(?<!\n)-\s+", r"\n- ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Langfuse logging
# ---------------------------------------------------------------------------

def _log_generation(question: str, answer: str, query_type: str, model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
    input_tokens = len(question.split())
    output_tokens = len(answer.split()) if answer else 0
    _langfuse.generation(
        name="career-query",
        model=model,
        input=question,
        output=answer,
        usage={
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
            "unit": "TOKENS",
        },
        metadata={"type": query_type},
    )
    _langfuse.flush()


# ---------------------------------------------------------------------------
# Public API (same signatures as before — chatbot.py requires no changes)
# ---------------------------------------------------------------------------

def build_related_occupations_response(occupation_title: str, related: list[str]) -> str:
    if not related:
        return f"No related occupations found for {occupation_title}."
    lines = [f"Related Occupations for {occupation_title}\n"]
    for i, occ in enumerate(related[:10], 1):
        lines.append(f"{i}. {occ}")
    return "\n".join(lines)


def generate_response(question: str, results) -> str:
    if not results:
        return "No relevant career information was found for your query."

    # Pre-built context string (passed directly)
    if isinstance(results, str):
        context = _clean(results)
        if not context:
            return "No relevant career information was found for your query."
        answer = _call_llm(question, context)
        _log_generation(question, answer, "general")
        return answer

    # List of result dicts
    dict_results = [r for r in results if isinstance(r, dict)]

    # Education fast-path: use structured lookup data when available
    if _detect_education_intent(question):
        edu_answer = _build_education_response(question, dict_results)
        if edu_answer:
            _log_generation(question, edu_answer, "education", model="education-lookup")
            return edu_answer

    context, usable_chunks = _build_context(results)
    if not context:
        return "No relevant career information was found for your query."

    answer = _call_llm(question, context)
    if not answer:
        return "No relevant career information was found for your query."
    query_type = "education_llm" if _detect_education_intent(question) else "general"
    _log_generation(question, answer, query_type)
    return answer
