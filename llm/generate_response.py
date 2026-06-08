import math
import os
import logging
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging as hf_logging
from llm.education_data import lookup_education, format_education_response

_model = None
_tokenizer = None
logger = logging.getLogger(__name__)
DEBUG_GENERATION = os.getenv("RAG_DEBUG_GENERATION", "0").lower() in {"1", "true", "yes", "on"}
hf_logging.set_verbosity_error()

_NAN_STRINGS = {"nan", "none", "null", "n/a", "na", "not available", ""}


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


def _unique_items(text: str, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for raw in re.split(r";|\n|,", text):
        item = re.sub(r"^(skill|skills|ability|abilities|knowledge areas?|tasks?|description):\s*", "", raw.strip(), flags=re.I)
        item = " ".join(item.split())
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _unique_phrases(text: str, limit: int = 6) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for raw in re.split(r";|\n", text):
        item = re.sub(r"^(tasks?|work activities?):\s*", "", raw.strip(), flags=re.I)
        item = " ".join(item.split()).strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _extract_labeled_items(text: str, labels: tuple[str, ...], limit: int = 6) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key_n = re.sub(r"\s+", " ", key.strip().lower())
        if key_n not in labels:
            continue
        val = " ".join(val.strip().split())
        k = val.lower()
        if not val or k in seen:
            continue
        seen.add(k)
        items.append(val)
        if len(items) >= limit:
            break
    return items


def _format_bullets(lines: list[str], max_items: int = 6, max_chars: int = 160) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for line in lines:
        text = _clean(line)
        if not text:
            continue
        text = re.sub(r"^[-*]\s*", "", text).strip()
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:") + "..."
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(f"- {text}")
        if len(cleaned) >= max_items:
            break
    return "\n\n".join(cleaned)


def _extract_field(content: str, field_name: str) -> str:
    pattern = rf"{re.escape(field_name)}:\s*(.*?)(?=\n\n[A-Z][A-Za-z ]+:\s*|\Z)"
    match = re.search(pattern, content, flags=re.S | re.I)
    return match.group(1).strip() if match else ""


def _short_text(text: str, limit: int = 520) -> str:
    compact = " ".join(_clean(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rsplit(" ", 1)[0] + "."


def _join_items(items: list[str], separator: str = ", ") -> str:
    cleaned = [item.rstrip(". ") for item in items if _clean(item)]
    return separator.join(cleaned)


def _infer_items_from_text(text: str, kind: str, limit: int = 8) -> list[str]:
    haystack = text.lower()
    if kind == "skills":
        candidates = [
            ("data-oriented programming", "Data-oriented programming"),
            ("statistical software", "Statistical software"),
            ("data mining", "Data mining"),
            ("data modeling", "Data modeling"),
            ("natural language processing", "Natural language processing"),
            ("machine learning", "Machine learning"),
            ("visualization", "Data visualization"),
            ("model", "Model evaluation"),
            ("clean", "Data cleaning"),
            ("analyze", "Data analysis"),
            ("security", "Security analysis"),
            ("vulnerab", "Vulnerability assessment"),
            ("legal", "Legal analysis"),
            ("interpret laws", "Interpreting laws"),
        ]
    else:
        candidates = [
            ("data", "Data analysis"),
            ("statistical", "Statistics"),
            ("machine learning", "Machine learning"),
            ("natural language processing", "Natural language processing"),
            ("programming", "Programming"),
            ("visualization", "Data visualization"),
            ("computer", "Computers and electronics"),
            ("security", "Cybersecurity"),
            ("network", "Computer networks"),
            ("law", "Law and government"),
            ("legal", "Legal procedures"),
        ]

    found = [label for token, label in candidates if token in haystack]
    return _unique_items("; ".join(found), limit=limit)


_EDUCATION_KEYWORDS = {
    "education", "degree", "qualification", "qualifications",
    "certification", "certifications", "certificate", "certificates",
    "study", "studies", "college", "university",
    "bachelor", "bachelor's", "masters", "master's", "phd", "doctorate",
    "academic", "training", "license", "licensing",
    "required education", "required degree", "educational requirement",
}

def _detect_education_intent(question: str) -> bool:
    """Return True when the question is asking about education/qualifications."""
    q = question.lower()
    return any(kw in q for kw in _EDUCATION_KEYWORDS)


def _extract_occupation_from_query(question: str, chunks: list[dict]) -> str:
    """Best-effort extraction of the occupation name from question or top chunk."""
    # Try to strip common question prefixes to isolate the occupation
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

    # Fall back to the top retrieved chunk's title
    if chunks:
        return chunks[0].get("title", "")
    return ""


def _build_education_response(question: str, chunks: list[dict]) -> str | None:
    """
    Try to return an education-focused answer.
    Returns None if no education data is found (caller should fall back to overview).
    """
    occupation = _extract_occupation_from_query(question, chunks)
    edu_data = lookup_education(occupation) if occupation else None

    if edu_data:
        return format_education_response(occupation.title(), edu_data)

    # No lookup hit — try to scrape education hints from chunk content
    for chunk in chunks:
        content = _clean(chunk.get("content", ""))
        title = _clean(chunk.get("title", ""))
        edu_data = lookup_education(title)
        if edu_data:
            return format_education_response(title, edu_data)

    return None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        _model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return _model, _tokenizer


def _looks_weak_answer(text: str, question: str = "") -> bool:
    t = _clean(text)
    if not t:
        return True
    if len(t) < 20:
        return True
    lower = t.lower()
    q = _clean(question).lower()
    weak_markers = {"[esco]", "[onet]", "-", "not available"}
    if lower.strip() in weak_markers:
        return True
    normalized_answer = re.sub(r"[^a-z0-9 ]+", " ", lower)
    normalized_question = re.sub(r"[^a-z0-9 ]+", " ", q)
    if normalized_answer.strip() and normalized_answer.strip() in normalized_question:
        return True
    return False


def _fallback_answer(question: str, chunks: list[dict]) -> str:
    q = question.lower()
    asks_knowledge = any(word in q for word in ("knowledge", "subject", "subjects", "study"))
    if asks_knowledge:
        best = next((
            c for c in chunks
            if _clean(c.get("source", "")).upper() == "ONET"
            and _extract_field(_clean(c.get("content", "")), "Knowledge")
        ), None)
    else:
        best = next((c for c in chunks if _clean(c.get("source", "")).upper() == "ONET"), None)
    if best is None and chunks:
        best = chunks[0]

    if best:
        title = _clean(best.get("title", ""))
        content = _clean(best.get("content", ""))
        description = _short_text(_extract_field(content, "Description"))
        skills = _unique_items(_extract_field(content, "Skills"))
        abilities = _unique_items(_extract_field(content, "Abilities"), limit=5)
        knowledge = _unique_items(_extract_field(content, "Knowledge"), limit=7)
        tasks = _unique_phrases(_extract_field(content, "Tasks"), limit=6)
        activities = _unique_phrases(_extract_field(content, "Work Activities"), limit=6)
        styles = _unique_items(_extract_field(content, "Work Styles"), limit=4)
        if not skills:
            skills = _infer_items_from_text(content, "skills")
        if not knowledge:
            knowledge = _infer_items_from_text(content, "knowledge", limit=7)

        asks_skills = any(word in q for word in ("skill", "skills", "ability", "abilities", "required", "requirement", "requirements"))
        asks_duties = any(phrase in q for phrase in ("responsibilities", "responsibility", "what does", "what do", "duties", "do in", "does a", "does an"))
        asks_overview = any(phrase in q for phrase in ("tell me about", "about ", "describe", "overview", "explain"))
        has_specific_intent = asks_knowledge or asks_skills or asks_duties or asks_overview

        lines = []

        if asks_duties or asks_overview or not has_specific_intent:
            if description:
                label = "What they do" if asks_overview or not asks_duties else "Main purpose"
                lines.append(f"{label}: {description}")
            if tasks:
                lines.append(f"Typical responsibilities include: {_join_items(tasks, '; ')}.")
            elif activities:
                lines.append(f"Common work activities include: {_join_items(activities)}.")
            else:
                generic_duties = _extract_labeled_items(content, ("task", "tasks", "work activity", "work activities", "description"), limit=5)
                if generic_duties:
                    lines.append(f"Typical responsibilities include: {_join_items(generic_duties, '; ')}.")

        if asks_skills:
            if skills:
                lines.append(f"Important skills include: {_join_items(skills)}.")
            if abilities:
                lines.append(f"Useful abilities include: {_join_items(abilities)}.")

        if asks_knowledge or asks_skills or "subject" in q or "subjects" in q:
            if knowledge:
                lines.append(f"Important subjects/knowledge areas include: {_join_items(knowledge)}.")

        if not asks_duties and not asks_overview and not asks_skills and not asks_knowledge:
            if skills:
                lines.append(f"Important skills include: {_join_items(skills[:6])}.")
            if styles:
                lines.append(f"Helpful work styles include: {_join_items(styles)}.")

        # Strict intent mode: only answer exactly what was asked.
        if asks_skills and not asks_knowledge and not asks_duties and not asks_overview:
            lines = [l for l in lines if l.lower().startswith("important skills include") or l.lower().startswith("useful abilities include")]
        elif asks_knowledge and not asks_skills and not asks_duties and not asks_overview:
            lines = [l for l in lines if l.lower().startswith("important subjects/knowledge areas include")]
        elif asks_duties and not asks_skills and not asks_knowledge:
            lines = [
                l for l in lines
                if l.lower().startswith("main purpose")
                or l.lower().startswith("typical responsibilities include")
                or l.lower().startswith("common work activities include")
            ]

        if lines:
            return _format_bullets(lines)

    # Try to aggregate labeled entries (e.g. ESCO skill/ability records: "Skill: X")
    _LABEL_RE = re.compile(r"^(skill|ability|knowledge|task)s?:\s*(.+)$", re.I)
    labeled_items: list[str] = []
    labeled_seen: set[str] = set()
    for c in chunks[:8]:
        title = _clean(c.get("title", ""))
        m = _LABEL_RE.match(title)
        if m:
            item = m.group(2).strip()
            key = item.lower()
            if key and key not in labeled_seen:
                labeled_seen.add(key)
                labeled_items.append(item)
    if labeled_items:
        label_word = "Important skills include" if any(w in q for w in ("skill", "skills")) else "Relevant items include"
        return f"- {label_word}: {_join_items(labeled_items)}."

    lines: list[str] = []
    seen_lines: set[str] = set()
    wants_skills_only = any(word in q for word in ("skill", "skills"))
    for c in chunks[:1]:
        title = _clean(c.get("title", ""))
        content = _clean(c.get("content", ""))
        if not content:
            continue
        if not wants_skills_only:
            content_lines = [ln for ln in content.splitlines() if not ln.strip().lower().startswith("skill:")]
            content = "\n".join(content_lines).strip()
            if not content:
                continue
        # Avoid repeating the title in the snippet when content equals the title
        if content.lower() == title.lower():
            line = f"- {title}" if title else f"- {content}"
        else:
            compact = re.sub(r"\b(Skill|Ability|Knowledge|Task):\s*\1:\s*", r"\1: ", content, flags=re.I)
            compact = " ".join(compact.split())
            snippet = compact[:240].rsplit(" ", 1)[0] if len(compact) > 240 else compact
            line = f"- {title}: {snippet}" if title else f"- {snippet}"
        key = line.lower()
        if key in seen_lines:
            continue
        seen_lines.add(key)
        lines.append(line)
    if lines:
        return _format_bullets(lines)
    return "No relevant career information was found for your query."


def build_related_occupations_response(occupation_title: str, related: list[str]) -> str:
    """Format a numbered list of related occupations."""
    if not related:
        return f"No related occupations found for {occupation_title}."
    lines = [f"Related Occupations for {occupation_title}\n"]
    for i, occ in enumerate(related[:10], 1):
        lines.append(f"{i}. {occ}")
    return "\n".join(lines)


def generate_response(question: str, results) -> str:
    if not results:
        return "No relevant career information was found for your query."

    # Accept either a pre-built context string or a list of result dicts
    if isinstance(results, str):
        context = _clean(results)
    else:
        parts = []
        usable_chunks = []
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
            header = f"Source={source} | Title={title}" if source else f"Title={title}"
            parts.append(f"{header}\n{content}" if header else content)
            usable_chunks.append({
                "title": title,
                "source": source,
                "score": score,
                "similarity": _l2_to_similarity(score),
                "chars": len(content),
            })

        if not parts:
            return "No relevant career information was found for your query."
        context = "\n\n---\n\n".join(parts)

        if DEBUG_GENERATION:
            print("\n[GEN-DEBUG] Retrieved chunks passed to LLM:")
            for i, c in enumerate(usable_chunks, 1):
                sim = c.get("similarity")
                sim_txt = f"{sim:.4f}" if sim is not None else "n/a"
                print(
                    f"[GEN-DEBUG] {i}. source={c.get('source','?') or '?'} "
                    f"title={c.get('title','')[:80]!r} sim={sim_txt} chars={c.get('chars',0)}"
                )
            print(f"[GEN-DEBUG] Chunk count passed to LLM: {len(usable_chunks)}")

    if not context:
        return "No relevant career information was found for your query."

    if isinstance(results, list):
        dict_results = [r for r in results if isinstance(r, dict)]

        # Education / qualification intent — handled before generic fallback
        if _detect_education_intent(question):
            edu_answer = _build_education_response(question, dict_results)
            if edu_answer:
                return edu_answer
            # No education data found — tell the user and show overview instead
            overview = _fallback_answer(question, dict_results)
            if overview != "No relevant career information was found for your query.":
                return (
                    "I could not find specific education requirements for this occupation. "
                    "Here is the occupation overview instead.\n\n" + overview
                )
            return "I could not find specific education requirements for this occupation."

        structured_answer = _fallback_answer(question, dict_results)
        if structured_answer != "No relevant career information was found for your query.":
            return structured_answer

    # Keep a larger context budget while still respecting model max input.
    if len(context) > 2600:
        context = context[:2600].rsplit(" ", 1)[0]

    prompt = (
        "You are an intelligent career assistant.\n\n"
        "Use only the retrieved context below to answer the user's question.\n\n"
        "If the context is partially relevant, provide the best possible answer based on the available information.\n\n"
        "Do not hallucinate information not present in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly in 4-6 concise bullet points. Mention specific skills/tasks/salary/growth only if present."
    )

    if DEBUG_GENERATION:
        print(f"[GEN-DEBUG] Context length (chars): {len(context)}")
        print("[GEN-DEBUG] Final prompt sent to model:")
        print(prompt)

    model, tokenizer = _get_model()
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
    )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if DEBUG_GENERATION:
        print(f"[GEN-DEBUG] Raw model output: {raw!r}")

    lines = [l.strip() for l in raw.replace(". ", ".\n").splitlines() if l.strip()]
    # Drop any line that is purely a nan/none artefact
    bullet_lines = [
        (l if l.startswith("-") else f"- {l}")
        for l in lines
        if _clean(l)
    ]
    if not bullet_lines or _looks_weak_answer(raw, question):
        if DEBUG_GENERATION:
            print("[GEN-DEBUG] Weak generation detected, using fallback answer builder.")
        if isinstance(results, list):
            return _fallback_answer(question, [r for r in results if isinstance(r, dict)])
        return "No relevant career information was found for your query."
    return _format_bullets(bullet_lines)
