"""Tests for RELATED_OCCUPATIONS intent detection and retrieval."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot import (
    _detect_related_occupations_intent,
    _extract_related_occ_target,
    _resolve_onet_title,
    run_pipeline,
)
from retrievers.onet_retriever import get_related_occupations_by_title


# --- Unit: intent detection ---

def test_detect_related_occupations_software():
    assert _detect_related_occupations_intent("What occupations are related to software development?")

def test_detect_related_occupations_cybersecurity():
    assert _detect_related_occupations_intent("Show careers similar to cybersecurity")

def test_detect_related_occupations_data_science():
    assert _detect_related_occupations_intent("What jobs are related to data science")

def test_detect_related_occupations_cloud():
    assert _detect_related_occupations_intent("What occupations are closely related to cloud engineering")

def test_detect_does_not_trigger_on_overview():
    assert not _detect_related_occupations_intent("What does a software developer do?")
    assert not _detect_related_occupations_intent("Tell me about data science careers")


# --- Unit: target extraction ---

def test_extract_target_software():
    phrase = _extract_related_occ_target("What occupations are related to software development?")
    assert "software development" in phrase

def test_extract_target_cybersecurity():
    phrase = _extract_related_occ_target("Show careers similar to cybersecurity")
    assert "cybersecurity" in phrase

def test_extract_target_data_science():
    phrase = _extract_related_occ_target("What jobs are related to data science")
    assert "data science" in phrase

def test_extract_target_cloud():
    phrase = _extract_related_occ_target("What occupations are closely related to cloud engineering")
    assert "cloud engineering" in phrase


# --- Unit: synonym resolution ---

def test_resolve_software_development():
    assert _resolve_onet_title("software development") == "Software Developers"

def test_resolve_cybersecurity():
    assert _resolve_onet_title("cybersecurity") == "Information Security Analysts"

def test_resolve_data_science():
    assert _resolve_onet_title("data science") == "Data Scientists"

def test_resolve_cloud_engineering():
    assert _resolve_onet_title("cloud engineering") == "Computer Systems Engineers/Architects"


# --- Unit: O*NET related-occupation lookup ---

def test_onet_related_software_developers():
    related, matched, confidence = get_related_occupations_by_title("Software Developers")
    assert confidence >= 0.45, f"Low confidence: {confidence}"
    assert len(related) >= 3, f"Too few related occupations: {related}"
    titles_lower = [r.lower() for r in related]
    assert any("programmer" in t or "analyst" in t or "engineer" in t or "developer" in t for t in titles_lower)

def test_onet_related_information_security():
    related, matched, confidence = get_related_occupations_by_title("Information Security Analysts")
    assert confidence >= 0.45
    assert len(related) >= 3

def test_onet_related_data_scientists():
    related, matched, confidence = get_related_occupations_by_title("Data Scientists")
    assert confidence >= 0.45
    assert len(related) >= 3

def test_onet_related_cloud_engineering():
    related, matched, confidence = get_related_occupations_by_title("Computer Systems Engineers/Architects")
    assert confidence >= 0.45
    assert len(related) >= 3


# --- Integration: pipeline returns list, not overview ---

def _is_related_list(answer: str) -> bool:
    """True if answer looks like a numbered list of related occupations."""
    return "Related Occupations for" in answer and any(f"{i}." in answer for i in range(1, 6))

def _is_low_confidence_suggestion(answer: str) -> bool:
    return "couldn't confidently identify" in answer.lower()

def test_pipeline_software_development():
    answer = run_pipeline("What occupations are related to software development?")
    print("\n[software development]:", answer[:300])
    assert _is_related_list(answer) or _is_low_confidence_suggestion(answer)
    assert "Software Developers" not in answer.splitlines()[0] or "Related Occupations" in answer

def test_pipeline_cybersecurity():
    answer = run_pipeline("Show careers similar to cybersecurity")
    print("\n[cybersecurity]:", answer[:300])
    assert _is_related_list(answer) or _is_low_confidence_suggestion(answer)

def test_pipeline_data_science():
    answer = run_pipeline("What jobs are related to data science")
    print("\n[data science]:", answer[:300])
    assert _is_related_list(answer) or _is_low_confidence_suggestion(answer)

def test_pipeline_cloud_engineering():
    answer = run_pipeline("What occupations are closely related to cloud engineering")
    print("\n[cloud engineering]:", answer[:300])
    # Must not return fiber optics / unrelated occupations
    assert "fiber" not in answer.lower() and "optic" not in answer.lower()
    assert _is_related_list(answer) or _is_low_confidence_suggestion(answer)


if __name__ == "__main__":
    tests = [
        test_detect_related_occupations_software,
        test_detect_related_occupations_cybersecurity,
        test_detect_related_occupations_data_science,
        test_detect_related_occupations_cloud,
        test_detect_does_not_trigger_on_overview,
        test_extract_target_software,
        test_extract_target_cybersecurity,
        test_extract_target_data_science,
        test_extract_target_cloud,
        test_resolve_software_development,
        test_resolve_cybersecurity,
        test_resolve_data_science,
        test_resolve_cloud_engineering,
        test_onet_related_software_developers,
        test_onet_related_information_security,
        test_onet_related_data_scientists,
        test_onet_related_cloud_engineering,
        test_pipeline_software_development,
        test_pipeline_cybersecurity,
        test_pipeline_data_science,
        test_pipeline_cloud_engineering,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
