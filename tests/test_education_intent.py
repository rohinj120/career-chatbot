"""
Unit tests for education/qualification intent detection and response generation.

Run with:
    python -m pytest tests/test_education_intent.py -v
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from llm.education_data import lookup_education, format_education_response, EDUCATION_LOOKUP
from llm.generate_response import (
    _detect_education_intent,
    _extract_occupation_from_query,
    _build_education_response,
)


# ── Intent detection ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("query", [
    "What education is required for a software developer?",
    "Do data scientists need a master's degree?",
    "What qualifications are needed to become a database administrator?",
    "What degree is common among information security analysts?",
    "What certifications are useful for cybersecurity analysts?",
    "What training is required for cloud engineers?",
    "Do I need a bachelor's degree to become a data analyst?",
    "What college degree is best for a financial analyst?",
    "What PhD programs are available for machine learning engineers?",
    "Is a university degree required for web developers?",
])
def test_education_intent_detected(query):
    assert _detect_education_intent(query), f"Education intent NOT detected for: {query!r}"


@pytest.mark.parametrize("query", [
    "What does a software developer do?",
    "What are the responsibilities of a data scientist?",
    "Tell me about information security analysts.",
    "What skills does a cloud engineer need?",
])
def test_non_education_intent_not_triggered(query):
    assert not _detect_education_intent(query), f"Education intent wrongly detected for: {query!r}"


# ── Occupation extraction ────────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected_fragment", [
    ("What education is required for a software developer?", "software developer"),
    ("Do data scientists need a master's degree?", "data scientist"),
    ("What qualifications are needed to become a database administrator?", "database administrator"),
    ("What degree is common among information security analysts?", "information security analyst"),
])
def test_occupation_extracted(query, expected_fragment):
    occupation = _extract_occupation_from_query(query, [])
    assert expected_fragment.lower() in occupation.lower(), (
        f"Expected {expected_fragment!r} in extracted occupation {occupation!r} for query: {query!r}"
    )


# ── Education lookup ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("title", [
    "software developer",
    "data scientist",
    "database administrator",
    "information security analyst",
    "cybersecurity analyst",
    "cloud engineer",
])
def test_lookup_returns_data(title):
    data = lookup_education(title)
    assert data is not None, f"No education data found for {title!r}"
    assert "typical_education" in data
    assert "common_fields" in data


# ── Response generation (unit-level, no FAISS) ───────────────────────────────

@pytest.mark.parametrize("query, occ_title, expected_substring", [
    (
        "What education is required for a software developer?",
        "software developers",
        "Bachelor",
    ),
    (
        "Do data scientists need a master's degree?",
        "data scientists",
        "Master",
    ),
    (
        "What qualifications are needed to become a database administrator?",
        "database administrators",
        "Bachelor",
    ),
    (
        "What degree is common among information security analysts?",
        "information security analysts",
        "Bachelor",
    ),
    (
        "What certifications are useful for cybersecurity analysts?",
        "cybersecurity analysts",
        "Security+",
    ),
    (
        "What training is required for cloud engineers?",
        "cloud engineers",
        "AWS",
    ),
])
def test_education_response_content(query, occ_title, expected_substring):
    """
    Simulate what _build_education_response does with a single mock chunk
    whose title matches a known occupation.
    """
    mock_chunks = [{"title": occ_title, "content": "", "source": "ONET", "score": 1.0}]
    response = _build_education_response(query, mock_chunks)
    assert response is not None, f"No education response returned for: {query!r}"
    assert expected_substring.lower() in response.lower(), (
        f"Expected {expected_substring!r} in response for {query!r}.\nGot:\n{response}"
    )


def test_education_response_format():
    """Education response must contain the required sections."""
    data = lookup_education("software developers")
    response = format_education_response("Software Developers", data)
    assert "Typical education" in response
    assert "Common fields of study" in response
    assert "Relevant certifications" in response


def test_fallback_message_when_no_data():
    """When no lookup match exists, _build_education_response returns None."""
    mock_chunks = [{"title": "Underwater Basket Weaver", "content": "", "source": "ONET", "score": 0.5}]
    result = _build_education_response("What education is needed for an underwater basket weaver?", mock_chunks)
    assert result is None
