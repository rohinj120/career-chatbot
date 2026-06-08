import io
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chatbot
from retrievers import canada_retriever
from router.query_router import route_query


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANADA_INDEX = PROJECT_ROOT / "vector_db" / "descriptors_index.faiss"
CANADA_METADATA = PROJECT_ROOT / "embeddings" / "descriptors_metadata.pkl"
CANADA_QUERIES = [
    "software developer in canada",
    "data scientist in canada",
    "occupations related to software development in canada",
    "canadian career pathways",
    "noc software engineer",
]


def test_canada_artifacts_exist():
    assert CANADA_INDEX.exists(), f"Missing Canada index: {CANADA_INDEX}"
    assert CANADA_METADATA.exists(), f"Missing Canada metadata: {CANADA_METADATA}"


def test_canada_index_and_metadata_load():
    canada_retriever._index = None
    canada_retriever._metadata = []
    canada_retriever._load_attempted = False

    assert canada_retriever._load() is True
    assert canada_retriever._index is not None
    assert canada_retriever._index.ntotal > 0
    assert len(canada_retriever._metadata) > 0


def test_search_canada_returns_results():
    results = canada_retriever.search_canada("software developer in canada", top_k=3)
    assert results, "Expected Canada retriever to return at least one result"
    assert all(result["source"] == "CANADA" for result in results)
    assert all(result["title"] for result in results)


def test_router_selects_canada_for_canada_queries():
    for query in CANADA_QUERIES:
        selected_sources, scores, _ = route_query(query)
        assert "CANADA" in selected_sources, (query, selected_sources, scores)


def test_chatbot_passes_canada_context_to_generator():
    captured = {}
    original_generate_response = chatbot.generate_response

    def fake_generate_response(question, results):
        captured["question"] = question
        captured["results"] = results
        return "stubbed-response"

    chatbot.generate_response = fake_generate_response
    try:
        answer = chatbot.run_pipeline("What are the duties of a software developer in Canada?")
    finally:
        chatbot.generate_response = original_generate_response

    assert answer == "stubbed-response"
    assert "results" in captured
    assert any(item.get("source") == "CANADA" for item in captured["results"]), captured["results"]


def test_chatbot_logs_canada_execution():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    try:
        chatbot.run_pipeline("What are the duties of a software developer in Canada?")
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)

    logs = stream.getvalue()
    assert "Selected sources" in logs
    assert "Retrieval scores" in logs
    assert "CANADA retrieval executed" in logs


if __name__ == "__main__":
    tests = [
        test_canada_artifacts_exist,
        test_canada_index_and_metadata_load,
        test_search_canada_returns_results,
        test_router_selects_canada_for_canada_queries,
        test_chatbot_passes_canada_context_to_generator,
        test_chatbot_logs_canada_execution,
    ]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {test.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
