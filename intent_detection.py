import logging
import os
import sys
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s - %(message)s",
)
from router.query_router import route_query
from retrievers.esco_retriever import search_esco
from retrievers.onet_retriever import search_onet
from llm.generate_response import generate_response
def run_pipeline(query: str) -> None:
    selected_sources, _, query_embedding = route_query(query)
    if not selected_sources:
        print("No relevant career information was found for your query.")
        return
    results: list[dict] = []
    if "ESCO" in selected_sources:
        results.extend(search_esco(query, embedding=query_embedding))
    if "ONET" in selected_sources:
        results.extend(search_onet(query, embedding=query_embedding))
    seen: set[str] = set()
    unique: list[dict] = []
    for r in results:
        title = r.get("title", "")
        if title not in seen:
            seen.add(title)
            unique.append(r)
    answer = generate_response(query, unique)
    print(answer)
if __name__ == "__main__":
    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break
        run_pipeline(query)