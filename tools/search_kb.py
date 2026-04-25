from langchain_core.tools import tool
from vector_db.qdrant_search import search_by_text

@tool
def search_knowledge_base(query: str) -> str:
    """Search in uploaded documents and return relevant context."""

    results = search_by_text(query, limit=2)

    if not results:
        return "No relevant documents found."
    return "\n\n".join(
        [f"- {r['text']}" for r in results]
    )