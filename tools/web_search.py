from langchain_core.tools import tool
from ddgs import DDGS

@tool
def web_search(query: str) -> str:
    """Search the web for information about a query."""
    results_text = []

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)

        for r in results:
            title = r.get("title", "")
            link = r.get("href", "")
            snippet = r.get("body", "")

            results_text.append(
                f"""
Title: {title}
Link: {link}
Snippet: {snippet}
"""
            )

    return "\n---\n".join(results_text)

