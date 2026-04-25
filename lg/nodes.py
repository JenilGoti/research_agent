from langgraph.prebuilt import ToolNode


from llm.groq_client import get_llm
from tools.web_search import web_search
from tools.scraper import scrape_url
from tools.search_kb import search_knowledge_base
from langchain_core.messages import AIMessage, SystemMessage


llm = get_llm()

tools = [web_search, scrape_url,search_knowledge_base]

llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

tool_node = ToolNode(tools)

def llm_node(state):
    if state.get("tool_calls_count", 0) > 5:
        return {
            "messages": [AIMessage(content="I have gathered enough information.")],
            "tool_calls_count": state.get("tool_calls_count", 0)
        }

    messages = list(state["messages"])

    # ✅ prepend system message on first call only
    if not any(isinstance(m, SystemMessage) for m in messages):
        has_docs = state.get("docs_ingested", False)

        system_prompt = f"""You are an AI research assistant.

{"✅ The user has uploaded documents. ALWAYS use the `search_knowledge_base` tool first before searching the web." if has_docs else "⚠️ No documents have been uploaded. Use web search and scraping tools only."}

When you have gathered enough information, stop calling tools and write your findings.
If you cannot find relevant information after searching, respond EXACTLY with:
'INSUFFICIENT_DATA: <reason why>'
"""
        messages = [SystemMessage(content=system_prompt)] + messages

    response = llm_with_tools.invoke(messages)
    made_tool_calls = bool(getattr(response, "tool_calls", None))

    return {
        "messages": [response],
        "tool_calls_count": state.get("tool_calls_count", 0) + (1 if made_tool_calls else 0)
    }
def writer_node(state):
    context = "\n\n".join(
        f"{m.type.upper()}: {m.content}"
        for m in state["messages"]
        if hasattr(m, "content") and m.content
    )

    prompt = f"""You are a research report writer.
Based on the research below, write a clear, structured report.

RESEARCH:
{context}

Write the final report now:"""

    return {"messages": [llm.invoke(prompt)]}

# lg/nodes.py
def critic_node(state):
    report = state["messages"][-1]
    report_text = report.content if hasattr(report, "content") else str(report)
    feedback = state.get("feedback", None)

    review = llm.invoke(
        f"Review this report:\n\n{report_text}"
        + (f"\n\nUser feedback to address: {feedback}" if feedback else "")
    )
    return {"messages": [review], "feedback": None}   # ✅ clear feedback after use