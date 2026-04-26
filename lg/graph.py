from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from lg.state import AgentState
from lg.nodes import llm_node, tool_node, writer_node, critic_node
from langgraph.checkpoint.postgres import PostgresSaver
import os
from dotenv import load_dotenv
load_dotenv()


_checkpointer_cm = PostgresSaver.from_conn_string(os.getenv("SUPABASE_URL"))
checkpointer = _checkpointer_cm.__enter__()
checkpointer.setup() 

def route_after_llm(state: AgentState) -> str:
    last = state["messages"][-1]
    content = last.content or ""

    # ✅ LLM couldn't find enough data → exit early
    if content.startswith("INSUFFICIENT_DATA"):
        return "end"

    # ✅ hit step cap → go write with what we have
    if content == "I have gathered enough information.":
        return "writer"

    # ✅ still has tool calls → keep researching
    if getattr(last, "tool_calls", None):
        return "tools"

    return "writer"

def route_after_critic(state: AgentState) -> str:
    if state.get("feedback"):
        return "writer"                   
    return "__end__"                      

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("llm")

    graph.add_conditional_edges(
        "llm",
        route_after_llm,
        {"tools": "tools", "writer": "writer","end":END}
    )

    graph.add_edge("tools", "llm")
    graph.add_edge("writer", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"writer": "writer", "__end__": END}
    )

    return graph.compile(checkpointer=checkpointer)
