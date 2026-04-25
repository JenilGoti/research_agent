import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from lg.graph import build_graph
from ingestion.pipeline import ingest_all
import tempfile, os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    /* Tool call expander — muted */
    .tool-block {
        background: #1a1d2e;
        border-left: 3px solid #3d4166;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #8b8fa8;
        margin: 4px 0;
    }
    .tool-name {
        color: #6c8ebf;
        font-weight: 600;
        font-family: monospace;
    }
    /* Architecture diagram boxes */
    .arch-node {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 6px;
        padding: 6px 10px;
        margin: 3px 0;
        font-size: 0.78rem;
        color: #c9cde0;
    }
    .arch-arrow {
        color: #3d4166;
        text-align: center;
        font-size: 0.75rem;
        margin: 0;
    }
    /* Status badge */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-green  { background: #1a3a2a; color: #4ade80; }
    .badge-yellow { background: #3a2e1a; color: #fbbf24; }
    .badge-blue   { background: #1a2a3a; color: #60a5fa; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────
defaults = {
    "app": None,
    "thread_id": "user-1",
    "chat_history": [],
    "awaiting_feedback": False,
    "ingested": False,
    "pending_files": [],
    "tools_used": [],           # track tools used this session
    "total_chunks": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.app is None:
    st.session_state.app = build_graph()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Brand ─────────────────────────────────────────────────────────────────
    st.markdown("## 🤖 Research Agent")
    st.caption("Powered by LangGraph + Groq")
    st.divider()

    # ── System Architecture ───────────────────────────────────────────────────
    st.markdown("#### 🏗️ Architecture")
    st.markdown("""
<div class="arch-node">🧠 <b>LLM Node</b> — Planner & reasoner</div>
<div class="arch-arrow">↕ tool calls / results</div>
<div class="arch-node">🛠️ <b>Tool Node</b> — Executes tools</div>
<div class="arch-arrow">↓ when done researching</div>
<div class="arch-node">📄 <b>Writer Node</b> — Drafts report</div>
<div class="arch-arrow">↓</div>
<div class="arch-node">🔎 <b>Critic Node</b> — Reviews & scores</div>
<div class="arch-arrow">↓ feedback loop or END</div>
<div class="arch-node">✅ <b>END</b></div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Available Tools ───────────────────────────────────────────────────────
    st.markdown("#### 🛠️ Available Tools")
    tools_info = [
        ("🌐", "web_search",           "Search the web for current info"),
        ("🔗", "scrape_url",           "Scrape content from a URL"),
        ("📚", "search_knowledge_base","Search your uploaded documents"),
    ]
    for icon, name, desc in tools_info:
        st.markdown(f"""
<div class="arch-node">
    {icon} <span class="tool-name">{name}</span><br>
    <span style="color:#5a5f7a; font-size:0.72rem">{desc}</span>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Tools Used This Session ───────────────────────────────────────────────
    st.markdown("#### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tools Called", len(st.session_state.tools_used))
    with col2:
        st.metric("Chunks", st.session_state.total_chunks)

    if st.session_state.tools_used:
        with st.expander("Tools log"):
            for t in st.session_state.tools_used[-10:]:   # last 10
                st.caption(f"• `{t}`")

    st.divider()

    # ── Document Status ───────────────────────────────────────────────────────
    st.markdown("#### 📂 Documents")
    if st.session_state.ingested:
        st.markdown('<span class="badge badge-green">✅ Docs loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-yellow">⚠️ No docs loaded</span>', unsafe_allow_html=True)

    st.divider()

    # ── New Session ───────────────────────────────────────────────────────────
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.thread_id       = f"user-{os.urandom(4).hex()}"
        st.session_state.chat_history    = []
        st.session_state.awaiting_feedback = False
        st.session_state.tools_used      = []
        st.session_state.ingested        = False
        st.session_state.total_chunks    = 0
        st.rerun()

# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown("### 💬 Research Chat")

# ── File uploader (above chat, compact) ──────────────────────────────────────
uploaded_files = st.file_uploader(
    "📎 Attach documents (auto-ingested on send)",
    accept_multiple_files=True,
    type=["pdf", "txt", "csv", "docx", "xlsx", "json"],
    label_visibility="collapsed",
    help="Files are ingested automatically when you send your first message."
)
if uploaded_files:
    names = ", ".join(f.name for f in uploaded_files)
    st.caption(f"📎 Ready to ingest: **{names}**")

# ── Chat History ──────────────────────────────────────────────────────────────
role_cfg = {
    "user":    ("🧑", "user"),
    "planner": ("🧠", "assistant"),
    "writer":  ("📄", "assistant"),
    "critic":  ("🔎", "assistant"),
}

for msg in st.session_state.chat_history:
    role = msg["role"]

    if role == "tool_event":
        # minimized tool event — no chat bubble, just a tiny line
        st.markdown(
            f'<div class="tool-block">🛠️ <span class="tool-name">{msg["tool"]}</span>'
            f' &nbsp;›&nbsp; {msg["summary"]}</div>',
            unsafe_allow_html=True
        )
        continue

    icon, avatar = role_cfg.get(role, ("🤖", "assistant"))
    with st.chat_message(avatar):
        st.markdown(f"**{icon} {role.capitalize()}**")
        st.markdown(msg["content"])

# ─── Auto-ingest helper ───────────────────────────────────────────────────────
def maybe_ingest(files):
    """Ingest uploaded files if any. Returns True if something was ingested."""
    if not files:
        return False
    with tempfile.TemporaryDirectory() as tmp_dir:
        for f in files:
            path = os.path.join(tmp_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
        with st.spinner("📂 Ingesting documents..."):
            result = ingest_all(tmp_dir)
    # parse chunk count from "Ingested N chunks"
    try:
        n = int(result.split()[1])
        st.session_state.total_chunks += n
    except Exception:
        pass
    st.session_state.ingested = True
    st.toast(f"✅ {result}", icon="📂")
    return True

# ─── Stream helper ────────────────────────────────────────────────────────────
def run_stream(inputs: dict):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    events = st.session_state.app.stream(inputs, config=config)

    node_labels = {
        "llm":    ("🧠", "planner"),
        "writer": ("📄", "writer"),
        "critic": ("🔎", "critic"),
    }

    for event in events:
        for node_name, value in event.items():
            if "messages" not in value:
                continue

            messages = value["messages"]

            # ── Tool calls decided by LLM (minimized) ──────────────────────
            if node_name == "llm":
                last = messages[-1]
                if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                    for tc in last.tool_calls:
                        tool_name = tc["name"]
                        st.session_state.tools_used.append(tool_name)
                        summary = ", ".join(f"{k}={repr(v)[:40]}" for k, v in tc["args"].items())
                        st.session_state.chat_history.append({
                            "role":    "tool_event",
                            "tool":    tool_name,
                            "summary": summary or "…",
                        })
                    continue   # don't render LLM message if it only made tool calls

                # LLM produced a text response (no tool calls)
                last = messages[-1]
                if last.content and not last.content.startswith("INSUFFICIENT_DATA"):
                    icon, role = node_labels["llm"]
                    with st.chat_message("assistant"):
                        st.markdown(f"**{icon} Planner**")
                        st.markdown(last.content)
                    st.session_state.chat_history.append({
                        "role": role, "content": last.content
                    })

            # ── Tool results (minimized) ────────────────────────────────────
            elif node_name == "tools":
                for m in messages:
                    if isinstance(m, ToolMessage):
                        preview = m.content[:80] + "…" if len(m.content) > 80 else m.content
                        st.session_state.chat_history.append({
                            "role":    "tool_event",
                            "tool":    getattr(m, "name", "tool"),
                            "summary": preview,
                        })

            # ── Writer / Critic (full display) ─────────────────────────────
            elif node_name in node_labels:
                last = messages[-1]
                if not last.content:
                    continue
                icon, role = node_labels[node_name]
                with st.chat_message("assistant"):
                    st.markdown(f"**{icon} {role.capitalize()}**")
                    st.markdown(last.content)
                st.session_state.chat_history.append({
                    "role": role, "content": last.content
                })
                if node_name == "critic":
                    st.session_state.awaiting_feedback = True

# ─── Input Area ───────────────────────────────────────────────────────────────
if not st.session_state.awaiting_feedback:
    if query := st.chat_input("Enter research topic..."):
        # ── auto-ingest if files are attached ──────────────────────────────
        maybe_ingest(uploaded_files)

        # ── show user message ───────────────────────────────────────────────
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(f"**🧑 You**")
            st.markdown(query)

        # ── run the graph ───────────────────────────────────────────────────
        with st.spinner("🔍 Researching..."):
            run_stream({
                "messages": [HumanMessage(content=query)],
                "docs_ingested": st.session_state.ingested,
                "tool_calls_count": 0,
            })

        st.rerun()

else:
    # ── Feedback mode ───────────────────────────────────────────────────────
    st.info("📝 The critic has reviewed the report. Give feedback to revise, or click **Done**.")

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("✅ Done", use_container_width=True):
            st.session_state.awaiting_feedback = False
            st.rerun()

    if feedback := st.chat_input("Feedback to revise the report..."):
        st.session_state.chat_history.append({
            "role": "user",
            "content": f"📝 Feedback: {feedback}"
        })
        with st.spinner("✍️ Revising..."):
            run_stream({"feedback": feedback})
        st.rerun()