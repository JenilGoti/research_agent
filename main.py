from langchain_core.messages import HumanMessage
from lg.graph import build_graph
from ingestion.pipeline import ingest_all

def run_cli(app):
    thread_id = "user-1"
    config = {"configurable": {"thread_id": thread_id}}

    print("\n🤖 AI Research Agent Ready (type 'exit' to quit)\n")

    user_input = input("Research topic: ").strip()

    if not user_input:
        return

    # ─── Initial research run ───────────────────────────────────────────
    print("\n🔍 Researching...\n")

    _stream(app, {"messages": [HumanMessage(content=user_input)]}, config)

    # ─── Feedback loop ──────────────────────────────────────────────────
    while True:
        feedback = input("\n📝 Feedback (or press Enter to continue): ").strip()

        if not feedback:
            break                                   # no feedback → done

        print("\n✍️  Revising report...\n")

        # inject feedback into same thread — graph resumes from critic
        _stream(app, {"feedback": feedback}, config)


def _stream(app, inputs: dict, config: dict):
    """Stream events and print the last message from each node."""
    events = app.stream(inputs, config=config)

    for event in events:
        for node_name, value in event.items():
            if "messages" not in value:
                continue

            msg = value["messages"][-1]
            if not msg.content:
                continue

            # label which node is speaking
            label = {
                "llm":    "🧠 Planner",
                "writer": "📄 Writer",
                "critic": "🔎 Critic",
                "tools":  "🛠  Tools",
            }.get(node_name, node_name)

            print(f"{label}:\n{msg.content}\n")


app = build_graph()

if __name__ == "__main__":
    print("📂 Ingesting documents...")
    print(ingest_all("data"))
    run_cli(app)