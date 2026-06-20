import sqlite3
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from graph import get_graph
from utils import State
import sqlite3

conn = sqlite3.connect('chat_history.db',check_same_thread=False)
sqlMemory = SqliteSaver(conn=conn)

load_dotenv()


graph =get_graph().compile(checkpointer=sqlMemory)

config = {"configurable": {"thread_id": "12355"}}

def chat(user_input: str,history):
    state = State(
        messages=[{"role": "user", "content": user_input}],
        response="",
        safety="unknown",
        unsafe_retries=0
    )
    result = graph.invoke(state, config=config)
    #print(result)
    return result["messages"][-1].content

if __name__ == "__main__":
    print("🤖 LangGraph Chatbot (type 'exit' to quit)")
    print("-" * 40)
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        response = chat(user_input,history=None)
        print(f"\nAssistant: {response}\n")