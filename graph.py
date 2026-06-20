from utils import State
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from tools import get_tools

load_dotenv()

llm            = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(get_tools())
evaluator_llm  = ChatGroq(model="llama-3.3-70b-versatile")

MAX_UNSAFE_RETRIES = 3

# ================== LLM Node ==================
def chatllm(state: State) -> dict: 
    response = llm_with_tools.invoke(state.messages)
    print('-------------------')
    print("LLM Response:", response)
    print('-------------------')
    return {
        "messages":       [response],
        "response":       response.content,
        "safety":         "unknown",
        "unsafe_retries": state.unsafe_retries,
    }

# ================== Evaluator helper ==================
def evaluator(state: State) -> str:
    last_msg   = state.messages[-1]
    content    = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    commands = []
    for tc in tool_calls:
        if tc.get("name") == "execute_list_of_commands":
            commands.extend(tc.get("args", {}).get("commands", []))

    check_text = "\n".join(commands) if commands else content

    prompt = f"""Evaluate whether this content is safe.

Content:
{check_text}

Reply ONLY with: safe
or: unsafe"""

    result = evaluator_llm.invoke(prompt)
    return result.content.strip().lower()

# ================== Evaluator Node ==================
def evaluator_node(state: State) -> dict:   # ✅ return dict
    verdict = evaluator(state)
    retries = state.unsafe_retries

    if "unsafe" in verdict:
        retries += 1

        if retries >= MAX_UNSAFE_RETRIES:
            print(f"🚫 Blocked after {MAX_UNSAFE_RETRIES} unsafe attempts.")
            return {
                "messages": [AIMessage(content=f"🚫 Blocked after {MAX_UNSAFE_RETRIES} unsafe attempts.")],
                "safety":         "blocked",
                "unsafe_retries": retries,
            }

        print(f"⚠️ Unsafe response. Retry {retries}/{MAX_UNSAFE_RETRIES}")
        return {
            # ✅ inject feedback so LLM knows to try differently
            "messages": [{"role": "user", "content": "Your previous answer was unsafe. Generate a safer response."}],
            "safety":         "unsafe",
            "unsafe_retries": retries,
        }

    return {
        "safety":         "safe",
        "unsafe_retries": retries,
    }

# ================== Router ==================
def route_after_evaluation(state: State) -> str:
    safety = state.safety

    if safety == "blocked":
        return END

    if safety == "unsafe":
        return "chatllm"

    # safe — check if last message has pending tool calls
    last_msg   = state.messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []
    return "tools" if tool_calls else END

# ================== Graph ==================
graph_builder = StateGraph(State)

graph_builder.add_node("chatllm",   chatllm)
graph_builder.add_node("evaluator", evaluator_node)
graph_builder.add_node("tools",     ToolNode(get_tools()))

graph_builder.add_edge(START, "chatllm")

"""# ✅ chatllm → evaluator always (evaluator decides safe→tools or safe→end or unsafe→retry)
graph_builder.add_conditional_edges(
    "chatllm",
    tools_condition,
    {"tools": "evaluator", END: "evaluator"}
)"""
graph_builder.add_edge("chatllm", "evaluator")
graph_builder.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {"chatllm": "chatllm", "tools": "tools", END: END}
)

graph_builder.add_edge("tools", "chatllm")

def get_graph():
    return graph_builder


graph = graph_builder.compile()
png_data = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)

print("saved")