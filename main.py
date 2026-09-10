"""
Streamlit UI for the Planner -> Chat -> Tools LangGraph agent.
Shows a live reasoning trail: which node is running and which tool is called.
"""

import os
import sqlite3
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.tools import tool
from tavily import TavilyClient

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ---------------------------------------------------------------------------
# 1. GRAPH SETUP (same as notebook) — cached so it builds only once per session
# ---------------------------------------------------------------------------

@st.cache_resource
def build_workflow():

    connection = sqlite3.connect(database="context_memory.db", check_same_thread=False)
    sqlite_checkpointer = SqliteSaver(conn=connection)

    planner = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY2"))
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

    class GraphSchema(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        plan: str

    # ---- Tools ----
    @tool
    def calculator(first_num: float, second_num: float, operation: str) -> dict:
        """Perform a basic arithmetic operation on two numbers. Supported: add, sub, mul, div."""
        try:
            if operation == "add":
                result = first_num + second_num
            elif operation == "sub":
                result = first_num - second_num
            elif operation == "mul":
                result = first_num * second_num
            elif operation == "div":
                if second_num == 0:
                    return {"error": "Division by zero is not allowed"}
                result = first_num / second_num
            else:
                return {"error": f"Unsupported operation '{operation}'"}
            return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
        except Exception as e:
            return {"error": str(e)}

    @tool
    def get_stock_price(symbol: str) -> dict:
        """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') via Alpha Vantage."""
        api_key = os.environ["ALPHA_VANTAGE_API_KEY"]
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"get_stock_price failed: {e}"}

    @tool
    def get_weather(location: str) -> dict:
        """Fetch the current weather for a given city using the weather API."""
        api_key = os.environ["WEATHER_API_KEY"]
        url = "https://api.weatherapi.com/v1/current.json"
        params = {"key": api_key, "q": location, "aqi": "no"}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"get_weather failed: {e}"}

    @tool
    def web_search(query: str) -> Dict:
        """Perform a web search for the given query and return the relevant web results."""
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        try:
            client = TavilyClient(tavily_api_key)
            return client.search(query=query, search_depth="advanced", chunks_per_source=5, max_results=5)
        except Exception as e:
            return {"error": f"web_search failed: {e}"}

    tools = [web_search, get_stock_price, get_weather, calculator]
    main_llm = llm.bind_tools(tools)

    # ---- Nodes ----
    def safe_trim(messages, max_turns: int = 5):
        """
        Keep the last `max_turns` complete turns (from one HumanMessage up
        to, but not including, the next), so a tool_call/tool_response pair
        is never split.

        NOTE: an earlier version used langchain_core.trim_messages(start_on=
        "human", end_on=("human","tool")). That can return an EMPTY list
        when no valid human-message boundary falls inside a small
        max_tokens window — an empty message list then hits Gemini as
        "contents are required" and crashes the run mid-conversation.
        This version can never return empty as long as there's at least
        one HumanMessage anywhere in the history.
        """
        if not messages:
            return messages

        turn_starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]

        if not turn_starts:
            # No human turn found at all (shouldn't normally happen) —
            # don't drop everything, just pass history through unchanged.
            return messages

        if len(turn_starts) <= max_turns:
            return messages

        start_index = turn_starts[-max_turns]
        trimmed = messages[start_index:]
        return trimmed if trimmed else messages

    def Plan(state: GraphSchema) -> GraphSchema:
        messages = safe_trim(state["messages"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are the Planner for a multi-step problem-solving agent.

Use the conversation history and latest user message to understand
the user's task and create an execution plan.

Available tools:
- web_search
- get_stock_price
- get_weather
- calculator

Do not answer the question.
Only create the plan.

Output:

Goal: <one sentence>
Plan:
1. [tool: <tool_name or "none">] <what to do>
2. [tool: <tool_name or "none">] <what to do>
"""),
            MessagesPlaceholder("messages")
        ])

        chain = prompt | planner
        response = chain.invoke({"messages": messages})

        # Gemini can return content as a list of parts, e.g.
        # [{'type': 'text', 'text': '...', 'extras': {...}}] instead of
        # a plain string — normalize it so `plan` is clean text, not the
        # raw structure (extract_text is defined below; safe since this
        # only runs when the graph executes, after the whole module loads)
        return {"plan": extract_text(response.content)}

    def Chat(state: GraphSchema) -> GraphSchema:
        messages = safe_trim(state["messages"])
        plan = state.get("plan", "")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are the main problem-solving agent.

You have been given a plan by a Planner agent. Execute it step by step,
using the available tools whenever the plan calls for them:

- web_search
- get_stock_price
- get_weather
- calculator

Plan to follow:
{plan}

CRITICAL RULES:
- If a step needs a tool, you MUST actually call that tool's function.
  Never describe, narrate, or say you are "going to" call a tool instead
  of calling it — a sentence like "Step 1: retrieve the temperature using
  get_weather" without an actual function call is NOT allowed.
- Do not write any explanation before a needed tool call. Call the tool
  first, silently. Only write explanatory text once you have real tool
  results to explain.
- Only give a final text answer once every tool-requiring step in the
  plan has actually been executed and you have real results to work with.

Use the conversation history for context.
"""),
            MessagesPlaceholder("messages")
        ])

        chain = prompt | main_llm
        response = chain.invoke({"plan": plan, "messages": messages})
        return {"messages": response}

    tool_node = ToolNode(tools)

    graph = StateGraph(state_schema=GraphSchema)
    graph.add_node("planner", Plan)
    graph.add_node("chat_with_llm", Chat)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "chat_with_llm")
    graph.add_conditional_edges("chat_with_llm", tools_condition)
    graph.add_edge("tools", "chat_with_llm")

    return graph.compile(checkpointer=sqlite_checkpointer)


# ---------------------------------------------------------------------------
# UI setup FIRST — must run before build_workflow(), otherwise the browser
# stays blank while the graph/model clients initialize, since Streamlit
# sends nothing to the page until the first st.* call executes.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Step AI Agent", layout="centered")
st.title("🧭 Multi-Step Problem-Solving Agent")
st.caption("Planner → LLM reasoning → Tool calls, shown live.")

# `st.chat_input` is pinned to the bottom of the viewport, which can visually
# cover the last line(s) of a long response. Add bottom padding to the main
# content block so nothing ever renders underneath it.
st.markdown(
    """
    <style>
        .block-container {
            padding-bottom: 6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Setting up agent (models, tools, memory)..."):
    workflow = build_workflow()


# ---------------------------------------------------------------------------
# 2. Helpers to turn raw node output into readable reasoning lines
# ---------------------------------------------------------------------------

def extract_text(content):
    """Gemini wraps text in a list of parts: [{'type': 'text', 'text': ...}]."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


def describe_step(node_name: str, node_output: dict):
    """Convert one node's raw output into a short reasoning-trail string."""

    if node_name == "planner":
        return "planner", f"**Planner — building the plan:**\n\n{node_output['plan']}"

    if node_name == "chat_with_llm":
        msg = node_output["messages"]
        ai_msg = msg[-1] if isinstance(msg, list) else msg

        if getattr(ai_msg, "tool_calls", None):
            calls = []
            for tc in ai_msg.tool_calls:
                # render args as name=value instead of a raw {'k': 'v'} dict,
                # which reads as noisy symbols in the UI
                args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                calls.append(f"`{tc['name']}({args_str})`")
            calls_str = ", ".join(calls)
            return "tool_call", f"**LLM decided to call:** {calls_str}"
        else:
            text = extract_text(ai_msg.content)
            return "final", text

    if node_name == "tools":
        msg = node_output["messages"]
        results = msg if isinstance(msg, list) else [msg]
        lines = [f"**Tool result ({m.name}):**\n```\n{m.content}\n```" for m in results]
        return "tool_result", "\n\n".join(lines)

    return "other", f"{node_name}: {node_output}"


# ---------------------------------------------------------------------------
# 3. Rest of the UI
# ---------------------------------------------------------------------------

# thread_id: keep one per browser session so memory persists across turns
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session-1"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text) for display only

config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 25}

# render past turns
for role, text in st.session_state.chat_history:
    st.chat_message(role).markdown(text)

user_input = st.chat_input("Ask something...")

if user_input is not None and not user_input.strip():
    # chat_input can submit a string of only spaces — block that before
    # it ever reaches the graph.
    st.warning("Please provide an input.")

elif user_input:
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        reasoning_box = st.status("Planning...", expanded=True)
        final_answer = None

        try:
            for chunk in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    kind, text = describe_step(node_name, node_output)

                    if kind == "planner":
                        reasoning_box.update(label="Planner created a plan")
                        reasoning_box.markdown(text)
                    elif kind == "tool_call":
                        reasoning_box.update(label="Calling tool(s)...")
                        reasoning_box.markdown(text)
                    elif kind == "tool_result":
                        reasoning_box.update(label="Tool returned a result")
                        reasoning_box.markdown(text)
                    elif kind == "final":
                        final_answer = text

            reasoning_box.update(label="Done", state="complete", expanded=False)

        except Exception as e:
            reasoning_box.update(label="Error", state="error")
            st.error(f"Something went wrong: {e}")
            final_answer = None

        if final_answer:
            st.markdown(final_answer)
            st.session_state.chat_history.append(("assistant", final_answer))