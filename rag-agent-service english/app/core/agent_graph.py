"""
Agentic workflow core (skill keyword 2: LangGraph workflow).

Flow design:
    User question
        │
        ▼
   [agent node] ── LLM decides: call a tool or not?
        │                         │
        │ no tool needed          │ tool needed
        ▼                         ▼
   [guardrail node]          [tools node] ── runs search_knowledge_base / search_web
        │                         │
        │                         └──────► back to [agent node] (feeds the tool result back in,
        ▼                                    retrying up to MAX_AGENT_STEPS times)
      Done, return the answer + citations

This maps to the "intent parsing → tool selection → information retrieval → answer generation →
quality validation" pipeline commonly described in job postings.
The guardrail node is a simplified Human-in-the-Loop concept:
when the answer looks unsupported (e.g. the knowledge base should have had an answer but didn't),
it's flagged as needs_human_review. In a production system this would hook into LangGraph's
interrupt(), pausing the flow to wait for human review before proceeding.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.core.tools import ALL_TOOLS
from app.llm.llm_factory import get_llm_provider

SYSTEM_PROMPT = """你是一個負責任的 AI 助理。
規則：
1. 如果問題可能跟使用者上傳的文件有關，優先使用 search_knowledge_base 工具查詢。
2. 如果知識庫查不到，或問題明顯需要最新/公開資訊，才使用 search_web 工具。
3. 回答時務必根據工具回傳的內容作答，禁止憑空捏造；如果工具也查不到，要老實說「不知道」。
4. 如果引用了 search_knowledge_base 的內容，回答最後要附上來源檔名與頁碼。
"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    needs_human_review: bool


def _build_llm():
    llm = get_llm_provider().get_chat_model()
    return llm.bind_tools(ALL_TOOLS)


def _agent_node(state: AgentState) -> dict:
    llm = _build_llm()
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}


def _guardrail_node(state: AgentState) -> dict:
    """Simplified quality validation: checks whether the final answer is empty,
    or whether the knowledge base lookup came up short."""
    last_message = state["messages"][-1]
    answer_text = getattr(last_message, "content", "") or ""

    suspicious_markers = ["找不到相關內容", "沒有找到相關結果", "我不知道", "無法回答"]
    needs_review = (not answer_text.strip()) or any(m in answer_text for m in suspicious_markers)

    return {"needs_human_review": needs_review}


def _should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "tools"
    return "guardrail"


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("guardrail", _guardrail_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "guardrail": "guardrail"})
    graph.add_edge("tools", "agent")
    graph.add_edge("guardrail", END)

    return graph.compile()


# Singleton, to avoid recompiling the graph on every request
agent_app = build_agent_graph()


def run_agent(query: str) -> dict:
    """Main public entry point: pass in a user question, get back the final state."""
    result = agent_app.invoke(
        {"messages": [HumanMessage(content=query)], "needs_human_review": False},
        config={"recursion_limit": settings.MAX_AGENT_STEPS * 2},
    )
    final_answer = result["messages"][-1].content
    return {
        "answer": final_answer,
        "needs_human_review": result["needs_human_review"],
        "step_count": len(result["messages"]),
    }
