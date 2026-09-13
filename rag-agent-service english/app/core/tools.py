"""
Tools the agent can call (skill keywords 2 / 4: Tool Calling).

Each tool is an independent, single-responsibility function. The agent decides for
itself which one to call based on the question — this is the core design pattern
in LangGraph / LangChain agent development.
"""
from langchain_core.tools import tool

from app.core.rag_engine import rag_engine


@tool
def search_knowledge_base(query: str) -> str:
    """
    Query the internal documents already uploaded (PDFs / notes / etc.).
    Call this tool first when the question might relate to data the user uploaded.
    The returned content includes the source filename and page number, for citation.
    """
    chunks = rag_engine.retrieve(query)
    if not chunks:
        return "知識庫中找不到相關內容。"

    formatted = []
    for c in chunks:
        page_info = f"第 {c.page + 1} 頁" if c.page is not None else "頁碼未知"
        formatted.append(f"[來源: {c.source}, {page_info}, 相關度: {c.score:.2f}]\n{c.content}")
    return "\n\n---\n\n".join(formatted)


@tool
def search_web(query: str) -> str:
    """
    Call this tool to do a free web search when the knowledge base doesn't have the
    answer, or the question clearly needs current/public information.
    Uses DuckDuckGo, no API key or paid quota required.
    """
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if not results:
        return "網路搜尋沒有找到相關結果。"

    formatted = [f"[{r['title']}]({r['href']})\n{r['body']}" for r in results]
    return "\n\n---\n\n".join(formatted)


ALL_TOOLS = [search_knowledge_base, search_web]
