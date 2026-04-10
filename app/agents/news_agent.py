from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import MODEL_NAME

NEWS_SYSTEM_PROMPT = (
    "You are a financial news analyst. "
    "Search for the latest news and events relevant to the query. "
    "Summarize findings with source attribution. "
    "Focus on material events: earnings, M&A, regulatory actions, macro shifts."
)


@tool
def search_news(query: str, max_results: int = 5) -> list[dict]:
    """Search for recent news articles using DuckDuckGo.

    Args:
        query: News search query (e.g. 'NVDA earnings 2024').
        max_results: Maximum number of results to return.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        return [
            {
                "title": r.get("title"),
                "body": r.get("body"),
                "url": r.get("url"),
                "date": r.get("date"),
            }
            for r in results
        ]
    except Exception as exc:
        return [{"error": f"News search failed: {exc}"}]


@tool
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for broader context using DuckDuckGo.

    Args:
        query: Web search query (e.g. 'NVDA competitive landscape AI chips').
        max_results: Maximum number of results to return.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title"),
                "body": r.get("body"),
                "href": r.get("href"),
            }
            for r in results
        ]
    except Exception as exc:
        return [{"error": f"Web search failed: {exc}"}]


NEWS_TOOLS = [search_news, search_web]


def create_news_agent():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    return create_react_agent(llm, NEWS_TOOLS, prompt=NEWS_SYSTEM_PROMPT)
