from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import MODEL_NAME
from app.agents.math_agent import create_math_agent
from app.agents.news_agent import create_news_agent
from app.agents.price_agent import create_price_agent
from app.agents.theory_agent import create_theory_agent

COORDINATOR_SYSTEM_PROMPT = """\
You are the lead financial research coordinator.
You manage a team of four specialist agents. Delegate work to them and synthesize their findings.

Your specialists:
- math_analyst: Performs quantitative calculations (DCF, ratios, CAGR, etc.)
- news_searcher: Finds and summarizes recent news and events
- price_analyst: Fetches real-time price data, risk metrics, and company fundamentals
- economics_theorist: Provides economic theory and framework analysis using a research knowledge base

Workflow:
1. Analyze the user's question to determine which specialists to consult.
2. Delegate specific sub-tasks to the appropriate agents.
3. Collect and cross-reference their findings.
4. Synthesize everything into a coherent answer.

Always structure your final answer as:
1. Thesis
2. Bull case
3. Bear case
4. Risk flags
5. Final view (Buy / Hold / Avoid)
6. Disclaimer

Important:
- Always consult at least the price_analyst for any ticker-specific question.
- Do not give personalized financial advice.
- Do not claim certainty about future returns.
"""


def _extract_agent_response(result: dict) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and getattr(msg, "type", "") == "ai":
            return content if isinstance(content, str) else str(content)
    return "No response generated."


def _make_agent_tools(math_agent, news_agent, price_agent, theory_agent):
    @tool
    async def math_analyst(request: str) -> str:
        """Quantitative financial analyst that performs calculations:
        DCF valuations, financial ratios (P/E, P/B, EV/EBITDA), CAGR,
        and arbitrary math expressions. Send a natural-language request
        describing what to calculate and with what inputs."""
        result = await math_agent.ainvoke(
            {"messages": [HumanMessage(content=request)]}
        )
        return _extract_agent_response(result)

    @tool
    async def news_searcher(request: str) -> str:
        """Financial news analyst that searches the web for recent news,
        events, earnings reports, and market developments.
        Send a query describing what news to find."""
        result = await news_agent.ainvoke(
            {"messages": [HumanMessage(content=request)]}
        )
        return _extract_agent_response(result)

    @tool
    async def price_analyst(request: str) -> str:
        """Market data specialist that fetches real-time stock prices,
        risk flags (drawdown, volatility), and company snapshots
        (sector, P/E, revenue growth) from Yahoo Finance.
        Send a request with the ticker and what data you need."""
        result = await price_agent.ainvoke(
            {"messages": [HumanMessage(content=request)]}
        )
        return _extract_agent_response(result)

    @tool
    async def economics_theorist(request: str) -> str:
        """Economist and financial theorist with access to a research
        knowledge base. Analyzes situations through economic frameworks:
        efficient markets, behavioral finance, monetary policy, credit cycles.
        Send a question about the economic/theoretical angle."""
        result = await theory_agent.ainvoke(
            {"messages": [HumanMessage(content=request)]}
        )
        return _extract_agent_response(result)

    return [math_analyst, news_searcher, price_analyst, economics_theorist]


def build_multi_agent():
    math_agent = create_math_agent()
    news_agent = create_news_agent()
    price_agent = create_price_agent()
    theory_agent = create_theory_agent()

    agent_tools = _make_agent_tools(math_agent, news_agent, price_agent, theory_agent)

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    return create_react_agent(llm, agent_tools, prompt=COORDINATOR_SYSTEM_PROMPT)
