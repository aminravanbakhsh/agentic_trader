import yfinance as yf
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import MODEL_NAME

PRICE_SYSTEM_PROMPT = (
    "You are a market data specialist. "
    "Use tools to fetch accurate price data, risk metrics, and company fundamentals. "
    "Always present numbers with appropriate precision and context."
)


@tool
def get_price_summary(ticker: str, period: str = "3mo") -> dict:
    """Return a price summary for a stock ticker including returns and volatility."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return {"error": f"No price data found for {ticker}"}

        close = hist["Close"]
        returns = close.pct_change().dropna()

        return {
            "ticker": ticker.upper(),
            "period": period,
            "last_close": round(float(close.iloc[-1]), 2),
            "period_return_pct": round(
                float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2
            ),
            "annualized_volatility_pct": round(
                float(returns.std() * (252**0.5) * 100), 2
            ),
            "high": round(float(close.max()), 2),
            "low": round(float(close.min()), 2),
        }
    except Exception as exc:
        return {"error": f"Failed to fetch price summary for {ticker}: {exc}"}


@tool
def get_risk_flags(ticker: str, period: str = "3mo") -> dict:
    """Return simple technical risk flags based on drawdown and volatility."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return {"error": f"No price data found for {ticker}"}

        close = hist["Close"]
        rolling_max = close.cummax()
        drawdown = (close / rolling_max - 1.0).min() * 100
        returns = close.pct_change().dropna()
        vol = returns.std() * (252**0.5) * 100

        flags = []
        if drawdown < -20:
            flags.append("deep_drawdown")
        if vol > 45:
            flags.append("high_volatility")
        if not flags:
            flags.append("no_major_technical_flag")

        return {
            "ticker": ticker.upper(),
            "max_drawdown_pct": round(float(drawdown), 2),
            "annualized_volatility_pct": round(float(vol), 2),
            "flags": flags,
        }
    except Exception as exc:
        return {"error": f"Failed to fetch risk flags for {ticker}: {exc}"}


@tool
def get_company_snapshot(ticker: str) -> dict:
    """Return a lightweight company snapshot from Yahoo Finance metadata."""
    try:
        info = yf.Ticker(ticker).info
        if not info:
            return {"error": f"No company info found for {ticker}"}

        keys = [
            "shortName",
            "sector",
            "industry",
            "marketCap",
            "trailingPE",
            "forwardPE",
            "revenueGrowth",
            "profitMargins",
        ]
        return {"ticker": ticker.upper(), **{k: info.get(k) for k in keys}}
    except Exception as exc:
        return {"error": f"Failed to fetch company snapshot for {ticker}: {exc}"}


PRICE_TOOLS = [get_price_summary, get_risk_flags, get_company_snapshot]


def create_price_agent():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    return create_react_agent(llm, PRICE_TOOLS, prompt=PRICE_SYSTEM_PROMPT)
