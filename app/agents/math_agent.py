import math

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import MODEL_NAME

MATH_SYSTEM_PROMPT = (
    "You are a quantitative financial analyst. "
    "Use tools to perform precise calculations. "
    "Never estimate when you can compute. "
    "Return results with clear labels and units."
)


@tool
def calculate_expression(expression: str) -> str:
    """Safely evaluate a mathematical expression and return the result.

    Supports basic arithmetic (+, -, *, /, **), math functions (sqrt, log, exp, etc.),
    and constants (pi, e). Example: '(100 * 1.05**10) - 100'
    """
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@tool
def compute_financial_ratios(
    price: float,
    earnings_per_share: float = 0.0,
    book_value_per_share: float = 0.0,
    enterprise_value: float = 0.0,
    ebitda: float = 0.0,
    total_debt: float = 0.0,
    total_equity: float = 0.0,
) -> dict:
    """Compute common financial ratios from provided data.

    Supply whichever inputs are available; ratios that lack data will be null.
    """
    ratios: dict = {}
    if earnings_per_share:
        ratios["pe_ratio"] = round(price / earnings_per_share, 2)
    if book_value_per_share:
        ratios["pb_ratio"] = round(price / book_value_per_share, 2)
    if enterprise_value and ebitda:
        ratios["ev_ebitda"] = round(enterprise_value / ebitda, 2)
    if total_equity:
        ratios["debt_to_equity"] = round(total_debt / total_equity, 2)
    return ratios or {"note": "Insufficient data to compute any ratio."}


@tool
def dcf_valuation(
    free_cash_flow: float,
    growth_rate: float,
    discount_rate: float,
    projection_years: int = 10,
    terminal_growth_rate: float = 0.02,
) -> dict:
    """Run a simple Discounted Cash Flow valuation.

    Args:
        free_cash_flow: Current annual free cash flow in dollars.
        growth_rate: Expected annual FCF growth rate (e.g. 0.08 for 8%).
        discount_rate: WACC or required return (e.g. 0.10 for 10%).
        projection_years: Number of years to project (default 10).
        terminal_growth_rate: Long-term growth rate for terminal value (default 2%).
    """
    if discount_rate <= terminal_growth_rate:
        return {"error": "Discount rate must exceed terminal growth rate."}

    projected = []
    pv_sum = 0.0
    fcf = free_cash_flow
    for year in range(1, projection_years + 1):
        fcf *= 1 + growth_rate
        pv = fcf / (1 + discount_rate) ** year
        pv_sum += pv
        projected.append({"year": year, "fcf": round(fcf, 2), "pv": round(pv, 2)})

    terminal_value = (
        fcf * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    )
    pv_terminal = terminal_value / (1 + discount_rate) ** projection_years

    return {
        "projected_fcfs": projected,
        "terminal_value": round(terminal_value, 2),
        "pv_of_terminal": round(pv_terminal, 2),
        "pv_of_fcfs": round(pv_sum, 2),
        "intrinsic_value": round(pv_sum + pv_terminal, 2),
    }


@tool
def compound_growth(start_value: float, end_value: float, years: float) -> dict:
    """Calculate Compound Annual Growth Rate (CAGR).

    Args:
        start_value: Beginning value.
        end_value: Ending value.
        years: Number of years between start and end.
    """
    if start_value <= 0 or years <= 0:
        return {"error": "start_value and years must be positive."}
    cagr = (end_value / start_value) ** (1 / years) - 1
    return {
        "cagr_pct": round(cagr * 100, 4),
        "start_value": start_value,
        "end_value": end_value,
        "years": years,
    }


MATH_TOOLS = [calculate_expression, compute_financial_ratios, dcf_valuation, compound_growth]


def create_math_agent():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    return create_react_agent(llm, MATH_TOOLS, prompt=MATH_SYSTEM_PROMPT)
