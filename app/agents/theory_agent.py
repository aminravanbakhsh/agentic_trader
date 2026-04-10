from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import MODEL_NAME
from app.rag import get_retriever

THEORY_SYSTEM_PROMPT = (
    "You are an economist and financial theorist. "
    "Use the retrieval tool to find relevant economic theory and research, "
    "then apply it to analyze the situation. "
    "Reference specific frameworks: efficient market hypothesis, behavioral finance, "
    "monetary policy transmission, credit cycles, sector rotation, etc. "
    "Ground your analysis in theory but connect it to practical implications."
)

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


@tool
def retrieve_economic_context(query: str) -> str:
    """Search the knowledge base for economic theory, research notes, and market analysis.

    Args:
        query: A descriptive query about economic concepts or market conditions
               (e.g. 'monetary policy impact on tech valuations').
    """
    try:
        retriever = _get_retriever()
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    except Exception as exc:
        return f"Retrieval failed: {exc}"


THEORY_TOOLS = [retrieve_economic_context]


def create_theory_agent():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    return create_react_agent(llm, THEORY_TOOLS, prompt=THEORY_SYSTEM_PROMPT)
