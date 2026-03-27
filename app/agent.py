from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.config import MODEL_NAME
from app.rag import get_retriever

SYSTEM_PROMPT = """
You are a conservative finance research assistant.
Always use the available tools before answering.
Do not give personalized financial advice.
Do not claim certainty about future returns.
Base your answer on retrieved context and tool results.

Return your answer in this format:
1. Thesis
2. Bull case
3. Bear case
4. Risk flags
5. Final view (Buy / Hold / Avoid)
6. Disclaimer
"""


def _extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


class AgentRunner:
    def __init__(self, agent_graph):
        self.agent_graph = agent_graph

    async def ainvoke(self, payload):
        user_prompt = f"Ticker: {payload['ticker']}\nQuestion: {payload['question']}"
        result = await self.agent_graph.ainvoke(
            {"messages": [{"role": "user", "content": user_prompt}]}
        )

        messages = result.get("messages", [])
        if not messages:
            return {"output": "No answer generated"}

        last = messages[-1]
        content = getattr(last, "content", "")
        return {"output": _extract_text(content) or "No answer generated"}


async def build_agent():
    retriever = get_retriever()
    docs = retriever.invoke("valuation market risk company outlook macro trends")
    context_text = "\n\n".join(doc.page_content for doc in docs)

    client = MultiServerMCPClient(
        {
            "finance": {
                "command": "python",
                "args": ["mcp_server/server.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
        + f"\n\nRetrieved background context:\n{context_text}",
    )
    return AgentRunner(agent_graph)