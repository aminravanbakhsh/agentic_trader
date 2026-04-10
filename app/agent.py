from app.agents.coordinator import build_multi_agent


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

    async def astream_events(self, payload):
        user_prompt = f"Ticker: {payload['ticker']}\nQuestion: {payload['question']}"
        async for event in self.agent_graph.astream_events(
            {"messages": [{"role": "user", "content": user_prompt}]},
            version="v2",
        ):
            yield event

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
    agent_graph = build_multi_agent()
    return AgentRunner(agent_graph)
