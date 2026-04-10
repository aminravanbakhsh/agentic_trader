import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from app.schemas import AnalysisRequest, AnalysisResponse
from app.agent import build_agent

app = FastAPI(title="agentic_trader")

_STATIC_DIR = Path(__file__).resolve().parent / "static"

AGENT_TOOL_NAMES = frozenset(
    {"math_analyst", "news_searcher", "price_analyst", "economics_theorist"}
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return (_STATIC_DIR / "index.html").read_text()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest):
    try:
        executor = await build_agent()
        result = await executor.ainvoke(
            {"ticker": req.ticker.upper(), "question": req.question}
        )
        output = result.get("output", "No answer generated")
        return AnalysisResponse(ticker=req.ticker.upper(), answer=output)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze/stream")
async def analyze_stream(req: AnalysisRequest):
    executor = await build_agent()

    async def event_generator():
        depth = 0
        try:
            async for event in executor.astream_events(
                {"ticker": req.ticker.upper(), "question": req.question}
            ):
                kind = event.get("event")

                if kind == "on_tool_start" and event.get("name") in AGENT_TOOL_NAMES:
                    depth += 1
                    yield _sse({"type": "status", "agent": event["name"]})

                elif kind == "on_tool_end" and event.get("name") in AGENT_TOOL_NAMES:
                    yield _sse({"type": "status_done", "agent": event["name"]})
                    depth -= 1

                elif kind == "on_chat_model_stream" and depth == 0:
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is not None:
                        text = getattr(chunk, "content", "")
                        if text:
                            yield _sse({"type": "token", "content": text})

            yield _sse({"type": "done"})
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
