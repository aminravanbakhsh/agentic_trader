from fastapi import FastAPI, HTTPException
from app.schemas import AnalysisRequest, AnalysisResponse
from app.agent import build_agent

app = FastAPI(title="agentic_trader")


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