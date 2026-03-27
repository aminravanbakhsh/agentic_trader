from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    ticker: str
    question: str


class AnalysisResponse(BaseModel):
    ticker: str
    answer: str