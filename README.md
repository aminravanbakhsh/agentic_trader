# agentic_trader

An agentic finance analyst built with LangChain, MCP, RAG, FastAPI, AWS, Docker, and GitHub Actions.

## Features
- MCP-based finance tools
- LangChain tool-calling agent
- RAG with FAISS
- FastAPI inference endpoint
- Docker packaging
- GitHub Actions CI/CD
- AWS deployment

## Quickstart

### 1. Create conda environment
```bash
conda env create -f environment.yml
conda activate agentic_trader




2. Configure environment variables
cp .env.example .env


3 .build the vector index
python scripts/ingest.py

4. run the API loccally
uvicorn app.main:app --reload

5. health check
curl http://127.0.0.1:8000/health


6. analyze endpoint
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker":"NVDA","question":"Should I buy this week?"}'

7. test
pytest

