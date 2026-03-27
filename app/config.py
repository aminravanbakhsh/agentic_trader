import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
APP_NAME = os.getenv("APP_NAME", "agentic_trader")