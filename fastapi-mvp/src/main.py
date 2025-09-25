from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="FastAPI MVP", version="0.1.0")

class ModerationResult(BaseModel):
    decision: str
    reason: str
    confidence: float
    phash: str | None = None

@app.get("/")
async def root():
    return {"message": "FastAPI MVP up", "docs": "/docs"}