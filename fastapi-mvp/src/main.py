import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from .inference import classify_image, load_model
from .redis_consumer import listen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moderation")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    load_model()
    # Start Redis consumer in background
    task = asyncio.create_task(listen())
    yield
    task.cancel()

app = FastAPI(title="Whispr Moderation Service", version="1.0.0", lifespan=lifespan)

class ModerationResult(BaseModel):
    decision: str
    confidence: float
    category: str | None = None
    all_detections: int = 0

@app.get("/health")
async def health():
    return {"status": "ok", "service": "moderation"}

@app.post("/moderate/image", response_model=ModerationResult)
async def moderate_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        result = classify_image(tmp.name)

    return ModerationResult(**result)

@app.get("/")
async def root():
    return {"service": "whispr-moderation", "version": "1.0.0", "docs": "/docs"}
