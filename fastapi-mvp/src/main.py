import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .inference import BLOCK_THRESHOLD, classify_image, load_model
from .redis_consumer import listen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moderation")

MAX_UPLOAD_MB = int(os.getenv("MOD_MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
RATE_LIMIT = os.getenv("MOD_RATE_LIMIT", "30/minute")

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup (blocks readiness until done)
    load_model()
    app.state.ready = True
    # Start Redis consumer in background
    task = asyncio.create_task(listen())
    try:
        yield
    finally:
        app.state.ready = False
        task.cancel()


app = FastAPI(title="Whispr Moderation Service", version="1.1.0", lifespan=lifespan)
app.state.ready = False
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class ModerationResult(BaseModel):
    decision: str
    confidence: float
    category: str | None = None
    all_detections: int = 0
    latency_ms: float | None = None


@app.get("/health")
async def health():
    """Liveness probe: process is up."""
    return {"status": "ok", "service": "moderation"}


@app.get("/ready")
async def ready():
    """Readiness probe: model is loaded and app is accepting traffic."""
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready", "threshold": BLOCK_THRESHOLD}


@app.post("/moderate/image", response_model=ModerationResult)
@limiter.limit(RATE_LIMIT)
async def moderate_image(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB}MB limit")

    loop = asyncio.get_event_loop()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        # classify_image est CPU-bound : on l'exécute dans le thread pool
        # pour ne pas bloquer l'event loop pendant l'inférence ML
        result = await loop.run_in_executor(None, classify_image, tmp.name)

    return ModerationResult(**result)


@app.get("/")
async def root():
    return {
        "service": "whispr-moderation",
        "version": "1.1.0",
        "docs": "/docs",
        "threshold": BLOCK_THRESHOLD,
    }
