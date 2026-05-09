import asyncio
import io
import logging
import os
import secrets as pysecrets
import tempfile
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from PIL import Image, UnidentifiedImageError
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
# Whitelist de formats images supportes - SVG explicitement exclu (XSS / SSRF
# via uri externes embarques)
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}
# Feature flag pour gater l'auth interne pendant le rollout (le client
# media-service doit etre adapte pour envoyer le header avant de basculer
# le flag a true en preprod/prod)
REQUIRE_AUTH = os.getenv("MODERATION_REQUIRE_AUTH", "false").lower() == "true"
INTERNAL_SECRET = os.getenv("MODERATION_INTERNAL_SECRET")
INFERENCE_TIMEOUT_S = float(os.getenv("MOD_INFERENCE_TIMEOUT_S", "30.0"))


def verify_internal_secret(x_internal_secret: str | None = Header(default=None)):
    """Verifie le secret interne envoye par media-service.

    Gate par MODERATION_REQUIRE_AUTH pour ne pas casser le flux preprod tant
    que le client n'envoie pas le header. Quand le flag est true, le header
    est obligatoire et la comparaison se fait en temps constant.
    """
    if not REQUIRE_AUTH:
        return
    if not INTERNAL_SECRET:
        # config-error: refuser plutot que laisser passer en silence
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MODERATION_INTERNAL_SECRET non configure",
        )
    if not x_internal_secret or not pysecrets.compare_digest(
        x_internal_secret, INTERNAL_SECRET
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Internal secret invalide",
        )

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


@app.post(
    "/moderate/image",
    response_model=ModerationResult,
    dependencies=[Depends(verify_internal_secret)],
)
@limiter.limit(RATE_LIMIT)
async def moderate_image(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB}MB limit")

    # Le content-type seul est trompable (un attaquant peut envoyer un
    # script avec content-type image/png). On verifie les magic bytes via
    # PIL qui refuse aussi les payloads malformes.
    try:
        with Image.open(io.BytesIO(content)) as probe:
            probe.verify()
            img_format = probe.format
    except (UnidentifiedImageError, Exception) as exc:
        logger.warning("Image upload refused (verify failed): %s", exc)
        raise HTTPException(415, "Fichier image invalide")
    if img_format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(415, f"Format {img_format} non supporte")

    loop = asyncio.get_event_loop()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        # classify_image est CPU-bound : on l'execute dans le thread pool
        # pour ne pas bloquer l'event loop pendant l'inference ML.
        # Le timeout protege contre une image piege qui ferait tourner le
        # detecteur trop longtemps (DoS).
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, classify_image, tmp.name),
                timeout=INFERENCE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("Inference timeout after %.1fs", INFERENCE_TIMEOUT_S)
            raise HTTPException(503, "Inference timeout")

    return ModerationResult(**result)


@app.get("/")
async def root():
    return {
        "service": "whispr-moderation",
        "version": "1.1.0",
        "docs": "/docs",
        "threshold": BLOCK_THRESHOLD,
    }
