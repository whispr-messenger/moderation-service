import asyncio
import json
import os
import re
import tempfile
import logging
import redis.asyncio as redis
import httpx
import boto3
from botocore.client import Config as BotoConfig

from .inference import classify_image

logger = logging.getLogger("moderation")

# Allowlist pour le storage_path recu de Redis : caracteres safe pour S3 keys,
# pas de "..", pas de scheme, pas de slash double. La regex bloque aussi les
# tentatives d'exfiltration via des chemins absolus ou exotiques.
SAFE_STORAGE_PATH = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9/_.-]{0,511}$")


def is_safe_storage_path(path: str) -> bool:
    """Verifie qu'un storage_path est safe a passer a S3."""
    if not path or not SAFE_STORAGE_PATH.match(path):
        return False
    # ne pas oublier que la regex laisse passer "a/../b" si on n'exclut pas
    # explicitement la sequence ".."
    if ".." in path:
        return False
    return True

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MEDIA_SERVICE_URL = os.getenv("MEDIA_SERVICE_URL", "http://localhost:3012/media/v1")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "whispr-media")
# Meme timeout que la route HTTP /moderate/image : protege contre une image
# piege qui ferait crasher ou tourner NudeNet trop longtemps (DoS / fail-open).
INFERENCE_TIMEOUT_S = float(os.getenv("MOD_INFERENCE_TIMEOUT_S", "30.0"))

_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name='us-east-1',
            config=BotoConfig(signature_version='s3v4', s3={'addressing_style': 'path'}),
        )
    return _s3_client

async def download_from_s3(storage_path: str, dest_path: str):
    """Download file from MinIO/S3 to local temp file using authenticated S3 GET."""
    s3 = get_s3_client()
    # boto3 is sync, run in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: s3.download_file(MINIO_BUCKET, storage_path, dest_path),
    )

async def send_verdict(media_id: str, decision: str, score: float, category: str | None):
    """Send moderation verdict back to media-service."""
    async with httpx.AsyncClient() as client:
        resp = await client.patch(
            f"{MEDIA_SERVICE_URL}/{media_id}/moderation",
            json={"status": decision, "score": score, "category": category},
            timeout=10.0,
        )
        resp.raise_for_status()
    logger.info("Verdict sent for %s: %s", media_id, decision)

async def process_message(data: dict):
    """Process a single media.uploaded event."""
    media_id = data.get("mediaId")
    storage_path = data.get("storagePath")

    if not media_id or not storage_path:
        logger.warning(f"Invalid event data: {data}")
        return

    if not is_safe_storage_path(storage_path):
        logger.error(
            "Rejected unsafe storage_path media_id=%s path=%r",
            media_id, storage_path,
        )
        return

    logger.info(f"Processing media {media_id} at {storage_path}")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        try:
            await download_from_s3(storage_path, tmp.name)
            # classify_image est CPU-bound : on l'execute dans le thread pool
            # pour ne pas bloquer l'event loop pendant l'inference ML.
            # Le timeout protege contre une image piege qui ferait tourner le
            # detecteur trop longtemps (DoS).
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, classify_image, tmp.name),
                timeout=INFERENCE_TIMEOUT_S,
            )
            await send_verdict(media_id, result["decision"], result["confidence"], result["category"])
        except asyncio.TimeoutError:
            logger.error(
                "Inference timeout for media_id=%s after %.1fs, marking pending",
                media_id, INFERENCE_TIMEOUT_S,
            )
            # ne pas oublier : auto-approuver sur erreur = bypass modération.
            # On marque pending pour re-classification ou intervention humaine.
            await send_verdict(media_id, "pending", 0.0, None)
        except Exception as e:
            logger.error(f"Moderation failed for {media_id}: {e}")
            await send_verdict(media_id, "pending", 0.0, None)

async def listen():
    """Listen for media.uploaded events on Redis pub/sub."""
    logger.info("Starting Redis consumer...")
    r = redis.from_url(REDIS_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe("media.uploaded")

    logger.info("Subscribed to media.uploaded channel")

    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                data = json.loads(message["data"])
                await process_message(data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in event: {message['data']}")
            except Exception as e:
                logger.error(f"Error processing event: {e}")
