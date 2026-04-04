import asyncio
import json
import os
import tempfile
import logging
import redis.asyncio as redis
import httpx

from .inference import classify_image

logger = logging.getLogger("moderation")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MEDIA_SERVICE_URL = os.getenv("MEDIA_SERVICE_URL", "http://localhost:3012/media/v1")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "whispr-media")

async def download_from_s3(storage_path: str, dest_path: str):
    """Download file from MinIO/S3 to local temp file."""
    url = f"{MINIO_ENDPOINT}/{MINIO_BUCKET}/{storage_path}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(resp.content)

async def send_verdict(media_id: str, decision: str, score: float, category: str | None):
    """Send moderation verdict back to media-service."""
    async with httpx.AsyncClient() as client:
        await client.patch(
            f"{MEDIA_SERVICE_URL}/{media_id}/moderation",
            json={"status": decision, "score": score, "category": category},
            timeout=10.0,
        )
    logger.info(f"Verdict sent for {media_id}: {decision}")

async def process_message(data: dict):
    """Process a single media.uploaded event."""
    media_id = data.get("mediaId")
    storage_path = data.get("storagePath")

    if not media_id or not storage_path:
        logger.warning(f"Invalid event data: {data}")
        return

    logger.info(f"Processing media {media_id} at {storage_path}")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        try:
            await download_from_s3(storage_path, tmp.name)
            result = classify_image(tmp.name)
            await send_verdict(media_id, result["decision"], result["confidence"], result["category"])
        except Exception as e:
            logger.error(f"Moderation failed for {media_id}: {e}")
            # On error, approve to avoid blocking legitimate content
            await send_verdict(media_id, "approved", 0.0, None)

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
