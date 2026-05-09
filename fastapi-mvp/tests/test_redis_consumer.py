from unittest.mock import AsyncMock, patch

import pytest

from src import redis_consumer


@pytest.mark.parametrize("path", [
    "media/abc123.jpg",
    "user-uploads/2026/05/file_v2.png",
    "a.png",
    "x" * 512,
])
def test_is_safe_storage_path_accepts_valid(path):
    assert redis_consumer.is_safe_storage_path(path) is True


@pytest.mark.parametrize("path", [
    "",
    None,
    "../etc/passwd",
    "media/../../secrets/key",
    "/absolute/path",
    "media/file with space.jpg",
    "media/file;rm -rf.png",
    "s3://bucket/key",
    "http://evil.com/x.png",
    "x" * 513,
    "media/file\nname.png",
    "média/é.png",
])
def test_is_safe_storage_path_rejects_invalid(path):
    assert redis_consumer.is_safe_storage_path(path) is False


@pytest.mark.asyncio
async def test_process_message_rejects_unsafe_path():
    """process_message ne doit pas appeler download_from_s3 si path unsafe."""
    with patch.object(redis_consumer, "download_from_s3", new=AsyncMock()) as dl, \
         patch.object(redis_consumer, "send_verdict", new=AsyncMock()) as sv:
        await redis_consumer.process_message({
            "mediaId": "m1",
            "storagePath": "../../etc/passwd",
        })
    dl.assert_not_called()
    sv.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_accepts_safe_path():
    """Avec un path valide, le pipeline est invoque."""
    with patch.object(redis_consumer, "download_from_s3", new=AsyncMock()), \
         patch.object(redis_consumer, "send_verdict", new=AsyncMock()) as sv, \
         patch.object(redis_consumer, "classify_image", return_value={
             "decision": "approved", "confidence": 0.9, "category": None,
         }):
        await redis_consumer.process_message({
            "mediaId": "m2",
            "storagePath": "media/ok.jpg",
        })
    sv.assert_called_once_with("m2", "approved", 0.9, None)


@pytest.mark.asyncio
async def test_process_message_inference_error_marks_pending():
    """Si classify_image plante, on doit marquer pending PAS approved.
    Sinon un attaquant qui crashe l'inference bypass la moderation.
    """
    def boom(_path):
        raise RuntimeError("nudenet crashed on craft image")

    with patch.object(redis_consumer, "download_from_s3", new=AsyncMock()), \
         patch.object(redis_consumer, "send_verdict", new=AsyncMock()) as sv, \
         patch.object(redis_consumer, "classify_image", side_effect=boom):
        await redis_consumer.process_message({
            "mediaId": "m3",
            "storagePath": "media/crash.jpg",
        })

    sv.assert_called_once_with("m3", "pending", 0.0, None)


@pytest.mark.asyncio
async def test_process_message_inference_timeout_marks_pending():
    """Si l'inference depasse le timeout, on doit marquer pending."""
    def slow(_path):
        # bloque assez longtemps pour declencher asyncio.wait_for
        import time
        time.sleep(2.0)
        return {"decision": "approved", "confidence": 0.9, "category": None}

    with patch.object(redis_consumer, "download_from_s3", new=AsyncMock()), \
         patch.object(redis_consumer, "send_verdict", new=AsyncMock()) as sv, \
         patch.object(redis_consumer, "classify_image", side_effect=slow), \
         patch.object(redis_consumer, "INFERENCE_TIMEOUT_S", 0.1):
        await redis_consumer.process_message({
            "mediaId": "m4",
            "storagePath": "media/slow.jpg",
        })

    sv.assert_called_once_with("m4", "pending", 0.0, None)


@pytest.mark.asyncio
async def test_process_message_download_error_marks_pending():
    """Si le download S3 echoue, on doit marquer pending PAS approved."""
    async def boom_download(_path, _dest):
        raise IOError("s3 unreachable")

    with patch.object(redis_consumer, "download_from_s3", new=boom_download), \
         patch.object(redis_consumer, "send_verdict", new=AsyncMock()) as sv, \
         patch.object(redis_consumer, "classify_image"):
        await redis_consumer.process_message({
            "mediaId": "m5",
            "storagePath": "media/missing.jpg",
        })

    sv.assert_called_once_with("m5", "pending", 0.0, None)
