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
    sv.assert_called_once()
