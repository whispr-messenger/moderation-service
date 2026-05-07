FROM python:3.11-slim

WORKDIR /app

# Install system deps for ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/ src/

ENV PYTHONUNBUFFERED=1 \
    MOD_BLOCK_THRESHOLD=0.6 \
    MOD_MAX_UPLOAD_MB=10 \
    MOD_RATE_LIMIT=30/minute \
    HOME=/home/appuser

USER appuser

# Pre-download NudeNet model so startup is fast and reproducible
RUN python -c "from nudenet import NudeDetector; NudeDetector()"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
