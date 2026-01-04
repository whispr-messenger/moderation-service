# TensorFlow avec GPU support (CUDA + cuDNN inclus)
FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Mise à jour des paquets et installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates wget \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python", "src/efficientnet_lite_gpu/main.py", "--action=train"]
