FROM nvcr.io/nvidia/tensorflow:24.07-tf2-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Targeted copies to avoid ingesting sensitive or unrelated files
COPY main.py convert_to_tflite.py test.py config.yaml /workspace/src/efficientnet_lite_gpu/
COPY tools/ /workspace/src/efficientnet_lite_gpu/tools/
COPY train/ /workspace/src/efficientnet_lite_gpu/train/
COPY validation/ /workspace/src/efficientnet_lite_gpu/validation/
COPY test/ /workspace/src/efficientnet_lite_gpu/test/

ENV PYTHONPATH=/workspace/src

WORKDIR /workspace/src/efficientnet_lite_gpu

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /workspace
USER appuser

CMD ["python", "main.py", "--action=train"]
