FROM nvcr.io/nvidia/tensorflow:24.07-tf2-py3

# 2️⃣ 基本环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace/src/efficientnet_lite_gpu

ENV PYTHONPATH=/workspace/src

WORKDIR /workspace/src/efficientnet_lite_gpu

CMD ["python", "main.py", "--action=train"]
