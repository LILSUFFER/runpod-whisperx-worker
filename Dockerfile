FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        ca-certificates \
        curl \
        git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=mwader/static-ffmpeg:7.1.1 /ffmpeg /ffprobe /usr/local/bin/
RUN chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY builder/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Systran/faster-whisper-large-v2', local_dir='/models/Systran/faster-whisper-large-v2'); \
print('faster-whisper-large-v2 downloaded')"

COPY src/rp_handler.py /app/rp_handler.py

CMD ["python", "-u", "/app/rp_handler.py"]
