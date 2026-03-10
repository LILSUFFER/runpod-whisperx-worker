FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir \
    torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    runpod==1.7.7 \
    whisperx==3.3.1 \
    matplotlib \
    requests

RUN pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

COPY src/rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
