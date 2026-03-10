FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
    runpod==1.7.7 \
    whisperx==3.3.1 \
    matplotlib \
    requests

COPY src/rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
