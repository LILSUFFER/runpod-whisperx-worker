FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
WORKDIR /

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY builder/requirements.txt /builder/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /builder/requirements.txt

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('guillaumekln/faster-whisper-large-v2', local_dir='/models/faster-whisper-large-v2'); \
print('Model downloaded successfully')"

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('jonatasgrosman/wav2vec2-large-xlsr-53-russian', local_dir='/models/wav2vec2-large-xlsr-53-russian'); \
print('Alignment model downloaded successfully')"

COPY src/rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
