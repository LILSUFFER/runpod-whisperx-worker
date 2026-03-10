FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
WORKDIR /

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY builder/requirements.txt /builder/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /builder/requirements.txt && \
    pip install --no-cache-dir huggingface_hub

RUN python -c "from pathlib import Path; import whisperx; p = Path(whisperx.__file__).parent / 'assets' / 'pytorch_model.bin'; print(f'VAD model bundled: {p.exists()}')"

COPY builder/fetch_models.py /builder/fetch_models.py
RUN python /builder/fetch_models.py

COPY src/rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
