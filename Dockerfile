FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    openai-whisper \
    whisperx \
    torch torchaudio \
    requests

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
