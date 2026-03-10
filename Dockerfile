FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod==1.7.7 \
    git+https://github.com/m-bain/whisperx.git \
    matplotlib \
    requests

RUN pip install --no-cache-dir --force-reinstall --no-deps \
    ctranslate2==4.5.0

RUN pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN python -c "\
import torch, os, requests; \
model_dir = torch.hub._get_torch_home(); \
os.makedirs(model_dir, exist_ok=True); \
model_fp = os.path.join(model_dir, 'whisperx-vad-segmentation.bin'); \
print(f'Downloading VAD model to {model_fp}'); \
r = requests.get('https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin', allow_redirects=True, timeout=300); \
r.raise_for_status(); \
open(model_fp, 'wb').write(r.content); \
print(f'VAD model saved: {len(r.content)} bytes')"

COPY src/rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
