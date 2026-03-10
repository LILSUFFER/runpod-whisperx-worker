FROM realyashnag/worker-whisperx:latest

RUN python -c "\
import requests, os, torch; \
model_dir = torch.hub._get_torch_home(); \
os.makedirs(model_dir, exist_ok=True); \
model_fp = os.path.join(model_dir, 'whisperx-vad-segmentation.bin'); \
r = requests.get('https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin', allow_redirects=True, timeout=300); \
r.raise_for_status(); \
open(model_fp, 'wb').write(r.content); \
print(f'VAD model downloaded: {len(r.content)} bytes to {model_fp}')"
