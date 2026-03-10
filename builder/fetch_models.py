"""Pre-download faster-whisper large-v2 model during Docker build."""
import os

MODEL_DIR = "/models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Downloading faster-whisper large-v2 model via huggingface_hub...")
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Systran/faster-whisper-large-v2",
    local_dir=os.path.join(MODEL_DIR, "faster-whisper-large-v2"),
    local_dir_use_symlinks=False,
)
print("Model downloaded successfully.")
