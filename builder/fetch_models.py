"""Pre-download WhisperX large-v2 model during Docker build."""
import whisperx
import os

MODEL_DIR = "/models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Downloading WhisperX large-v2 model...")
model = whisperx.load_model(
    "large-v2",
    device="cpu",
    compute_type="int8",
    download_root=MODEL_DIR
)
del model
print("large-v2 model downloaded.")

print("Skipping alignment model download (will be fetched on first run).")
print("All models downloaded successfully.")
