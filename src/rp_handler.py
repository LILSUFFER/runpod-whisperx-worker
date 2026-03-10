"""RunPod serverless handler for faster-whisper transcription with word timestamps."""
import sys
print(f"[INIT] Python {sys.version}", flush=True)

import gc
import os
import tempfile
import traceback

import torch
print(f"[INIT] PyTorch {torch.__version__}", flush=True)

import requests
import runpod
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.audio import decode_audio
print("[INIT] faster-whisper imported successfully", flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
MODEL_NAME = os.environ.get("WHISPER_MODEL", "Systran/faster-whisper-large-v2")
MODEL_DIR = "/models"

print(f"[INIT] Device: {DEVICE}, compute: {COMPUTE_TYPE}, batch: {BATCH_SIZE}", flush=True)
print(f"[INIT] Model: {MODEL_NAME}", flush=True)

model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=MODEL_DIR,
)
batched_model = BatchedInferencePipeline(model)
print("[INIT] Model loaded successfully!", flush=True)


def download_audio(url):
    suffix = ".wav" if ".wav" in url else ".mp3" if ".mp3" in url else ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    print(f"Downloading audio from {url}...", flush=True)

    headers = {}
    auth_header = os.environ.get("AUDIO_AUTH_HEADER", "")
    if auth_header:
        headers["Authorization"] = auth_header

    resp = requests.get(url, headers=headers, timeout=600, stream=True, allow_redirects=True)
    resp.raise_for_status()
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if chunk:
            tmp.write(chunk)
    tmp.close()
    size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f}MB to {tmp.name}", flush=True)
    return tmp.name


def handler(job):
    inp = job["input"]
    audio_url = inp.get("audio", "")
    audio_base64 = inp.get("audio_base64", "")
    language = inp.get("language", "ru")
    batch_size = inp.get("batch_size", BATCH_SIZE)

    if not audio_url and not audio_base64:
        return {"error": "No audio provided. Use 'audio' (URL) or 'audio_base64'."}

    tmp_path = None
    try:
        if audio_url:
            tmp_path = download_audio(audio_url)
        else:
            import base64
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))

        audio = decode_audio(tmp_path)

        result_segments = None
        for attempt_batch in [batch_size, max(batch_size // 2, 4), 2]:
            try:
                print(f"Transcribing ({language}, batch={attempt_batch})...", flush=True)
                segments_iter, info = batched_model.transcribe(
                    audio,
                    batch_size=attempt_batch,
                    language=language,
                    word_timestamps=True,
                    log_progress=True,
                )
                result_segments = list(segments_iter)
                print(f"Transcription done: {len(result_segments)} segments, detected_lang={info.language}", flush=True)
                break
            except torch.cuda.OutOfMemoryError:
                print(f"OOM with batch={attempt_batch}, trying smaller...", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                continue

        if result_segments is None:
            return {"error": "Out of GPU memory even with batch_size=2"}

        segments = []
        word_segments = []
        for seg in result_segments:
            words = []
            if seg.words:
                for w in seg.words:
                    word_obj = {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "score": getattr(w, "probability", None),
                    }
                    words.append(word_obj)
                    word_segments.append(word_obj)
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words,
            })

        return {
            "segments": segments,
            "word_segments": word_segments,
            "language": info.language if info else language,
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
