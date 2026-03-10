"""RunPod serverless handler for WhisperX transcription + alignment."""
import sys
print(f"[INIT] Python {sys.version}", flush=True)

import torch
print(f"[INIT] PyTorch {torch.__version__}", flush=True)

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
print("[INIT] Patched torch.load (weights_only=False forced)", flush=True)

import gc
import os
import io
import tempfile
from pathlib import Path

import requests
import runpod
import whisperx
print(f"[INIT] WhisperX imported successfully", flush=True)

WHISPERX_DIR = Path(whisperx.__file__).parent
BUNDLED_VAD = WHISPERX_DIR / "assets" / "pytorch_model.bin"
print(f"[INIT] Bundled VAD model path: {BUNDLED_VAD} (exists={BUNDLED_VAD.exists()})", flush=True)

if not BUNDLED_VAD.exists():
    print("[INIT] Bundled VAD not found, downloading to torch cache...", flush=True)
    import urllib.request as _urllib_req
    _original_urlopen = _urllib_req.urlopen
    def _patched_urlopen(url, *args, **kwargs):
        target = url if isinstance(url, str) else getattr(url, 'full_url', None)
        if target and target.startswith("http"):
            resp = requests.get(target, allow_redirects=True, timeout=300)
            resp.raise_for_status()
            return io.BytesIO(resp.content)
        return _original_urlopen(url, *args, **kwargs)
    _urllib_req.urlopen = _patched_urlopen
    sys.modules['urllib.request'].urlopen = _patched_urlopen
    print("[INIT] Patched urllib for redirect support", flush=True)

MODEL = None
MODEL_DIR = "/models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

def setup():
    global MODEL
    model_name = os.environ.get("WHISPER_MODEL", "large-v2")
    print(f"[INIT] Loading WhisperX {model_name} on {DEVICE} ({COMPUTE_TYPE})...", flush=True)

    vad_options = {}
    if BUNDLED_VAD.exists():
        vad_options["model_fp"] = str(BUNDLED_VAD)
        print(f"[INIT] Using bundled VAD model: {BUNDLED_VAD}", flush=True)

    MODEL = whisperx.load_model(
        model_name,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=MODEL_DIR,
        vad_options=vad_options
    )
    print("[INIT] Model loaded successfully!", flush=True)

def download_audio(url):
    """Download audio from URL to temp file."""
    suffix = ".wav" if ".wav" in url else ".mp3" if ".mp3" in url else ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    print(f"Downloading audio from {url}...")

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
    print(f"Downloaded {size_mb:.1f}MB to {tmp.name}")
    return tmp.name

def handler(job):
    inp = job["input"]
    audio_url = inp.get("audio", "")
    audio_base64 = inp.get("audio_base64", "")
    language = inp.get("language", "ru")
    batch_size = inp.get("batch_size", 32)
    align_words = inp.get("align", True)

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

        audio = whisperx.load_audio(tmp_path)

        print(f"Transcribing ({language}, batch={batch_size})...")
        result = MODEL.transcribe(audio, batch_size=batch_size, language=language, print_progress=True)

        detected_lang = result.get("language", language)

        if align_words:
            print(f"Aligning words ({detected_lang})...")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=DEVICE,
                    model_dir=MODEL_DIR
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    DEVICE,
                    return_char_alignments=False,
                    print_progress=True
                )
                del model_a, metadata
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Alignment failed: {e}, returning without word timestamps")

        segments = []
        for seg in result.get("segments", []):
            words = []
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "score": w.get("score")
                })
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text", ""),
                "words": words
            })

        word_segments = []
        for ws in result.get("word_segments", []):
            word_segments.append({
                "word": ws.get("word", ""),
                "start": ws.get("start"),
                "end": ws.get("end"),
                "score": ws.get("score")
            })

        return {
            "segments": segments,
            "word_segments": word_segments,
            "language": detected_lang
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

setup()
runpod.serverless.start({"handler": handler})
