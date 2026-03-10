#!/usr/bin/env python3
import runpod
import whisperx
import torch
import json
import os
import tempfile
import requests
import base64
import time
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

align_models = {}

def get_align_model(language_code):
    if language_code not in align_models:
        sys.stderr.write(f"[whisperx] Loading alignment model for '{language_code}' on {device}...\n")
        model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        align_models[language_code] = (model, metadata)
        sys.stderr.write(f"[whisperx] Alignment model for '{language_code}' loaded\n")
    return align_models[language_code]


def download_audio(url, dest_path):
    sys.stderr.write(f"[whisperx] Downloading audio from {url[:80]}...\n")
    start = time.time()
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    elapsed = time.time() - start
    size_mb = len(resp.content) / 1024 / 1024
    sys.stderr.write(f"[whisperx] Downloaded {size_mb:.1f}MB in {elapsed:.1f}s\n")


def handler(job):
    job_input = job["input"]

    mode = job_input.get("mode", "transcribe_align")
    language = job_input.get("language", "ru")
    audio_url = job_input.get("audio_url") or job_input.get("audio_file") or job_input.get("audio")
    audio_base64 = job_input.get("audio_base64")
    segments_input = job_input.get("segments")
    model_size = job_input.get("model", "large-v2")
    batch_size = job_input.get("batch_size", 16)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        if audio_base64:
            sys.stderr.write(f"[whisperx] Decoding base64 audio...\n")
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
        elif audio_url:
            download_audio(audio_url, audio_path)
        else:
            return {"error": "No audio provided. Use 'audio_url', 'audio_file', or 'audio_base64'"}

        audio = whisperx.load_audio(audio_path)

        if mode == "align_only":
            if not segments_input:
                return {"error": "mode='align_only' requires 'segments' input"}

            sys.stderr.write(f"[whisperx] Align-only mode: {len(segments_input)} segments, lang={language}\n")
            align_model, metadata = get_align_model(language)

            start = time.time()
            result = whisperx.align(
                segments_input,
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
            elapsed = time.time() - start

            aligned = format_segments(result)
            total_words = sum(len(s.get("words", [])) for s in aligned)
            sys.stderr.write(f"[whisperx] Alignment done in {elapsed:.1f}s: {len(aligned)} segments, {total_words} words\n")

            return {
                "segments": aligned,
                "language": language,
                "mode": "align_only",
                "align_time": round(elapsed, 2)
            }

        else:
            sys.stderr.write(f"[whisperx] Transcribe+Align mode: model={model_size}, lang={language}, batch={batch_size}\n")

            sys.stderr.write(f"[whisperx] Loading whisper model '{model_size}'...\n")
            model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)

            sys.stderr.write(f"[whisperx] Transcribing...\n")
            t0 = time.time()
            result = model.transcribe(audio, batch_size=batch_size, language=language)
            transcribe_time = time.time() - t0
            sys.stderr.write(f"[whisperx] Transcription done in {transcribe_time:.1f}s: {len(result['segments'])} segments\n")

            del model
            torch.cuda.empty_cache()

            sys.stderr.write(f"[whisperx] Aligning...\n")
            align_model, metadata = get_align_model(language)

            t1 = time.time()
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
            align_time = time.time() - t1
            sys.stderr.write(f"[whisperx] Alignment done in {align_time:.1f}s\n")

            aligned = format_segments(result)
            total_words = sum(len(s.get("words", [])) for s in aligned)
            sys.stderr.write(f"[whisperx] Total: {len(aligned)} segments, {total_words} words\n")

            return {
                "segments": aligned,
                "language": result.get("language", language),
                "mode": "transcribe_align",
                "transcribe_time": round(transcribe_time, 2),
                "align_time": round(align_time, 2)
            }


def format_segments(result):
    aligned = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word": w.get("word", ""),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "score": round(w.get("score", 0), 3)
                })
        aligned.append({
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "text": seg.get("text", ""),
            "words": words
        })
    return aligned


runpod.serverless.start({"handler": handler})
