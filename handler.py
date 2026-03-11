#!/usr/bin/env python3
import runpod
import whisper
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
        sys.stderr.write(f"[worker] Loading alignment model for '{language_code}' on {device}...\n")
        model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        align_models[language_code] = (model, metadata)
        sys.stderr.write(f"[worker] Alignment model for '{language_code}' loaded\n")
    return align_models[language_code]


def download_audio(url, dest_path):
    sys.stderr.write(f"[worker] Downloading audio from {url[:80]}...\n")
    start = time.time()
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    elapsed = time.time() - start
    size_mb = len(resp.content) / 1024 / 1024
    sys.stderr.write(f"[worker] Downloaded {size_mb:.1f}MB in {elapsed:.1f}s\n")


def handler(job):
    job_input = job["input"]

    mode = job_input.get("mode", "transcribe_align")
    language = job_input.get("language", "ru")
    audio_url = job_input.get("audio_url") or job_input.get("audio_file") or job_input.get("audio")
    audio_base64 = job_input.get("audio_base64")
    segments_input = job_input.get("segments")
    model_size = job_input.get("model", "large-v3")
    batch_size = job_input.get("batch_size", 16)
    initial_prompt = job_input.get("initial_prompt", None)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        if audio_base64:
            sys.stderr.write(f"[worker] Decoding base64 audio...\n")
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
        elif audio_url:
            download_audio(audio_url, audio_path)
        else:
            return {"error": "No audio provided. Use 'audio_url', 'audio_file', or 'audio_base64'"}

        if mode == "align_only":
            if not segments_input:
                return {"error": "mode='align_only' requires 'segments' input"}

            audio = whisperx.load_audio(audio_path)
            sys.stderr.write(f"[worker] Align-only mode: {len(segments_input)} segments, lang={language}\n")
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

            aligned, stats = format_segments(result)
            sys.stderr.write(f"[worker] Alignment done in {elapsed:.1f}s: {len(aligned)} segments, {stats['total_words']} words ({stats['aligned_words']} aligned, {stats['unaligned_words']} unaligned)\n")

            return {
                "segments": aligned,
                "language": language,
                "mode": "align_only",
                "align_time": round(elapsed, 2),
                "total_words": stats["total_words"],
                "aligned_words": stats["aligned_words"],
                "unaligned_words": stats["unaligned_words"],
            }

        else:
            sys.stderr.write(f"[worker] Transcribe+Align mode: model={model_size}, lang={language}, engine=openai-whisper\n")
            if initial_prompt:
                sys.stderr.write(f"[worker] Using initial_prompt: {initial_prompt[:80]}...\n")

            sys.stderr.write(f"[worker] Loading OpenAI Whisper model '{model_size}'...\n")
            model = whisper.load_model(model_size, device=device)

            sys.stderr.write(f"[worker] Transcribing with OpenAI Whisper...\n")
            t0 = time.time()
            decode_options = {
                "language": language,
                "word_timestamps": True,
            }
            if initial_prompt:
                decode_options["initial_prompt"] = initial_prompt

            result = model.transcribe(
                audio_path,
                **decode_options
            )
            transcribe_time = time.time() - t0
            sys.stderr.write(f"[worker] Transcription done in {transcribe_time:.1f}s: {len(result.get('segments', []))} segments\n")

            whisperx_segments = []
            for seg in result.get("segments", []):
                whisperx_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                })

            del model
            torch.cuda.empty_cache()

            audio = whisperx.load_audio(audio_path)

            sys.stderr.write(f"[worker] Aligning with WhisperX ({len(whisperx_segments)} segments)...\n")
            align_model, metadata = get_align_model(language)

            t1 = time.time()
            aligned_result = whisperx.align(
                whisperx_segments,
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
            align_time = time.time() - t1
            sys.stderr.write(f"[worker] Alignment done in {align_time:.1f}s\n")

            aligned, stats = format_segments(aligned_result)
            sys.stderr.write(f"[worker] Total: {len(aligned)} segments, {stats['total_words']} words ({stats['aligned_words']} aligned, {stats['unaligned_words']} unaligned)\n")

            return {
                "segments": aligned,
                "language": result.get("language", language),
                "mode": "transcribe_align",
                "engine": "openai-whisper",
                "transcribe_time": round(transcribe_time, 2),
                "align_time": round(align_time, 2),
                "total_words": stats["total_words"],
                "aligned_words": stats["aligned_words"],
                "unaligned_words": stats["unaligned_words"],
            }


def format_segments(result):
    aligned = []
    total_words = 0
    aligned_words = 0
    unaligned_words = 0

    for seg in result.get("segments", []):
        words = []
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        raw_words = seg.get("words", [])

        for w in raw_words:
            has_start = "start" in w and w["start"] is not None
            has_end = "end" in w and w["end"] is not None

            if has_start and has_end:
                words.append({
                    "word": w.get("word", ""),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "score": round(w.get("score", 0), 3)
                })
                aligned_words += 1
            else:
                words.append({
                    "word": w.get("word", ""),
                    "start": None,
                    "end": None,
                    "score": 0.0
                })
                unaligned_words += 1

            total_words += 1

        if not words and seg.get("text", "").strip():
            text_words = seg["text"].strip().split()
            if text_words:
                for i, tw in enumerate(text_words):
                    words.append({
                        "word": tw,
                        "start": None,
                        "end": None,
                        "score": 0.0
                    })
                    unaligned_words += 1
                    total_words += 1

        aligned.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": seg.get("text", ""),
            "words": words
        })

    stats = {
        "total_words": total_words,
        "aligned_words": aligned_words,
        "unaligned_words": unaligned_words,
    }
    return aligned, stats


runpod.serverless.start({"handler": handler})
