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
import re
import math

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


ONES_RU = {
    0: '', 1: 'один', 2: 'два', 3: 'три', 4: 'четыре',
    5: 'пять', 6: 'шесть', 7: 'семь', 8: 'восемь', 9: 'девять',
    10: 'десять', 11: 'одиннадцать', 12: 'двенадцать', 13: 'тринадцать',
    14: 'четырнадцать', 15: 'пятнадцать', 16: 'шестнадцать',
    17: 'семнадцать', 18: 'восемнадцать', 19: 'девятнадцать',
}
TENS_RU = {
    2: 'двадцать', 3: 'тридцать', 4: 'сорок', 5: 'пятьдесят',
    6: 'шестьдесят', 7: 'семьдесят', 8: 'восемьдесят', 9: 'девяносто',
}
HUNDREDS_RU = {
    1: 'сто', 2: 'двести', 3: 'триста', 4: 'четыреста', 5: 'пятьсот',
    6: 'шестьсот', 7: 'семьсот', 8: 'восемьсот', 9: 'девятьсот',
}

def number_to_russian(n):
    if n == 0:
        return 'ноль'
    if n < 0:
        return 'минус ' + number_to_russian(-n)
    if n > 999999999:
        return str(n)

    parts = []

    if n >= 1000000:
        millions = n // 1000000
        n %= 1000000
        m_text = _small_number_ru(millions)
        if millions == 1:
            parts.append('один миллион')
        elif millions in (2, 3, 4):
            parts.append(m_text + ' миллиона')
        else:
            parts.append(m_text + ' миллионов')

    if n >= 1000:
        thousands = n // 1000
        n %= 1000
        if thousands == 1:
            parts.append('одна тысяча')
        elif thousands == 2:
            parts.append('две тысячи')
        elif thousands in (3, 4):
            parts.append(_small_number_ru(thousands) + ' тысячи')
        elif thousands >= 5 and thousands <= 20:
            parts.append(_small_number_ru(thousands) + ' тысяч')
        else:
            last_digit = thousands % 10
            last_two = thousands % 100
            t_text = _small_number_ru(thousands)
            if last_two >= 11 and last_two <= 19:
                parts.append(t_text + ' тысяч')
            elif last_digit == 1:
                t_text = _small_number_ru(thousands - 1) + ' одна' if thousands > 1 else 'одна'
                parts.append(t_text + ' тысяча')
            elif last_digit == 2:
                t_text = _small_number_ru(thousands - 2) + ' две' if thousands > 2 else 'две'
                parts.append(t_text + ' тысячи')
            elif last_digit in (3, 4):
                parts.append(t_text + ' тысячи')
            else:
                parts.append(t_text + ' тысяч')

    if n > 0:
        parts.append(_small_number_ru(n))

    return ' '.join(parts).strip()


def _small_number_ru(n):
    if n == 0:
        return ''
    if n < 20:
        return ONES_RU[n]
    if n < 100:
        tens = n // 10
        ones = n % 10
        return (TENS_RU[tens] + ' ' + ONES_RU.get(ones, '')).strip()

    hundreds = n // 100
    remainder = n % 100
    h_text = HUNDREDS_RU[hundreds]
    if remainder == 0:
        return h_text
    if remainder < 20:
        return h_text + ' ' + ONES_RU[remainder]
    tens = remainder // 10
    ones = remainder % 10
    return (h_text + ' ' + TENS_RU[tens] + ' ' + ONES_RU.get(ones, '')).strip()


ABBREV_RU = {
    'PSA': 'пи эс эй',
    'MVP': 'эм ви пи',
    'GG': 'джи джи',
    'LOL': 'лол',
    'OMG': 'о эм джи',
    'WTF': 'вэ тэ эф',
    'CSGO': 'кс го',
    'CS': 'кс',
    'VPN': 'ви пи эн',
    'PC': 'пи си',
    'CPU': 'си пи ю',
    'GPU': 'джи пи ю',
    'USB': 'ю эс би',
    'VK': 'вэ ка',
    'TikTok': 'тикток',
    'YouTube': 'ютуб',
    'OK': 'окей',
}


def normalize_text_for_alignment(text, language='ru'):
    if language != 'ru':
        return text

    def replace_number(match):
        num_str = match.group()
        try:
            n = int(num_str)
            if 0 <= n <= 999999999:
                return number_to_russian(n)
        except (ValueError, OverflowError):
            pass
        return num_str

    text = re.sub(r'\b\d+\b', replace_number, text)

    for abbr, replacement in ABBREV_RU.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', replacement, text, flags=re.IGNORECASE)

    return text


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

            for seg in segments_input:
                seg["text"] = normalize_text_for_alignment(seg.get("text", ""), language)

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
                return_char_alignments=False,
                interpolate_method="nearest"
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
                original_text = seg["text"]
                normalized_text = normalize_text_for_alignment(original_text, language)
                whisperx_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": normalized_text,
                    "original_text": original_text,
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
                return_char_alignments=False,
                interpolate_method="nearest"
            )
            align_time = time.time() - t1
            sys.stderr.write(f"[worker] Alignment done in {align_time:.1f}s\n")

            original_texts = {seg["start"]: seg.get("original_text", seg["text"]) for seg in whisperx_segments}
            aligned, stats = format_segments(aligned_result, original_texts)
            sys.stderr.write(f"[worker] Total: {len(aligned)} segments, {stats['total_words']} words ({stats['aligned_words']} aligned, {stats['unaligned_words']} unaligned, {stats['interpolated_words']} interpolated)\n")

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
                "interpolated_words": stats["interpolated_words"],
            }


def format_segments(result, original_texts=None):
    aligned = []
    total_words = 0
    aligned_words = 0
    unaligned_words = 0
    interpolated_words = 0

    for seg in result.get("segments", []):
        words = []
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        raw_words = seg.get("words", [])

        for w in raw_words:
            has_start = "start" in w and w["start"] is not None
            has_end = "end" in w and w["end"] is not None

            if has_start and has_end:
                word_start = w["start"]
                word_end = w["end"]
                if not (isinstance(word_start, (int, float)) and not math.isnan(word_start)):
                    has_start = False
                if not (isinstance(word_end, (int, float)) and not math.isnan(word_end)):
                    has_end = False

            if has_start and has_end:
                words.append({
                    "word": w.get("word", ""),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "score": round(w.get("score", 0) if w.get("score") is not None else 0, 3)
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

        words, n_interpolated = interpolate_word_timestamps(words, seg_start, seg_end)
        interpolated_words += n_interpolated

        seg_text = seg.get("text", "")
        if original_texts and seg_start in original_texts:
            seg_text = original_texts[seg_start]

        aligned.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": seg_text,
            "words": words
        })

    stats = {
        "total_words": total_words,
        "aligned_words": aligned_words,
        "unaligned_words": unaligned_words,
        "interpolated_words": interpolated_words,
    }
    return aligned, stats


def interpolate_word_timestamps(words, seg_start, seg_end):
    if not words:
        return words, 0

    n_interpolated = 0

    unaligned_runs = []
    run_start = None
    for i, w in enumerate(words):
        if w["start"] is None:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                unaligned_runs.append((run_start, i - 1))
                run_start = None
    if run_start is not None:
        unaligned_runs.append((run_start, len(words) - 1))

    for (rs, re_) in unaligned_runs:
        prev_end = seg_start
        if rs > 0 and words[rs - 1]["end"] is not None:
            prev_end = words[rs - 1]["end"]

        next_start = seg_end
        if re_ < len(words) - 1 and words[re_ + 1]["start"] is not None:
            next_start = words[re_ + 1]["start"]

        count = re_ - rs + 1
        duration = next_start - prev_end
        slot = duration / count if count > 0 else 0

        for j in range(count):
            idx = rs + j
            words[idx]["start"] = round(prev_end + j * slot, 3)
            words[idx]["end"] = round(prev_end + (j + 1) * slot, 3)
            words[idx]["score"] = 0.001
            n_interpolated += 1

    return words, n_interpolated


runpod.serverless.start({"handler": handler})
