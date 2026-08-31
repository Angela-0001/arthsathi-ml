"""
Offline ASR using Vosk — runs entirely on CPU, no internet, no API.
Model download is one-time; after that it's fully local.

Member C owns this file.

Vosk Hindi model (~50MB): https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip
Vosk Marathi: not officially released — use hi model as fallback for Marathi.
"""

import json
import wave
import os
from pathlib import Path

VOSK_MODEL_DIR = Path("speech/models/vosk")
HINDI_MODEL_PATH = VOSK_MODEL_DIR / "vosk-model-small-hi-0.22"


def download_vosk_model(lang: str = "hi") -> Path:
    """Download Vosk model if not present."""
    import urllib.request
    import zipfile

    VOSK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = VOSK_MODEL_DIR / f"vosk-model-small-{lang}-0.22"

    if model_path.exists():
        print(f"[ASR] Model already exists: {model_path}")
        return model_path

    urls = {
        "hi": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
        "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    }
    url = urls.get(lang)
    if not url:
        raise ValueError(f"No Vosk model URL for lang={lang}")

    zip_path = VOSK_MODEL_DIR / f"vosk_{lang}.zip"
    print(f"[ASR] Downloading Vosk {lang} model (~50MB)...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(VOSK_MODEL_DIR)
    zip_path.unlink()

    print(f"[ASR] Model ready: {model_path}")
    return model_path


def transcribe_file(wav_path: str, lang: str = "hi") -> str:
    """
    Transcribe a WAV file to text using Vosk (offline).
    WAV must be: mono, 16kHz, 16-bit PCM.
    """
    from vosk import Model, KaldiRecognizer

    model_path = VOSK_MODEL_DIR / f"vosk-model-small-{lang}-0.22"
    if not model_path.exists():
        download_vosk_model(lang)

    model = Model(str(model_path))

    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("WAV must be mono 16-bit PCM. Convert with: ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav")

        sample_rate = wf.getframerate()
        rec = KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)

        results = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                results.append(result.get("text", ""))

        final = json.loads(rec.FinalResult())
        results.append(final.get("text", ""))

    return " ".join(r for r in results if r).strip()


def transcribe_bytes(audio_bytes: bytes, lang: str = "hi", sample_rate: int = 16000) -> str:
    """Transcribe raw PCM audio bytes (from IVR/microphone)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        _write_wav(tmp.name, audio_bytes, sample_rate)
        tmp_path = tmp.name

    try:
        return transcribe_file(tmp_path, lang)
    finally:
        os.unlink(tmp_path)


def _write_wav(path: str, pcm_bytes: bytes, sample_rate: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python speech/asr.py <path_to_wav>")
    else:
        text = transcribe_file(sys.argv[1])
        print(f"Transcript: {text}")
