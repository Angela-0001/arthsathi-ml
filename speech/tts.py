"""
Offline TTS using pyttsx3 — runs on CPU, no internet, no API.
Uses system voices (SAPI5 on Windows, espeak on Linux, NSSpeechSynthesizer on Mac).

For Hindi voice: install espeak-ng with Hindi support on Linux.
  sudo apt-get install espeak-ng espeak-ng-data

Member C owns this file.
"""

import os
import tempfile
from pathlib import Path

AUDIO_OUT_DIR = Path("speech/output")
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Language code → espeak voice name mapping
VOICE_MAP = {
    "hi": "hi",       # Hindi
    "mr": "mr",       # Marathi
    "en": "en-us",    # English
}


def speak_text(text: str, lang: str = "hi", output_file: str | None = None) -> str:
    """
    Convert text to speech and save as WAV.
    Returns path to the output WAV file.

    If output_file is None, saves to speech/output/tts_<timestamp>.wav
    """
    import pyttsx3

    if output_file is None:
        import time
        output_file = str(AUDIO_OUT_DIR / f"tts_{int(time.time())}.wav")

    engine = pyttsx3.init()

    # Try to set language-appropriate voice
    voices = engine.getProperty("voices")
    _set_voice(engine, voices, lang)

    # Slower rate for clarity (rural users, low literacy)
    engine.setProperty("rate", 140)   # default ~200, we slow it down
    engine.setProperty("volume", 1.0)

    engine.save_to_file(text, output_file)
    engine.runAndWait()

    return output_file


def _set_voice(engine, voices, lang: str):
    """Pick the best available voice for the requested language."""
    lang_lower = lang.lower()
    for voice in voices:
        voice_id = voice.id.lower()
        voice_name = voice.name.lower()
        if lang_lower == "hi" and ("hindi" in voice_name or "hi-in" in voice_id or "hin" in voice_id):
            engine.setProperty("voice", voice.id)
            return
        if lang_lower == "mr" and ("marathi" in voice_name or "mr-in" in voice_id):
            engine.setProperty("voice", voice.id)
            return
        if lang_lower == "en" and ("english" in voice_name or "en-us" in voice_id or "en-gb" in voice_id):
            engine.setProperty("voice", voice.id)
            return
    # Fallback: first available voice
    if voices:
        engine.setProperty("voice", voices[0].id)


def speak_bytes(text: str, lang: str = "hi") -> bytes:
    """Return WAV audio as bytes (for IVR streaming)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name

    try:
        speak_text(text, lang, output_file=path)
        return Path(path).read_bytes()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def list_available_voices():
    """Helper to see what voices are installed on the system."""
    import pyttsx3
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    for v in voices:
        print(f"  ID: {v.id}")
        print(f"  Name: {v.name}")
        print(f"  Languages: {v.languages}")
        print()
    engine.stop()


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "नमस्ते, मैं ArthSathi हूँ।"
    lang = sys.argv[2] if len(sys.argv) > 2 else "hi"
    out = speak_text(text, lang)
    print(f"Saved: {out}")
