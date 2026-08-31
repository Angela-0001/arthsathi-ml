"""
IVR Handler — Asterisk AGI (Asterisk Gateway Interface) script.

How it works:
  - User dials a number on ANY phone (feature phone, no smartphone needed)
  - Asterisk receives the call and runs this Python script via AGI
  - Script plays prompts using TTS, records user speech, runs ASR
  - Sends normalized message to our backend API
  - Plays the response back via TTS

No Twilio, no external API — just Asterisk (open-source PBX) + our own models.

Member C owns this file.

Setup (on Linux server):
  sudo apt-get install asterisk
  Place this file at: /var/lib/asterisk/agi-bin/ivr_handler.py
  chmod +x /var/lib/asterisk/agi-bin/ivr_handler.py

  In /etc/asterisk/extensions.conf:
    [arthsathi]
    exten => 1234,1,AGI(ivr_handler.py)
"""

import sys
import os
import json
import time
import subprocess
import requests
import tempfile
from pathlib import Path

# Add project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TTS_API_URL  = os.getenv("TTS_API_URL",  "http://localhost:5002/tts")
ASR_RECORD_DIR = Path("/tmp/arthsathi_ivr")
ASR_RECORD_DIR.mkdir(parents=True, exist_ok=True)


# ── Asterisk AGI protocol helpers ─────────────────────────────────────────────

def agi_send(cmd: str) -> str:
    """Send a command to Asterisk and read the response."""
    sys.stdout.write(cmd + "\n")
    sys.stdout.flush()
    return sys.stdin.readline().strip()


def agi_get_variable(name: str) -> str:
    resp = agi_send(f"GET VARIABLE {name}")
    # Response format: 200 result=1 (value)
    if "(" in resp and ")" in resp:
        return resp.split("(")[1].rstrip(")")
    return ""


def agi_stream_file(filename: str, escape_digits: str = "") -> str:
    """Play an audio file. filename without extension."""
    resp = agi_send(f"STREAM FILE {filename} \"{escape_digits}\"")
    return resp


def agi_record_file(filename: str, timeout_ms: int = 5000, silence_sec: int = 3) -> str:
    """Record caller's voice to a file."""
    resp = agi_send(f"RECORD FILE {filename} wav # {timeout_ms} {silence_sec} BEEP s={silence_sec}")
    return resp


def agi_say_number(number: int) -> str:
    return agi_send(f"SAY NUMBER {number} \"\"")


def agi_hangup():
    agi_send("HANGUP")


# ── Main IVR call flow ─────────────────────────────────────────────────────────

def run_ivr():
    # Read AGI environment variables from Asterisk
    agi_env = {}
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            break
        if ":" in line:
            key, val = line.split(":", 1)
            agi_env[key.strip()] = val.strip()

    caller_id = agi_env.get("agi_callerid", "unknown")
    lang = "hi"  # default Hindi; could add DTMF menu to select language

    # ── Step 1: Welcome ──────────────────────────────────────────────────
    welcome_audio = _tts_to_file(
        "नमस्ते! ArthSathi में आपका स्वागत है। "
        "सरकारी योजना के लिए 1 दबाएं। "
        "दस्तावेज़ जाँच के लिए 2 दबाएं। "
        "वित्तीय सलाह के लिए 3 दबाएं।",
        lang=lang,
        filename="welcome"
    )
    digit = agi_stream_file(welcome_audio, escape_digits="123")

    # ── Step 2: Dispatch by DTMF choice ─────────────────────────────────
    intent_map = {
        "1": "scheme_match",
        "2": "document_analysis",
        "3": "financial_roadmap",
    }
    intent = intent_map.get(digit.split("=")[-1].strip().strip(")"), "general_query")

    # ── Step 3: Prompt and record if needed ─────────────────────────────
    record_path = str(ASR_RECORD_DIR / f"{caller_id}_{int(time.time())}")

    if intent == "scheme_match":
        prompt_audio = _tts_to_file("कृपया अपनी उम्र, आय और पेशा बोलें।", lang=lang, filename="prompt_scheme")
        agi_stream_file(prompt_audio)
        agi_record_file(record_path, timeout_ms=8000, silence_sec=3)
        raw_text = _transcribe(record_path + ".wav", lang)
    else:
        raw_text = ""

    # ── Step 4: Call our backend ─────────────────────────────────────────
    response_text = _call_backend(caller_id, intent, raw_text, lang)

    # ── Step 5: Speak the response ───────────────────────────────────────
    response_audio = _tts_to_file(response_text, lang=lang, filename=f"response_{int(time.time())}")
    agi_stream_file(response_audio)

    # ── Step 6: Repeat or hang up ────────────────────────────────────────
    bye_audio = _tts_to_file("धन्यवाद। ArthSathi का उपयोग करने के लिए धन्यवाद।", lang=lang, filename="bye")
    agi_stream_file(bye_audio)
    agi_hangup()


def _tts_to_file(text: str, lang: str, filename: str) -> str:
    """Call our own TTS API and save to /tmp. Returns filename without extension."""
    try:
        resp = requests.post(TTS_API_URL, json={"text": text, "lang": lang}, timeout=10)
        resp.raise_for_status()
        out_path = ASR_RECORD_DIR / f"{filename}.wav"
        out_path.write_bytes(resp.content)
        return str(out_path.with_suffix(""))
    except Exception as e:
        # Fallback: use local pyttsx3 directly
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from speech.tts import speak_text
        out_path = str(ASR_RECORD_DIR / f"{filename}.wav")
        speak_text(text, lang=lang, output_file=out_path)
        return str(ASR_RECORD_DIR / filename)


def _transcribe(wav_path: str, lang: str) -> str:
    try:
        from speech.asr import transcribe_file
        return transcribe_file(wav_path, lang)
    except Exception as e:
        return ""


def _call_backend(caller_id: str, intent: str, raw_text: str, lang: str) -> str:
    try:
        payload = {
            "user_id": caller_id,
            "channel": "ivr",
            "intent": intent,
            "raw_text": raw_text,
            "detected_language": lang,
        }
        resp = requests.post(f"{BACKEND_URL}/gateway/message", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json().get("text_response", "क्षमा करें, कोई त्रुटि हुई।")
    except Exception:
        return "क्षमा करें, अभी सेवा उपलब्ध नहीं है। कृपया बाद में प्रयास करें।"


if __name__ == "__main__":
    run_ivr()
