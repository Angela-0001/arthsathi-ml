"""
WhatsApp channel handler.
Receives Twilio WhatsApp webhook POST, normalizes to our internal message format,
calls our backend, sends reply back through Twilio.

No external AI APIs — only Twilio for the WhatsApp transport layer.

Member C owns this file.
"""

import os
import requests
from fastapi import APIRouter, Form, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from typing import Optional

router = APIRouter()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TWILIO_SID   = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WA_NUM = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")


@router.post("/whatsapp/webhook")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(""),
    MediaUrl0: Optional[str] = Form(None),
    MediaContentType0: Optional[str] = Form(None),
    NumMedia: str = Form("0"),
):
    user_id = From  # e.g. "whatsapp:+919876543210"
    lang = "hi"

    # Determine intent from media type
    intent = None
    payload = {}
    if int(NumMedia) > 0 and MediaUrl0:
        if MediaContentType0 and MediaContentType0.startswith("image/"):
            intent = "document_analysis"
            payload["media_url"] = MediaUrl0

    # Build normalized message for our backend
    msg = {
        "user_id": user_id,
        "channel": "whatsapp",
        "raw_text": Body.strip() or None,
        "detected_language": lang,
        "intent": intent,
        "payload": payload if payload else None,
    }

    response_text = _call_backend(msg)

    # Reply via Twilio TwiML
    twiml = MessagingResponse()
    twiml.message(response_text)
    return Response(content=str(twiml), media_type="application/xml")


def send_whatsapp_message(to: str, text: str):
    """Proactively send a WhatsApp message (e.g. scheme reminders)."""
    client = Client(TWILIO_SID, TWILIO_TOKEN)
    client.messages.create(
        from_=TWILIO_WA_NUM,
        to=to,
        body=text,
    )


def _call_backend(msg: dict) -> str:
    try:
        resp = requests.post(f"{BACKEND_URL}/gateway/message", json=msg, timeout=15)
        resp.raise_for_status()
        return resp.json().get("text_response", "Sorry, something went wrong.")
    except Exception:
        return "क्षमा करें, अभी सेवा उपलब्ध नहीं है।"
