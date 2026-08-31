"""
Telegram channel handler using python-telegram-bot.
Handles text, voice notes, and document images.
Calls our own backend — no external AI APIs.

Member C owns this file.

Run: python channels/telegram_handler.py
"""

import os
import logging
import requests
import tempfile
from pathlib import Path
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BACKEND_URL    = os.getenv("BACKEND_URL", "http://localhost:8000")
BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN", "")
TRANSLATION_URL = os.getenv("TRANSLATION_API_URL", "http://localhost:5001")


# ── Commands ───────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🙏 नमस्ते! मैं ArthSathi हूँ।\n\n"
        "मैं आपकी मदद कर सकता हूँ:\n"
        "📋 /schemes — योग्य सरकारी योजनाएँ\n"
        "🗺️ /roadmap — वित्तीय रोडमैप\n"
        "📄 /analyze — दस्तावेज़ जाँचें\n\n"
        "Hello! I'm ArthSathi. Send me a message or voice note."
    )


async def cmd_schemes(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    result = _call_backend(user_id, "scheme_match", "", "hi")
    await update.message.reply_text(result)


async def cmd_roadmap(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    result = _call_backend(user_id, "financial_roadmap", "", "hi")
    await update.message.reply_text(result)


# ── Message handlers ───────────────────────────────────────────────────────────

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    text = update.message.text or ""
    result = _call_backend(user_id, "general_query", text, "hi")
    await update.message.reply_text(result)


async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Download voice note → transcribe locally → call backend."""
    user_id = str(update.effective_user.id)
    await update.message.reply_text("🎙️ Processing your voice message...")

    voice_file = await ctx.bot.get_file(update.message.voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await voice_file.download_to_drive(tmp.name)
        ogg_path = tmp.name

    # Convert OGG → WAV for Vosk (requires ffmpeg)
    wav_path = ogg_path.replace(".ogg", ".wav")
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, check=True
        )
        from speech.asr import transcribe_file
        transcribed = transcribe_file(wav_path, lang="hi")
    except Exception as e:
        log.error(f"ASR error: {e}")
        transcribed = ""
    finally:
        for p in [ogg_path, wav_path]:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    if not transcribed:
        await update.message.reply_text("Sorry, could not transcribe your voice. Please type your question.")
        return

    result = _call_backend(user_id, "general_query", transcribed, "hi")
    await update.message.reply_text(f"📝 Heard: {transcribed}\n\n{result}")


async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Download photo/document → send to backend for risk analysis."""
    user_id = str(update.effective_user.id)
    await update.message.reply_text("📄 Analyzing your document for risky clauses...")

    # Get photo (highest resolution)
    if update.message.photo:
        file_obj = await ctx.bot.get_file(update.message.photo[-1].file_id)
    elif update.message.document:
        file_obj = await ctx.bot.get_file(update.message.document.file_id)
    else:
        await update.message.reply_text("Please send an image of the document.")
        return

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        await file_obj.download_to_drive(tmp.name)
        img_path = tmp.name

    try:
        with open(img_path, "rb") as f:
            resp = requests.post(
                f"{BACKEND_URL}/documents/analyze?lang=hi",
                files={"file": f},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

        summary = data.get("summary", "")
        flags = data.get("risk_flags", [])

        reply = f"📋 *Summary*\n{summary}\n"
        if flags:
            reply += f"\n⚠️ *{len(flags)} Risk Clause(s) Found:*\n"
            for flag in flags[:5]:  # cap at 5 for readability
                emoji = "🔴" if flag["risk_level"] == "high" else ("🟡" if flag["risk_level"] == "medium" else "🔵")
                reply += f"{emoji} {flag['risk_level'].upper()}: {flag['explanation']}\n"
        else:
            reply += "\n✅ No obvious risky clauses detected."

        await update.message.reply_text(reply, parse_mode="Markdown")

    except Exception as e:
        log.error(f"Document analysis error: {e}")
        await update.message.reply_text("Could not analyze document. Please try a clearer image.")
    finally:
        Path(img_path).unlink(missing_ok=True)


# ── Backend call helper ────────────────────────────────────────────────────────

def _call_backend(user_id: str, intent: str, raw_text: str, lang: str) -> str:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/gateway/message",
            json={
                "user_id": user_id,
                "channel": "telegram",
                "intent": intent,
                "raw_text": raw_text,
                "detected_language": lang,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("text_response", "Could not get a response.")
    except Exception as e:
        log.error(f"Backend error: {e}")
        return "क्षमा करें, अभी सेवा उपलब्ध नहीं है। Please try again later."


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if not BOT_TOKEN:
        raise ValueError("Set TELEGRAM_BOT_TOKEN env variable")

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("schemes", cmd_schemes))
    app.add_handler(CommandHandler("roadmap", cmd_roadmap))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("Telegram bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
