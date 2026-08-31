"""
ArthSathi Translation API — self-hosted, no external API calls.
This is the team's OWN API, serving inference from the model we trained.

Member C owns this file.

Run:
    uvicorn translation.serve:app --host 0.0.0.0 --port 5001 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import lru_cache

app = FastAPI(
    title="ArthSathi Translation API",
    description="Self-hosted seq2seq translation for Hindi/Marathi ↔ English. No external APIs.",
    version="1.0.0",
)


class TranslationRequest(BaseModel):
    text: str
    src_lang: str   # "hi", "mr", "en"
    tgt_lang: str   # "hi", "mr", "en"
    beam: bool = True


class TranslationResponse(BaseModel):
    translated_text: str
    src_lang: str
    tgt_lang: str
    model: str = "arthsathi-seq2seq-v1"


@lru_cache(maxsize=1)
def get_translator():
    """Load model once, reuse across requests."""
    import torch
    from translation.inference import Translator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[translation API] Loading model on {device}...")
    return Translator(device=device)


SUPPORTED_LANGS = {"hi", "mr", "en"}
SUPPORTED_PAIRS = {
    ("hi", "en"), ("en", "hi"),
    ("mr", "en"), ("en", "mr"),
}


@app.get("/health")
def health():
    return {"status": "ok", "service": "arthsathi-translation"}


@app.post("/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):
    if req.src_lang not in SUPPORTED_LANGS or req.tgt_lang not in SUPPORTED_LANGS:
        raise HTTPException(400, f"Unsupported language. Supported: {SUPPORTED_LANGS}")
    if (req.src_lang, req.tgt_lang) not in SUPPORTED_PAIRS:
        raise HTTPException(400, f"Unsupported language pair: {req.src_lang}→{req.tgt_lang}")
    if req.src_lang == req.tgt_lang:
        return TranslationResponse(
            translated_text=req.text,
            src_lang=req.src_lang,
            tgt_lang=req.tgt_lang,
        )
    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    try:
        translator = get_translator()
        result = translator.translate(req.text, req.src_lang, req.tgt_lang, beam=req.beam)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not loaded: {e}. Train the model first.")
    except Exception as e:
        raise HTTPException(500, f"Translation error: {e}")

    return TranslationResponse(
        translated_text=result,
        src_lang=req.src_lang,
        tgt_lang=req.tgt_lang,
    )


@app.get("/supported")
def supported_pairs():
    return {"pairs": [{"src": s, "tgt": t} for s, t in SUPPORTED_PAIRS]}
