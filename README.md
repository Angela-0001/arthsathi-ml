# ArthSathi ML Core

> This repo is the ML/AI core. Frontend (already scaffolded separately) integrates later.
> Two members handle data + models. One member handles translation + voice — all offline, no external APIs.

---

## Research Gaps This Project Addresses

### Gap 1 — No multilingual financial/legal LLM for rural Indian languages
BharatGen's FinanceParam (the closest existing model) is English+Hindi only.
Its quantized version scores **20.8% on its own benchmark — below random guessing (25%)**.
Critically, it scores **0% on Taxation and Legal Finance** — the exact domains ArthSathi needs.
Source: BharatGen BhashaBench-Finance evaluation; IndicFinNLP (LREC-COLING 2024).

### Gap 2 — No scheme/welfare eligibility domain in any existing model or dataset
IndiaFinBench (arxiv 2025) documents that all Indian finance benchmarks draw from banking/taxation.
No model has been trained or evaluated on government welfare scheme eligibility language.
ArthSathi builds this: a fine-tuned model + evaluation set specifically for scheme+insurance domain.

### Gap 3 — Digital financial inclusion tools assume smartphone + internet
UPI 123Pay covers payments for feature-phone users, but nothing covers scheme guidance,
document safety, or insurance discovery for users without smartphones.
ArthSathi's IVR channel directly fills this gap using Asterisk (open-source PBX) + local inference.

### Gap 4 — No low-latency offline translation for financial/legal language
IndicTrans2 (AI Kosh) is the best open base, but it is general-purpose.
Financial/legal domain vocabulary (e.g. "forfeit", "irrevocable", "PM-JAY eligibility") degrades
quality in general NMT models. ArthSathi fine-tunes IndicTrans2 on domain-specific parallel data
and serves it via a self-hosted FastAPI endpoint — making it "our own API", not a third-party call.

---

## Structure

```
arthsathi-ml/
├── data/
│   ├── scripts/
│   │   ├── collect_schemes.py       # fetch from HuggingFace + myScheme API
│   │   ├── collect_insurance.py     # hand-curated GoI insurance schemes
│   │   ├── generate_synthetic.py    # augment with synthetic profiles/interactions
│   │   └── clean_and_format.py      # unified cleaning + embed_text field
│   ├── schemes/                     # raw + clean scheme JSONL files
│   ├── insurance/                   # raw + clean insurance JSONL files
│   └── training/                    # user-item interaction data for adaptive model
│
├── models/
│   ├── scheme_recommender/          # MEMBER A
│   │   ├── train.py                 # build FAISS index over scheme embeddings
│   │   ├── recommender.py           # eligibility filter + cosine rank + adaptive weights
│   │   └── evaluate.py              # precision@k against labeled test set
│   ├── insurance_recommender/       # MEMBER B
│   │   ├── train.py
│   │   ├── recommender.py
│   │   └── evaluate.py
│   └── adaptive_engine/             # SHARED — online weight updates from user interactions
│       ├── engine.py                # BanditWeightEngine: Thompson sampling + decay
│       └── storage.py               # weights persistence (JSON file, swap to DB later)
│
├── language_model/                  # MEMBER A or B — fine-tune Param-1 for scheme+legal domain
│   ├── finetune.py                  # LoRA fine-tune on scheme QA + legal clause data
│   ├── build_dataset.py             # build instruction-tuning dataset from scheme records
│   └── evaluate.py                  # compare vs base Param-1 on BhashaBench-Finance
│
├── translation/                     # MEMBER C
│   ├── finetune_indictrans2.py      # LoRA fine-tune IndicTrans2 on domain corpus
│   ├── serve.py                     # self-hosted FastAPI translation endpoint (YOUR OWN API)
│   ├── build_corpus.py              # assemble parallel corpus from PIB + myScheme + curated
│   └── evaluate_bleu.py             # BLEU: fine-tuned vs base vs general model
│
├── speech/                          # MEMBER C
│   ├── asr.py                       # offline ASR: faster-whisper (local, no API)
│   ├── tts.py                       # offline TTS: Coqui XTTS-v2 (local, no API)
│   └── evaluate_asr.py              # WER on Vistaar Hindi/Marathi benchmark
│
├── channels/                        # SHARED — channel handlers that call ML services
│   ├── ivr_handler.py               # Asterisk AGI script for IVR call flow
│   ├── whatsapp_handler.py          # Twilio/Meta webhook → normalised message
│   └── telegram_handler.py          # python-telegram-bot → normalised message
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_scheme_recommender_dev.ipynb
    └── 03_translation_eval.ipynb
```

---

## Team Split

| Member | Owns | Key deliverable |
|--------|------|-----------------|
| A | `data/schemes/` + `models/scheme_recommender/` + `language_model/` | Fine-tuned Param-1 for scheme domain; scheme FAISS recommender |
| B | `data/insurance/` + `models/insurance_recommender/` + `models/adaptive_engine/` | Insurance recommender with online bandit learning |
| C | `translation/` + `speech/` + `channels/` | Self-hosted translation API; offline ASR/TTS; IVR + WhatsApp + Telegram |

---

## Setup

```bash
cd arthsathi-ml
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# IndicTrans2 toolkit (needed for translation fine-tuning)
git clone https://github.com/AI4Bharat/IndicTrans2
pip install -e IndicTrans2/

# Playwright (for form automation, later)
playwright install chromium
```

## Run order (first time)

```bash
# Step 1: Collect data
python data/scripts/collect_schemes.py
python data/scripts/collect_insurance.py
python data/scripts/generate_synthetic.py   # if real data insufficient
python data/scripts/clean_and_format.py

# Step 2: Build recommender indexes
python models/scheme_recommender/train.py
python models/insurance_recommender/train.py

# Step 3: Fine-tune language model (GPU recommended)
python language_model/build_dataset.py
python language_model/finetune.py

# Step 4: Fine-tune translation model (GPU recommended)
python translation/build_corpus.py
python translation/finetune_indictrans2.py

# Step 5: Start self-hosted translation API
uvicorn translation.serve:app --port 5001

# Step 6: Evaluate everything
python models/scheme_recommender/evaluate.py
python language_model/evaluate.py
python translation/evaluate_bleu.py
python speech/evaluate_asr.py
```

---

## AI Kosh Resources Used

| Resource | URL | Used for |
|----------|-----|---------|
| IndicTrans2 | aikosh.indiaai.gov.in/home/models/details/indic_trans2.html | Base translation model |
| Param-1-2.9B | aikosh.indiaai.gov.in/home/models/details/bharatgen_param_1 | Base LLM for fine-tuning |
| Samanantar | aikosh.indiaai.gov.in/home/datasets/details/samanantar | Translation base corpus |
| BhashaBench-Finance | aikosh.indiaai.gov.in/home/datasets/details/bhashabench_finance.html | Evaluation benchmark |
| Sangraha | aikosh.indiaai.gov.in/home/datasets/details/sangraha.html | Pretraining data (Indic) |
