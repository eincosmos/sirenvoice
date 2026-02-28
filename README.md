# 🎙️ SirenVoice — AI Voice Forensics API

> Real-time detection of AI-generated speech using neural deepfake classification + acoustic forensics.


---

## What It Does

SirenVoice analyses an audio clip and tells you whether the voice is **human or AI-generated**. It combines a fine-tuned deepfake classification model with a 5-signal acoustic forensics ensemble to produce a forensic verdict and score.

```
POST /v1/detect  →  { verdict, forensic_score, neural, pci, glottal_asymmetry, ... }
```

**Verdicts:** `AI-Confident` · `AI-Likely` · `Uncertain` · `Human-Likely`

---

## Demo Results

Tested on 11 audio files (5 AI-generated, 6 human):

| # | Truth | Verdict | Score | Correct? |
|---|-------|---------|-------|----------|
| AI Sample 1 | AI | Human-Likely | 0.33 | ✖ |
| AI Sample 2 | AI | AI-Likely | 0.61 | ✔ |
| AI Sample 3 | AI | AI-Likely | 0.52 | ✔ |
| AI Sample 4 | AI | Uncertain | 0.40 | ~ |
| AI Sample 5 | AI | Human-Likely | 0.27 | ✖ |
| Human Sample 1 | Human | Human-Likely | 0.34 | ✔ |
| Human Sample 2 | Human | Uncertain | 0.35 | ~ |
| Human Sample 3 | Human | Uncertain | 0.41 | ~ |
| Human Sample 4 | Human | Uncertain | 0.41 | ~ |
| Human Sample 5 | Human | Uncertain | 0.36 | ~ |
| Human Sample 6 | Human | Human-Likely | 0.30 | ✔ |

**Result: 4 correct · 2 wrong · 5 uncertain · 0 false positives on human speech**

> **Why the misses?** AI Samples 1, 4, 5 were generated with **Google AI Studio TTS** — one of the hardest modern systems to detect. ASVspoof2019 (training data) predates these models. Human samples include Tamil and code-switched Tamil/English speech, outside the model's English training distribution. Both are acknowledged open research problems.

---

## Architecture

```
Audio Input (Base64 MP3/WAV)
        │
        ▼
┌─────────────────────────────────────────┐
│           SirenAuditor                  │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  Neural Engine (55% weight)      │   │
│  │  wav2vec2-large-xlsr             │   │
│  │  fine-tuned on ASVspoof2019      │   │
│  │  Accuracy: 92.9% · F1: 0.936    │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  Acoustic Ensemble (45% weight)  │   │
│  │  · Coarticulation Inertia (PCI)  │   │
│  │  · Glottal Asymmetry             │   │
│  │  · MFCC Jerk Variance            │   │
│  │  · Spectral Entropy              │   │
│  │  · Prosodic Flatness             │   │
│  └──────────────────────────────────┘   │
│                                         │
│  Weighted score → Verdict + Explanation │
└─────────────────────────────────────────┘
```

**Why two systems?** The neural classifier is highly accurate on English TTS but can be fooled by newer models. The acoustic features capture language-agnostic physiological signals (glottal pulse, articulatory motion) that TTS systems still struggle to replicate perfectly. Together they're more robust than either alone.

---

## Quick Start

### 1. Install dependencies
```bash
pip install fastapi uvicorn librosa torch transformers numpy requests scipy
```

### 2. Project structure
```
SirenVoice/
├── app/
│   ├── __init__.py
│   ├── engine.py
│   └── main.py
└── test_api.py
```

### 3. Start the server
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
> First run downloads the model (~1.2 GB). Wait for `Application startup complete`.

### 4. Run the test suite
```bash
python test_api.py
```

### 5. Hit the API directly
```bash
# Health check
curl http://127.0.0.1:8000/health

# Detect (base64-encode your audio first)
curl -X POST http://127.0.0.1:8000/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "<base64-encoded-audio>"}'
```

---

## API Reference

### `POST /v1/detect`

**Request**
```json
{
  "audio_data": "<base64-encoded audio — MP3, WAV, FLAC, OGG>",
  "label": "AI"
}
```

**Response**
```json
{
  "verdict": "AI-Likely",
  "forensic_score": 0.6123,
  "confidence": "Moderate",
  "explanation": "The deepfake classifier indicates likely synthetic origin...",
  "neural": 0.7832,
  "pci": 0.0221,
  "glottal_asymmetry": 0.074,
  "physical_jerk": 23.579,
  "spectral_entropy": 7.766,
  "prosodic_flatness": 0.312,
  "duration_sec": 17.15,
  "processing_ms": 14258.7
}
```

### `GET /health`
```json
{ "status": "ok", "version": "4.0.0" }
```

---

## Neural Model

**[Gustking/wav2vec2-large-xlsr-deepfake-audio-classification](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification)**

Fine-tuned on ASVspoof2019 — the standard benchmark for anti-spoofing research.

| Metric | Score |
|--------|-------|
| Accuracy | 92.86% |
| Precision | 99.99% |
| Recall | 92.05% |
| F1 | 93.63% |
| Equal Error Rate | 4.01% |

---

## Acoustic Features Explained

| Feature | What It Measures | AI Pattern |
|---------|-----------------|------------|
| **PCI** (Phonetic Coarticulation Inertia) | Asymmetry in spectral centroid motion | Low — AI transitions are too symmetric |
| **Glottal Asymmetry** | Biomechanical irregularity in the vocal source | Low — TTS glottal pulses are over-regular |
| **MFCC Jerk** | Second-order articulatory movement variance | Low — AI articulation is unnaturally smooth |
| **Spectral Entropy** | Spectral distribution breadth | Low — AI voices have narrower spectra |
| **Prosodic Flatness** | Syllable rhythm rate (envelope peaks/sec) | Low — TTS intonation is flatter |

---

## Known Limitations

- **Modern neural TTS** (Google AI Studio, ElevenLabs, OpenAI TTS) is significantly harder to detect than older systems. The underlying ASVspoof2019 training data predates these models.
- **Non-English speech** — the neural model was trained primarily on English. Tamil and other languages may produce unreliable neural scores. Acoustic features are language-agnostic and remain valid.
- **Short clips** (< 2 seconds) reduce reliability.
- **Processing speed** — currently ~8–24 seconds on CPU. On AMD GPU with ROCm, this would drop to under 1 second per clip.

---

## Roadmap

- [ ] ROCm/AMD GPU acceleration for real-time inference
- [ ] Fine-tune on multilingual data (Tamil, Hindi, code-switched speech)
- [ ] Collect modern TTS samples (Google, ElevenLabs) for fine-tuning
- [ ] Streaming API for live call monitoring
- [ ] Confidence calibration via temperature scaling

---

## Tech Stack

- **FastAPI** — API framework
- **HuggingFace Transformers** — model loading and inference
- **librosa** — audio feature extraction
- **PyTorch** — neural inference
- **scipy** — signal processing

---

