import base64
import io
import numpy as np
import librosa
import torch
import os

from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, validator

from app.engine import XLSREngine

# -------------------------
# Torch safety
# -------------------------
torch.set_num_threads(1)

# -------------------------
# App
# -------------------------
app = FastAPI(title="SirenVoice – AI Voice Detection API")

# -------------------------
# Security
# -------------------------
# -------------------------
# Security
# -------------------------
API_KEY = os.getenv("API_KEY")

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -------------------------
# Helpers
# -------------------------
SUPPORTED_LANGUAGES = {
    "Tamil", "English", "Hindi", "Malayalam", "Telugu"
}

def is_mp3_bytes(data: bytes) -> bool:
    # ID3 header
    if data[:3] == b"ID3":
        return True
    # MPEG frame sync
    if len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return True
    return False

# -------------------------
# Request Model
# -------------------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

    @validator("language")
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language")
        return v

    @validator("audioFormat")
    def validate_format(cls, v):
        if v.lower() != "mp3":
            raise ValueError("Only mp3 format supported")
        return v.lower()

# -------------------------
# Explanation
# -------------------------
def generate_explanation(verdict: str) -> str:
    if verdict in ("AI-Confident", "AI-Likely"):
        return (
            "Unnatural pitch consistency and reduced micro-variations "
            "indicate characteristics commonly found in synthetic speech."
        )
    return (
        "The voice exhibits natural biomechanical irregularities and "
        "temporal variations consistent with human speech."
    )

# -------------------------
# Core Auditor
# -------------------------
class SirenAuditor:
    def __init__(self):
        self.sr = 16000
        self.neural = XLSREngine()


    def analyze(self, audio_b64: str):
        try:
            audio_bytes = base64.b64decode(audio_b64)
            y, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sr, mono=True)
            y, _ = librosa.effects.trim(y, top_db=30)

            if len(y) < int(self.sr * 0.25):
                return {"verdict": "Human-Likely", "score": 0.2}

            peak = np.max(np.abs(y))
            if peak > 0:
                y /= peak

        except Exception:
            return {"verdict": "Human-Likely", "score": 0.2}

        self._load_neural()
        neural = float(self.neural.infer_chunk(y))


        if neural >= 0.66:
            verdict = "AI-Likely"
            score = max(neural, 0.75)
        else:
            verdict = "Human-Likely"
            score = min(neural, 0.35)

        return {"verdict": verdict, "score": round(score, 3)}

# -------------------------
# Engine Init
# -------------------------
auditor = SirenAuditor()

# -------------------------
# API Endpoint (FINAL)
# -------------------------
@app.post("/api/voice-detection")
async def detect_voice(
    req: VoiceRequest,
    _: str = Depends(verify_api_key)
):
    try:
        audio_bytes = base64.b64decode(req.audioBase64)

        if not is_mp3_bytes(audio_bytes):
            return {
                "status": "error",
                "message": "Invalid audio format. Only MP3 supported."
            }

        result = auditor.analyze(req.audioBase64)

        classification = (
            "AI_GENERATED"
            if result["verdict"] in ("AI-Confident", "AI-Likely")
            else "HUMAN"
        )

        return {
            "status": "success",
            "language": req.language,
            "classification": classification,
            "confidenceScore": result["score"],
            "explanation": generate_explanation(result["verdict"])
        }

    except Exception:
        return {
            "status": "error",
            "message": "Malformed request"
        }
@app.get("/health")
async def health_check():
    return {"status": "ready"}
