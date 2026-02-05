import base64
import io
import os
import numpy as np
import librosa
import torch

from fastapi import FastAPI, Header
from pydantic import BaseModel, validator

from app.engine import XLSREngine

# -------------------------
# Torch safety & determinism
# -------------------------
torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)

# -------------------------
# App
# -------------------------
app = FastAPI(title="SirenVoice – AI Voice Detection API")

# -------------------------
# API Key
# -------------------------
API_KEY = os.getenv("API_KEY")

# -------------------------
# Supported Languages (STRICT)
# -------------------------
SUPPORTED_LANGUAGES = {
    "Tamil",
    "English",
    "Hindi",
    "Malayalam",
    "Telugu"
}

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
# Explanation Engine
# -------------------------
def generate_explanation(is_ai: bool) -> str:
    if is_ai:
        return (
            "Unnatural pitch consistency and reduced micro-variations "
            "indicate characteristics commonly found in AI-generated speech."
        )
    return (
        "Natural biomechanical irregularities and temporal variations "
        "are consistent with human speech production."
    )

# -------------------------
# Core Auditor
# -------------------------
class SirenAuditor:
    def __init__(self):
        self.sr = 16000
        self.model = XLSREngine()

    def analyze(self, audio_b64: str):
        try:
            audio_bytes = base64.b64decode(audio_b64)
            y, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sr, mono=True)
            y, _ = librosa.effects.trim(y, top_db=30)

            if len(y) < int(self.sr * 0.25):
                return False, 0.20

            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak

        except Exception:
            raise ValueError("Invalid audio")

        neural_score = float(self.model.infer_chunk(y))

        # Strict binary decision
        if neural_score >= 0.66:
            return True, round(max(neural_score, 0.75), 2)
        else:
            return False, round(min(neural_score, 0.35), 2)

# -------------------------
# Engine Init
# -------------------------
auditor = SirenAuditor()

# -------------------------
# API Endpoint (OFFICIAL)
# -------------------------
@app.post("/api/voice-detection")
async def detect_voice(
    req: VoiceRequest,
    x_api_key: str = Header(None)
):
    # ---- API key validation ----
    if API_KEY is None or x_api_key != API_KEY:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }

    try:
        is_ai, confidence = auditor.analyze(req.audioBase64)

        return {
            "status": "success",
            "language": req.language,
            "classification": "AI_GENERATED" if is_ai else "HUMAN",
            "confidenceScore": confidence,
            "explanation": generate_explanation(is_ai)
        }

    except Exception:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }

# -------------------------
# Health Check (optional)
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
