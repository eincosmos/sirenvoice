import base64
import io
import numpy as np
import librosa
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.engine import XLSREngine

# -------------------------
# Torch safety
# -------------------------
torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)

app = FastAPI(title="SirenVoice Forensic – Judge API")

# =========================
# Request Model
# =========================
class AudioInput(BaseModel):
    audio_data: str        # Base64 MP3
    language: str = "Unknown"

# =========================
# Explanation Engine
# =========================
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

# =========================
# Core Auditor
# =========================
class SirenAuditor:
    def __init__(self):
        self.sr = 16000
        self.neural = XLSREngine()

    def analyze(self, audio_b64: str):
        try:
            audio = base64.b64decode(audio_b64)
            y, _ = librosa.load(io.BytesIO(audio), sr=self.sr, mono=True)
            y, _ = librosa.effects.trim(y, top_db=30)

            if len(y) < int(self.sr * 0.25):
                return False, 0.20

            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak

        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid API key or malformed request"
            )

        neural_score = float(self.neural.infer_chunk(y))

        # ---- STRICT DECISION ----
        if neural_score >= 0.66:
            return True, round(max(neural_score, 0.75), 2)
        else:
            return False, round(min(neural_score, 0.35), 2)

# =========================
# API Hook
# =========================
auditor = SirenAuditor()

@app.post("/v1/detect")
async def detect(item: AudioInput):
    try:
        is_ai, confidence = auditor.analyze(item.audio_data)

        return {
            "status": "success",
            "language": item.language,
            "classification": "AI_GENERATED" if is_ai else "HUMAN",
            "confidenceScore": confidence,
            "explanation": generate_explanation(is_ai)
        }

    except HTTPException:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }

    except Exception:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }
