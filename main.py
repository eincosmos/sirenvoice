"""
main.py — SirenVoice Forensic API v4

Changes from v3:
- Neural engine now uses a real fine-tuned deepfake classifier (92.9% acc)
- Neural weight raised from 0.30 → 0.55 since it's now trustworthy
- Acoustic features kept as a secondary ensemble for extra robustness
- Thresholds recalibrated for the new score distribution:
    Neural alone on fake audio should score ~0.85-0.95
    Neural alone on real audio should score ~0.05-0.20
    Combined score should cleanly separate the two classes
"""

import base64
import io
import logging
import time
from typing import Optional

import librosa
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from scipy.signal import find_peaks

from app.engine import XLSREngine

torch.set_num_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sirenvoice")

app = FastAPI(title="SirenVoice Forensic API", version="4.0.0")

SR = 16_000


# ══════════════════════════════════════════════════════════════════════════
# Request / Response
# ══════════════════════════════════════════════════════════════════════════

class AudioInput(BaseModel):
    audio_data: str
    label: Optional[str] = None

    @field_validator("audio_data")
    @classmethod
    def must_be_nonempty(cls, v):
        if not v or len(v) < 100:
            raise ValueError("audio_data appears empty or too short.")
        return v


class ForensicResult(BaseModel):
    verdict: str
    forensic_score: Optional[float]
    confidence: str
    explanation: str
    neural: Optional[float]            = None
    pci: Optional[float]               = None
    glottal_asymmetry: Optional[float] = None
    physical_jerk: Optional[float]     = None
    spectral_entropy: Optional[float]  = None
    prosodic_flatness: Optional[float] = None
    duration_sec: Optional[float]      = None
    processing_ms: Optional[float]     = None


# ══════════════════════════════════════════════════════════════════════════
# Explanations
# ══════════════════════════════════════════════════════════════════════════

_EXPLANATIONS = {
    "AI-Confident": (
        "The deepfake classifier assigns a high synthetic-speech probability, "
        "and supporting acoustic features confirm reduced biomechanical "
        "irregularity. The combined evidence strongly indicates AI generation."
    ),
    "AI-Likely": (
        "The deepfake classifier indicates likely synthetic origin. Acoustic "
        "features provide moderate supporting evidence. Combined, the signals "
        "favour AI generation."
    ),
    "Uncertain": (
        "The classifier score is in an ambiguous range. The voice does not "
        "align strongly with either human or AI profiles. A longer or "
        "cleaner recording may allow a more confident result."
    ),
    "Human-Likely": (
        "The deepfake classifier assigns a low synthetic-speech probability. "
        "Acoustic features are consistent with natural human vocal production."
    ),
}

_CONFIDENCE = {
    "AI-Confident": "High",
    "AI-Likely":    "Moderate",
    "Uncertain":    "Low",
    "Human-Likely": "Moderate",
}


# ══════════════════════════════════════════════════════════════════════════
# Acoustic features  (secondary ensemble — unchanged from v3)
# ══════════════════════════════════════════════════════════════════════════

def coarticulation_inertia(y: np.ndarray) -> float:
    try:
        if len(y) < int(SR * 0.3):
            return 0.0
        centroid = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
        d = np.diff(centroid)
        pos, neg = d[d > 0], d[d < 0]
        if pos.size == 0 or neg.size == 0:
            return 0.0
        fe, be = np.mean(np.abs(pos)), np.mean(np.abs(neg))
        return float(np.clip(abs(fe - be) / (fe + be + 1e-8), 0.0, 1.0))
    except Exception:
        return 0.0


def glottal_asymmetry(y: np.ndarray) -> float:
    try:
        if len(y) < int(SR * 0.5):
            return 0.0
        yp = librosa.effects.preemphasis(y)
        S  = np.abs(librosa.stft(yp, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=SR)
        band  = np.where((freqs >= 80) & (freqs <= 400))[0]
        if band.size < 5:
            return 0.0
        harmonic_skew = np.mean(np.abs(np.diff(S[band, :], axis=0)))
        zcr_var       = np.var(librosa.feature.zero_crossing_rate(yp)[0])
        env = librosa.feature.rms(y=yp)[0]
        d   = np.diff(env)
        pos, neg = d[d > 0], d[d < 0]
        if pos.size == 0 or neg.size == 0:
            return 0.0
        temporal_skew = abs(np.mean(pos) - abs(np.mean(neg)))
        raw = (np.log1p(harmonic_skew) * 0.4 +
               np.log1p(zcr_var)       * 0.3 +
               np.log1p(temporal_skew) * 0.3)
        return float(np.clip(raw / (raw + 1.0), 0.0, 1.0))
    except Exception:
        return 0.0


def mfcc_jerk(y: np.ndarray) -> tuple[float, float]:
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
        jerk = librosa.feature.delta(mfcc, order=2)
        raw  = float(np.mean(np.var(jerk, axis=1)))
        risk = float(np.clip(1.0 - (raw - 5.0) / 20.0, 0.0, 1.0))
        return raw, risk
    except Exception:
        return 0.0, 0.5


def spectral_entropy(y: np.ndarray) -> tuple[float, float]:
    try:
        S   = np.abs(librosa.stft(y)) ** 2
        psd = np.mean(S, axis=1)
        psd = psd / (np.sum(psd) + 1e-10)
        H   = float(-np.sum(psd * np.log2(psd + 1e-10)))
        risk = float(np.clip(1.0 - (H - 5.0) / 4.0, 0.0, 1.0))
        return H, risk
    except Exception:
        return 7.0, 0.5


def prosodic_flatness(y: np.ndarray) -> float:
    try:
        env = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
        duration = len(y) / SR
        if duration < 0.5:
            return 0.5
        kernel = np.ones(5) / 5
        env_sm = np.convolve(env, kernel, mode="same")
        threshold = np.percentile(env_sm, 40)
        peaks, _ = find_peaks(
            env_sm,
            height=threshold,
            distance=int(0.1 * SR / 160),
            prominence=threshold * 0.15
        )
        pps = len(peaks) / duration
        if pps < 1.5:
            risk = 0.8 + 0.2 * np.clip(1.0 - pps / 1.5, 0.0, 1.0)
        elif pps <= 5.0:
            risk = 0.1 + 0.2 * abs(pps - 3.0) / 2.0
        else:
            risk = 0.3 + 0.1 * np.clip((pps - 5.0) / 3.0, 0.0, 1.0)
        return float(np.clip(risk, 0.0, 1.0))
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════

# Neural weight raised to 0.55 — it's now a real deepfake classifier
_W = dict(neural=0.55, jerk=0.15, pci=0.12, entropy=0.10, glottal=0.05, prosodic=0.03)

# Thresholds calibrated for new distribution:
# Real deepfake classifier should push AI audio to 0.7+, human to <0.35
_THRESHOLDS = [
    (0.65, "AI-Confident"),
    (0.50, "AI-Likely"),
    (0.35, "Uncertain"),
    (0.0,  "Human-Likely"),
]


def _verdict(score: float) -> str:
    for t, label in _THRESHOLDS:
        if score >= t:
            return label
    return "Human-Likely"


def _score(neural, jerk_risk, pci, entropy_risk, glottal, prosodic) -> float:
    pci_risk     = float(np.clip(1.0 - pci * 8.0,     0.0, 1.0))
    glottal_risk = float(np.clip(1.0 - glottal * 4.0, 0.0, 1.0))
    raw = (
        _W["neural"]   * neural       +
        _W["jerk"]     * jerk_risk    +
        _W["pci"]      * pci_risk     +
        _W["entropy"]  * entropy_risk +
        _W["glottal"]  * glottal_risk +
        _W["prosodic"] * prosodic
    )
    return float(np.clip(raw, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════
# Auditor
# ══════════════════════════════════════════════════════════════════════════

class SirenAuditor:
    def __init__(self):
        self.engine = XLSREngine()

    def analyze(self, audio_b64: str) -> dict:
        t0 = time.perf_counter()

        try:
            raw  = base64.b64decode(audio_b64)
            y, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
            y, _ = librosa.effects.trim(y, top_db=30)
        except Exception as e:
            logger.warning(f"Decode failed: {e}")
            return self._uncertain("Audio could not be decoded.", t0)

        dur = len(y) / SR
        if dur < 0.25:
            return self._uncertain("Audio too short (< 250 ms).", t0)

        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        # Neural (now trustworthy)
        neural = self.engine.infer_segments(y, SR)

        # Acoustic secondary features
        pci                    = coarticulation_inertia(y)
        glottal                = glottal_asymmetry(y)
        jerk_raw,   jerk_risk  = mfcc_jerk(y)
        entropy_val, ent_risk  = spectral_entropy(y)
        prosodic               = prosodic_flatness(y)

        score   = _score(neural, jerk_risk, pci, ent_risk, glottal, prosodic)
        verdict = _verdict(score)
        ms      = (time.perf_counter() - t0) * 1000

        logger.info(
            f"verdict={verdict} score={score:.3f} neural={neural:.3f} "
            f"pci={pci:.4f} glottal={glottal:.4f} jerk={jerk_raw:.1f} "
            f"entropy={entropy_val:.3f} prosodic={prosodic:.3f} "
            f"dur={dur:.1f}s ms={ms:.0f}"
        )

        return ForensicResult(
            verdict           = verdict,
            forensic_score    = round(score, 4),
            confidence        = _CONFIDENCE[verdict],
            explanation       = _EXPLANATIONS[verdict],
            neural            = round(neural, 4),
            pci               = round(pci, 4),
            glottal_asymmetry = round(glottal, 4),
            physical_jerk     = round(jerk_raw, 3),
            spectral_entropy  = round(entropy_val, 3),
            prosodic_flatness = round(prosodic, 4),
            duration_sec      = round(dur, 2),
            processing_ms     = round(ms, 1),
        ).model_dump()

    def _uncertain(self, reason: str, t0: float) -> dict:
        ms = (time.perf_counter() - t0) * 1000
        return ForensicResult(
            verdict="Uncertain", forensic_score=None,
            confidence="Low", explanation=reason,
            processing_ms=round(ms, 1)
        ).model_dump()


# ══════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════

auditor = SirenAuditor()


@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health")
async def health():
    return {"status": "ok", "version": app.version}


@app.post("/v1/detect", response_model=ForensicResult)
async def detect(item: AudioInput):
    if item.label:
        logger.info(f"Eval request — ground truth: {item.label}")
    return auditor.analyze(item.audio_data)