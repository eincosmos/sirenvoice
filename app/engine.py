"""
engine.py — SirenVoice Neural Forensic Engine v4

Replaces all XLSR-53 feature hacking with a model that was actually
trained to detect deepfakes:

  Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
  - Fine-tuned on ASVspoof2019
  - Accuracy: 92.86%  |  F1: 0.936  |  EER: 4.01%
  - Labels: {"fake": AI-generated, "real": human}

The engine simply runs this classifier and returns the "fake" probability
as the neural risk score. No manual feature engineering needed.
"""

import torch
import numpy as np
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"


class XLSREngine:
    """
    Neural forensic engine backed by a deepfake-detection fine-tuned model.
    Returns a risk score in [0, 1] — higher = more likely AI/fake.
    """

    def __init__(self, device: str = "cpu"):
        self.device   = device
        self.sr       = 16_000

        print(f"[Engine] Loading {MODEL_ID} …")
        self.extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self.model     = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(device)
        self.model.eval()

        # Map label string → index so we can pull "fake" probability reliably
        self.id2label  = self.model.config.id2label   # e.g. {0: "fake", 1: "real"}
        self.fake_idx  = next(
            (i for i, l in self.id2label.items() if l.lower() == "fake"),
            0   # fallback: index 0
        )
        print(f"[Engine] Ready. Labels: {self.id2label}  fake_idx={self.fake_idx}")

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def infer_chunk(self, audio_chunk: np.ndarray, sr: int = 16_000) -> float:
        """
        Score a single audio chunk. Returns P(fake) in [0, 1].
        """
        try:
            inputs = self.extractor(
                audio_chunk,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            logits = self.model(**inputs).logits          # (1, num_labels)
            probs  = torch.softmax(logits, dim=-1)[0]     # (num_labels,)
            fake_prob = probs[self.fake_idx].item()

            return float(round(fake_prob, 4))

        except Exception as e:
            print(f"[XLSR ERROR] {e}")
            return 0.5

    @torch.no_grad()
    def infer_segments(
        self,
        audio: np.ndarray,
        sr: int = 16_000,
        segment_sec: float = 5.0,
        overlap_sec: float = 0.5,
    ) -> float:
        """
        Weighted average over overlapping segments.
        Longer segments get proportionally more weight.
        """
        seg_len = int(segment_sec * sr)
        hop_len = int((segment_sec - overlap_sec) * sr)

        if len(audio) <= seg_len:
            return self.infer_chunk(audio, sr)

        scores, weights = [], []
        start = 0
        while start < len(audio):
            end   = min(start + seg_len, len(audio))
            chunk = audio[start:end]
            if len(chunk) >= int(sr * 0.5):          # skip very short tail
                scores.append(self.infer_chunk(chunk, sr))
                weights.append(len(chunk))
            start += hop_len

        if not scores:
            return 0.5

        return float(round(
            np.average(scores, weights=np.array(weights, dtype=float)), 4
        ))