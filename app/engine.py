import torch
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

class XLSREngine:
    def __init__(self):
        self.device = "cpu"
        self.model_id = "facebook/wav2vec2-base-960h"

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_id
        )

        self.model = Wav2Vec2Model.from_pretrained(
            self.model_id,
            output_hidden_states=True
        ).to(self.device)

        self.model.eval()
        torch.set_grad_enabled(False)

    @torch.no_grad()
    def infer_chunk(self, audio_chunk, sr=16000):
        try:
            inputs = self.feature_extractor(
                audio_chunk,
                sampling_rate=sr,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            if not outputs.hidden_states:
                return 0.5

            # -------------------------
            # Stable forensic layer
            # -------------------------
            # Use mid-layer (most stable)
            layer = outputs.hidden_states[6]   # safer than last layer

            # Variance over time, mean over features
            var_t = torch.var(layer, dim=1)    # (batch, features)
            mean_var = torch.mean(var_t).item()

            # Length normalization
            length = layer.shape[1]
            norm_var = mean_var / np.log(length + 1.0)

            # Fixed calibration (no std explosion)
            z = (norm_var - 0.015) / 0.010
            neural_risk = 1.0 / (1.0 + np.exp(-z))

            return float(round(np.clip(neural_risk, 0.0, 1.0), 3))

        except Exception as e:
            print(f"[XLSR ERROR] {e}")
            return 0.5
