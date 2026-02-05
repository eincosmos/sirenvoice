import torch
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# In engine.py
class XLSREngine:
    def __init__(self):
        self.device = "cpu"
        # Change 'large' to 'base' for much faster CPU inference
        model_id = "facebook/wav2vec2-base-960h" 
        # Or "facebook/wav2vec2-xlsr-53-espeak-cv-ft" if you need multilingual XLS-R specifically

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2Model.from_pretrained(
            model_id,
            output_hidden_states=True
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def infer_chunk(self, audio_chunk, sr=16000):
        try:
            inputs = self.feature_extractor(
                audio_chunk,
                sampling_rate=sr,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)

            if outputs.hidden_states is None:
                return 0.5

            # Stable forensic variance
            layer = outputs.hidden_states[12]
            var = torch.var(layer, dim=-1)
            mean_var = torch.mean(var).item()
            std_var = torch.std(var).item() + 1e-9

            # Z-score → sigmoid (scale invariant)
            z = (mean_var - 0.02) / std_var
            neural_risk = 1 / (1 + np.exp(-z))

            return float(round(neural_risk, 3))

        except Exception as e:
            print(f"[XLSR ERROR] {e}")
            return 0.5
	
