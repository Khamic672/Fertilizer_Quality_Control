"""
NPK regression wrapper.

If a trained regression checkpoint is available (e.g., scikit-learn model saved
with joblib/pickle), it will be loaded. Otherwise, a deterministic heuristic
produces placeholder NPK values so the UI remains functional.
"""

from pathlib import Path
import pickle
from typing import Dict

import numpy as np


def _extract_features(image_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract simple statistics from the image and mask.
    """
    norm_img = image_np.astype(np.float32) / 255.0
    coverage = float((mask > 0).mean())
    mean_channels = norm_img.mean(axis=(0, 1))
    std_channels = norm_img.std(axis=(0, 1))
    return np.concatenate([mean_channels, std_channels, [coverage]])


class NPKPredictor:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            print(f"NPK checkpoint not found at {self.checkpoint_path}. Using heuristic predictions.")
            return

        try:
            with self.checkpoint_path.open("rb") as f:
                self.model = pickle.load(f)
            print("Loaded regression model.")
        except Exception as exc:
            print(f"Failed to load regression model ({exc}). Using heuristic predictions.")
            self.model = None

    def predict(self, image_np: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        features = _extract_features(image_np, mask)
        if self.model is not None and hasattr(self.model, "predict"):
            try:
                preds = self.model.predict(features.reshape(1, -1))[0]
                # Ensure iterable output
                if np.ndim(preds) == 0:
                    preds = [float(preds)] * 3
            except Exception as exc:
                print(f"Regression inference failed ({exc}). Falling back to heuristic.")
                preds = None
        else:
            preds = None

        if preds is None:
            # Simple heuristic: map coverage and color stats into plausible NPK numbers
            N = float(10 + features[0] * 5 + features[-1] * 20)
            P = float(5 + features[1] * 4 + features[-1] * 10)
            K = float(8 + features[2] * 6 + features[-1] * 15)
            preds = [N, P, K]

        # Clip to positive values
        preds = [max(0.0, float(x)) for x in preds[:3]]
        return {"N": preds[0], "P": preds[1], "K": preds[2]}

