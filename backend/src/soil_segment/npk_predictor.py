"""
NPK regression wrapper.

If a trained regression checkpoint is available (e.g., scikit-learn model saved
with joblib/pickle), it will be loaded. Otherwise, a deterministic heuristic
produces placeholder NPK values so the UI remains functional.
"""

from pathlib import Path
import pickle
from typing import Dict

import joblib
import numpy as np

# Class composition mapping (7 classes total, background + 6 pellet types)
CLASS_COMPOSITIONS = {
    1: {"N": 18.0, "P": 45.5, "K": 0.0},   # Black_DAP
    2: {"N": 0.0,  "P": 0.0,  "K": 60.0},  # Red_MOP
    3: {"N": 20.5, "P": 0.0,  "K": 0.0},   # White_AMP (ammonium sulphate)
    4: {"N": 0.0,  "P": 0.0,  "K": 0.0},   # White_Boron (ignored for NPK)
    5: {"N": 0.0,  "P": 0.0,  "K": 0.0},   # White_Mg (ignored for NPK)
    6: {"N": 46.0, "P": 0.0,  "K": 0.0},   # Yellow_Urea
}


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
            # Prefer joblib for sklearn regressors
            self.model = joblib.load(self.checkpoint_path)
            print("Loaded regression model.")
        except Exception as exc:
            try:
                with self.checkpoint_path.open("rb") as f:
                    self.model = pickle.load(f)
                print("Loaded regression model via pickle.")
            except Exception as exc2:
                print(f"Failed to load regression model ({exc}; {exc2}). Using heuristic predictions.")
                self.model = None

    def predict(self, image_np: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        # Pixel counts per class (ignore background=0)
        class_pixels = {cls: int(np.sum(mask == cls)) for cls in CLASS_COMPOSITIONS.keys()}
        total_pixels = sum(class_pixels.values())

        if total_pixels == 0:
            approx_npk = [0.0, 0.0, 0.0]
        else:
            npk_total = {"N": 0.0, "P": 0.0, "K": 0.0}
            for cls, area in class_pixels.items():
                comp = CLASS_COMPOSITIONS.get(cls, {"N": 0.0, "P": 0.0, "K": 0.0})
                for key in npk_total:
                    npk_total[key] += comp[key] * area
            approx_npk = [
                npk_total["N"] / total_pixels,
                npk_total["P"] / total_pixels,
                npk_total["K"] / total_pixels,
            ]

        preds = None
        if self.model is not None and hasattr(self.model, "predict"):
            try:
                preds = self.model.predict(np.array(approx_npk).reshape(1, -1))[0]
            except Exception as exc:
                print(f"Regression inference failed ({exc}). Falling back to approximate NPK.")

        if preds is None:
            preds = approx_npk

        preds = [max(0.0, float(x)) for x in preds[:3]]
        return {"N": preds[0], "P": preds[1], "K": preds[2]}
