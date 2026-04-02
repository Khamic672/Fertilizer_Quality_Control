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

UNCOATED_ACTIVE_CLASS_IDS = (1, 2, 6)
# Mild uncoated calibration tuned against the recent 14-7-35 debug fractions:
UNCOATED_CLASS_NORMALIZATION_WEIGHTS = {
    1: 1.06,  # Black DAP
    2: 1.12,  # Red MOP
    6: 0.80,  # Yellow Urea
}


def class_fractions_from_mask(
    mask: np.ndarray,
    *,
    class_ids=None,
) -> Dict[int, float]:
    ids = tuple(CLASS_COMPOSITIONS.keys()) if class_ids is None else tuple(class_ids)
    class_pixels = {cls: int(np.sum(mask == cls)) for cls in ids}
    total_pixels = sum(class_pixels.values())
    if total_pixels == 0:
        return {cls: 0.0 for cls in ids}
    return {cls: area / total_pixels for cls, area in class_pixels.items()}


def round_fraction_map(values: Dict[int, float]) -> Dict[int, float]:
    return {int(cls): float(round(float(value), 6)) for cls, value in values.items()}


def npk_from_class_fractions(class_fractions: Dict[int, float]) -> list[float]:
    npk_total = {"N": 0.0, "P": 0.0, "K": 0.0}
    for cls, fraction in class_fractions.items():
        comp = CLASS_COMPOSITIONS.get(cls, {"N": 0.0, "P": 0.0, "K": 0.0})
        for key in npk_total:
            npk_total[key] += comp[key] * float(fraction)
    return [npk_total["N"], npk_total["P"], npk_total["K"]]


def approximate_npk_from_mask(mask: np.ndarray) -> list[float]:
    return npk_from_class_fractions(class_fractions_from_mask(mask))


class NPKPredictor:
    def __init__(self, checkpoint_path: str, strict: bool = False):
        self.checkpoint_path = Path(checkpoint_path)
        self.strict = strict
        self.model = None
        self.load_error = None
        self._load_model()
        if self.strict and self.model is None:
            raise RuntimeError(self.load_error or "Failed to load regression model.")

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            self.load_error = f"NPK checkpoint not found at {self.checkpoint_path}."
            print(f"{self.load_error} Using heuristic predictions.")
            return

        try:
            # Prefer joblib for sklearn regressors
            self.model = joblib.load(self.checkpoint_path)
            self.load_error = None
            print("Loaded regression model.")
        except Exception as exc:
            try:
                with self.checkpoint_path.open("rb") as f:
                    self.model = pickle.load(f)
                self.load_error = None
                print("Loaded regression model via pickle.")
            except Exception as exc2:
                self.load_error = f"Failed to load regression model ({exc}; {exc2})."
                print(f"{self.load_error} Using heuristic predictions.")
                self.model = None

    def predict(self, image_np: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        approx_npk = approximate_npk_from_mask(mask)

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


class ApproximateNPKPredictor:
    def predict(self, image_np: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        preds = approximate_npk_from_mask(mask)
        return {"N": preds[0], "P": preds[1], "K": preds[2]}


class UncoatedNormalizedPredictor:
    def __init__(self, class_weights: Dict[int, float] | None = None):
        self.class_weights = dict(UNCOATED_CLASS_NORMALIZATION_WEIGHTS)
        if class_weights:
            self.class_weights.update({int(k): float(v) for k, v in class_weights.items()})

    def predict(self, image_np: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        fractions = class_fractions_from_mask(mask, class_ids=UNCOATED_ACTIVE_CLASS_IDS)
        weighted = {
            cls: fractions.get(cls, 0.0) * self.class_weights.get(cls, 1.0)
            for cls in UNCOATED_ACTIVE_CLASS_IDS
        }
        total = sum(weighted.values())
        if total <= 0:
            return {"N": 0.0, "P": 0.0, "K": 0.0}

        normalized = {cls: value / total for cls, value in weighted.items()}
        preds = npk_from_class_fractions(normalized)
        return {"N": preds[0], "P": preds[1], "K": preds[2]}

    def debug_info(self, mask: np.ndarray) -> Dict[str, Dict[int, float]]:
        raw_fractions = class_fractions_from_mask(mask, class_ids=UNCOATED_ACTIVE_CLASS_IDS)
        weighted_fractions = {
            cls: raw_fractions.get(cls, 0.0) * self.class_weights.get(cls, 1.0)
            for cls in UNCOATED_ACTIVE_CLASS_IDS
        }
        total = sum(weighted_fractions.values())
        if total <= 0:
            normalized_fractions = {cls: 0.0 for cls in UNCOATED_ACTIVE_CLASS_IDS}
        else:
            normalized_fractions = {
                cls: value / total for cls, value in weighted_fractions.items()
            }

        return {
            "raw_class_fractions": round_fraction_map(raw_fractions),
            "weighted_class_fractions": round_fraction_map(weighted_fractions),
            "normalized_class_fractions": round_fraction_map(normalized_fractions),
            "class_weights": round_fraction_map(self.class_weights),
        }
