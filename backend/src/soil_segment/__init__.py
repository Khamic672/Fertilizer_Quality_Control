"""
Soil segmentation and NPK prediction helpers.

This package is intentionally lightweight so the app can run locally with
either the provided checkpoints or a fallback heuristic pipeline.
"""

from .model import UNet  # noqa: F401
from .inference import load_segmentation_model, predict_segmentation  # noqa: F401
from .npk_predictor import NPKPredictor  # noqa: F401

__all__ = [
    "UNet",
    "load_segmentation_model",
    "predict_segmentation",
    "NPKPredictor",
]
