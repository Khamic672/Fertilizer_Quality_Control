"""
Soil segmentation and NPK prediction helpers.
"""

from .model import SimpleUNet  # noqa: F401
from .inference import (
    CLASS_COLORS,
    DEFAULT_NUM_CLASSES,
    load_segmentation_model,
    predict_segmentation,
)  # noqa: F401
from .npk_predictor import NPKPredictor  # noqa: F401

__all__ = [
    "SimpleUNet",
    "load_segmentation_model",
    "predict_segmentation",
    "CLASS_COLORS",
    "DEFAULT_NUM_CLASSES",
    "NPKPredictor",
]
