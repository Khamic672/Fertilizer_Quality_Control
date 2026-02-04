"""
Shared backend paths.
"""

import os
from pathlib import Path

# backend/src/soil_segment/config.py -> backend/
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
HISTORY_FILE = BASE_DIR / "history.csv"
LOG_DIR = BASE_DIR / "logs"
INFERENCE_LOG_FILE = LOG_DIR / "inference.log"
RUNTIME_LOG_FILE = LOG_DIR / "runtime.log"
SEGMENTATION_QUANTIZATION = os.environ.get("SEGMENTATION_QUANTIZATION", "none").strip().lower()

__all__ = [
    "BASE_DIR",
    "MODELS_DIR",
    "HISTORY_FILE",
    "LOG_DIR",
    "INFERENCE_LOG_FILE",
    "RUNTIME_LOG_FILE",
    "SEGMENTATION_QUANTIZATION",
]
