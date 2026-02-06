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
SEGMENTATION_RUNTIME = os.environ.get("SEGMENTATION_RUNTIME", "torch").strip().lower()
SEGMENTATION_MODEL_SIZE = int(os.environ.get("SEGMENTATION_MODEL_SIZE", "512"))
SEGMENTATION_ONNX_PATH = Path(
    os.environ.get("SEGMENTATION_ONNX_PATH", str(MODELS_DIR / "segmentation.onnx"))
)
SEGMENTATION_ONNX_INT8_PATH = Path(
    os.environ.get("SEGMENTATION_ONNX_INT8_PATH", str(MODELS_DIR / "segmentation.int8.onnx"))
)
SEGMENTATION_ONNX_EXPORT = os.environ.get("SEGMENTATION_ONNX_EXPORT", "auto").strip().lower()
SEGMENTATION_ONNX_QUANTIZE = os.environ.get("SEGMENTATION_ONNX_QUANTIZE", "int8").strip().lower()
SEGMENTATION_ONNX_CALIBRATION_DIR = os.environ.get("SEGMENTATION_ONNX_CALIBRATION_DIR", "").strip()
SEGMENTATION_ONNX_CALIBRATION_SAMPLES = int(
    os.environ.get("SEGMENTATION_ONNX_CALIBRATION_SAMPLES", "16")
)
SEGMENTATION_ONNX_OPSET = int(os.environ.get("SEGMENTATION_ONNX_OPSET", "17"))

__all__ = [
    "BASE_DIR",
    "MODELS_DIR",
    "HISTORY_FILE",
    "LOG_DIR",
    "INFERENCE_LOG_FILE",
    "RUNTIME_LOG_FILE",
    "SEGMENTATION_QUANTIZATION",
    "SEGMENTATION_RUNTIME",
    "SEGMENTATION_MODEL_SIZE",
    "SEGMENTATION_ONNX_PATH",
    "SEGMENTATION_ONNX_INT8_PATH",
    "SEGMENTATION_ONNX_EXPORT",
    "SEGMENTATION_ONNX_QUANTIZE",
    "SEGMENTATION_ONNX_CALIBRATION_DIR",
    "SEGMENTATION_ONNX_CALIBRATION_SAMPLES",
    "SEGMENTATION_ONNX_OPSET",
]
