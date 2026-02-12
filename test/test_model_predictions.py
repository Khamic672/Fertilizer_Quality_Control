from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from soil_segment.inference import load_segmentation_model, predict_segmentation
from soil_segment.model import DummySegmenter
from soil_segment.npk_predictor import NPKPredictor

SEGMENTATION_MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "best_model.pth"
REGRESSION_MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "regression_model.pkl"
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "test" / "20-3-3.JPG"

# Snapshot values captured on 2026-02-12 from current checkpoints.
# If checkpoints are updated, refresh these values.
EXPECTED_SEGMENTATION_COUNTS = {
    0: 13715872,
    1: 523747,
    2: 402135,
    3: 4357036,
    4: 10226,
    5: 206172,
    6: 746668,
}
EXPECTED_REGRESSION_NPK = {
    "N": 19.810791,
    "P": 3.001709,
    "K": 3.073068,
}
PIXEL_COUNT_TOLERANCE_RATIO = 0.05
PIXEL_COUNT_TOLERANCE_MIN = 2_000
NPK_ABS_TOLERANCE = 1.0


def _load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


@pytest.fixture(scope="module")
def test_image() -> np.ndarray:
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip(f"Sample image not found: {SAMPLE_IMAGE_PATH}")
    return _load_image_rgb(SAMPLE_IMAGE_PATH)


@pytest.fixture(scope="module")
def segmentation_mask(test_image: np.ndarray) -> np.ndarray:
    if not SEGMENTATION_MODEL_PATH.exists():
        pytest.skip(f"Segmentation checkpoint not found: {SEGMENTATION_MODEL_PATH}")

    device = torch.device("cpu")
    model = load_segmentation_model(str(SEGMENTATION_MODEL_PATH), device)
    assert not isinstance(model, DummySegmenter), "Loaded fallback DummySegmenter instead of real checkpoint."
    if hasattr(model, "eval"):
        model.eval()
    return predict_segmentation(model, test_image, device)


def test_model_pipeline(test_image: np.ndarray, segmentation_mask: np.ndarray) -> None:
    values, counts = np.unique(segmentation_mask, return_counts=True)
    actual_counts = {int(v): int(c) for v, c in zip(values, counts)}

    assert set(actual_counts) == set(EXPECTED_SEGMENTATION_COUNTS)
    for class_id, expected_count in EXPECTED_SEGMENTATION_COUNTS.items():
        tolerance = max(PIXEL_COUNT_TOLERANCE_MIN, int(expected_count * PIXEL_COUNT_TOLERANCE_RATIO))
        assert abs(actual_counts[class_id] - expected_count) <= tolerance, (
            f"class {class_id} pixel count drifted: got={actual_counts[class_id]}, "
            f"expected={expected_count}, tolerance={tolerance}"
        )
    if not REGRESSION_MODEL_PATH.exists():
        pytest.skip(f"Regression checkpoint not found: {REGRESSION_MODEL_PATH}")

    predictor = NPKPredictor(str(REGRESSION_MODEL_PATH), strict=True)
    prediction = predictor.predict(test_image, segmentation_mask)

    for key, expected_value in EXPECTED_REGRESSION_NPK.items():
        assert prediction[key] == pytest.approx(expected_value, abs=NPK_ABS_TOLERANCE)
