"""
Shared backend paths.
"""

from pathlib import Path

# backend/src/soil_segment/config.py -> backend/
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
HISTORY_FILE = BASE_DIR / "history.csv"

__all__ = ["BASE_DIR", "MODELS_DIR", "HISTORY_FILE"]
