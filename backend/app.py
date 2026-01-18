"""
Backward-compatible entrypoint for the Flask API.
"""

from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soil_segment.api.app import app, initialize_models, run_dev

__all__ = ["app", "initialize_models"]


if __name__ == "__main__":
    run_dev()
