"""
Backward-compatible history storage import.
"""

from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soil_segment.storage.history import append_history, load_history

__all__ = ["append_history", "load_history"]
