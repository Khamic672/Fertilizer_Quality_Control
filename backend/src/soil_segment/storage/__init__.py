"""
Storage helpers for backend persistence.
"""

from .history import append_history, load_history  # noqa: F401

__all__ = ["append_history", "load_history"]
