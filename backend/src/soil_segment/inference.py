"""
Inference utilities for segmentation.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .model import DummySegmenter, UNet


def preprocess_for_model(image_np: np.ndarray) -> torch.Tensor:
    """
    Convert a 1024x1024 RGB numpy image into a normalized torch tensor.
    """
    tensor = torch.from_numpy(image_np).float() / 255.0  # [H, W, C]
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    return tensor


def _load_state_dict(model: torch.nn.Module, checkpoint: dict) -> None:
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")


def load_segmentation_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a segmentation model checkpoint. Falls back to a dummy model if loading fails.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Checkpoint not found at {path}. Using dummy segmenter.")
        model = DummySegmenter()
        model.to(device)
        model.eval()
        return model

    try:
        checkpoint = torch.load(path, map_location=device)
        # TorchScript module
        if isinstance(checkpoint, torch.jit.ScriptModule):
            model = checkpoint
        else:
            num_classes = 2
            if isinstance(checkpoint, dict):
                num_classes = int(checkpoint.get("num_classes", num_classes))
            model = UNet(num_classes=num_classes)
            _load_state_dict(model, checkpoint)
        model.to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f"Failed to load checkpoint ({exc}). Using dummy segmenter.")
        model = DummySegmenter()
        model.to(device)
        model.eval()
        return model


def _run_model(model: torch.nn.Module, image_np: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Run the model and return logits as numpy array [H, W, C].
    """
    tensor = preprocess_for_model(image_np).to(device)
    with torch.no_grad():
        logits = model(tensor)
    # Ensure output shape [B, C, H, W]
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    logits_np = logits.detach().cpu().numpy()[0]  # [C, H, W]
    return np.transpose(logits_np, (1, 2, 0))  # [H, W, C]


def predict_segmentation(model: torch.nn.Module, image_np: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Generate a mask with integer labels from the segmentation model.
    Background is label 0.
    """
    logits = _run_model(model, image_np, device)
    mask = np.argmax(logits, axis=-1).astype(np.uint8)
    return mask

