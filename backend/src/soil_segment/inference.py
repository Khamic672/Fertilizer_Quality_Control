"""
Inference utilities for segmentation.
"""

import contextlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model import DummySegmenter, SimpleUNet

# Match the 2D-soil-segment repo defaults
MODEL_SIZE = 1024
DEFAULT_NUM_CLASSES = 7
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Palette from DEFAULT_CLASS_COLORS in the original repo
CLASS_COLORS = [
    (0, 0, 0),        # background - black
    (45, 42, 50),     # Black_DAP - graphite
    (225, 29, 72),    # Red_MOP - rose
    (56, 189, 248),   # White_AMP - sky blue
    (244, 114, 182),  # White_Boron - pink
    (34, 197, 94),    # White_Mg - green
    (245, 158, 11),   # Yellow_Urea - amber
]

_NORMALIZATION_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def _normalization_tensors(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = str(device)
    cached = _NORMALIZATION_CACHE.get(key)
    if cached is not None:
        return cached
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    _NORMALIZATION_CACHE[key] = (mean, std)
    return mean, std


def preprocess_for_model(image_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert an RGB numpy image into a normalized torch tensor resized for the model.
    """
    tensor = torch.from_numpy(image_np).float() / 255.0  # [H, W, C]
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

    if tensor.shape[-2:] != (MODEL_SIZE, MODEL_SIZE):
        tensor = F.interpolate(tensor, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False)

    tensor = tensor.to(device, non_blocking=True)
    mean, std = _normalization_tensors(device)
    tensor = (tensor - mean) / std
    return tensor


def _load_state_dict(model: torch.nn.Module, checkpoint: dict) -> None:
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
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
        model = DummySegmenter(num_classes=DEFAULT_NUM_CLASSES)
        model.to(device)
        model.eval()
        return model

    try:
        checkpoint = torch.load(path, map_location=device)
        # TorchScript module
        if isinstance(checkpoint, torch.jit.ScriptModule):
            model = checkpoint
        else:
            num_classes = DEFAULT_NUM_CLASSES
            if isinstance(checkpoint, dict):
                num_classes = int(checkpoint.get("num_classes", num_classes))
            model = SimpleUNet(n_classes=num_classes)
            _load_state_dict(model, checkpoint)
        model.to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f"Failed to load checkpoint ({exc}). Using dummy segmenter.")
        model = DummySegmenter(num_classes=DEFAULT_NUM_CLASSES)
        model.to(device)
        model.eval()
        return model


def _run_model(model: torch.nn.Module, image_np: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Run the model and return logits as numpy array [H, W, C].
    """
    tensor = preprocess_for_model(image_np, device)
    with torch.inference_mode():
        if device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda")
        else:
            autocast_ctx = contextlib.nullcontext()
        with autocast_ctx:
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
    The returned mask is resized back to the original image resolution.
    """
    logits = _run_model(model, image_np, device)
    mask = np.argmax(logits, axis=-1).astype(np.uint8)

    # Resize mask back to original image size if needed
    if mask.shape != image_np.shape[:2]:
        from PIL import Image

        mask_img = Image.fromarray(mask, mode="L")
        mask_img = mask_img.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
        mask = np.array(mask_img, dtype=np.uint8)

    return mask
