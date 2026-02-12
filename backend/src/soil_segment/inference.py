"""
Inference utilities for segmentation.
"""

import contextlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    SEGMENTATION_MODEL_SIZE,
    SEGMENTATION_ONNX_CALIBRATION_DIR,
    SEGMENTATION_ONNX_CALIBRATION_SAMPLES,
    SEGMENTATION_ONNX_EXPORT,
    SEGMENTATION_ONNX_INT8_PATH,
    SEGMENTATION_ONNX_OPSET,
    SEGMENTATION_ONNX_PATH,
    SEGMENTATION_ONNX_QUANTIZE,
    SEGMENTATION_QUANTIZATION,
    SEGMENTATION_RUNTIME,
)
from .model import DummySegmenter, SimpleUNet

# Match the 2D-soil-segment repo defaults (override via SEGMENTATION_MODEL_SIZE)
MODEL_SIZE = SEGMENTATION_MODEL_SIZE
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
_QUANTIZABLE_MODULES = (torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU)
_SUPPORTED_QUANTIZATION_MODES = {"none", "dynamic"}
_SUPPORTED_RUNTIMES = {"torch", "onnx"}
_SUPPORTED_ONNX_EXPORT_MODES = {"auto", "always", "never"}
_SUPPORTED_ONNX_QUANTIZE_MODES = {"none", "int8"}


def _has_quantizable_modules(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, _QUANTIZABLE_MODULES):
            return True
    return False


def _apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    try:
        from torch.ao.quantization import quantize_dynamic
    except Exception:
        try:
            from torch.quantization import quantize_dynamic
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"Dynamic quantization unavailable ({exc}).")
            return model
    return quantize_dynamic(model, set(_QUANTIZABLE_MODULES), dtype=torch.qint8)


def _maybe_quantize_model(
    model: torch.nn.Module, device: torch.device, mode: str
) -> torch.nn.Module:
    if not mode or mode == "none":
        return model
    if mode not in _SUPPORTED_QUANTIZATION_MODES:
        print(f"Unknown quantization mode '{mode}'. Skipping quantization.")
        return model
    if device.type != "cpu":
        print("Quantization requested but device is not CPU. Skipping quantization.")
        return model
    if isinstance(model, torch.jit.ScriptModule):
        print("Quantization requested for TorchScript model. Skipping quantization.")
        return model
    if not _has_quantizable_modules(model):
        print("Quantization requested but no quantizable modules found. Skipping quantization.")
        return model
    try:
        quantized = _apply_dynamic_quantization(model)
    except Exception as exc:
        print(f"Failed to apply dynamic quantization ({exc}). Using float model.")
        return model
    print("Applied dynamic quantization (qint8) to CPU model.")
    return quantized


class OrtSegmenter:
    """Wrapper for ONNX Runtime sessions to align with torch inference."""

    def __init__(self, session, input_name: str, output_name: str):
        self.session = session
        self.input_name = input_name
        self.output_name = output_name

    def run(self, input_array: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        return outputs[0]


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


def _load_torch_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if not path.exists():
        load_error = f"Checkpoint not found at {path}."
        print(f"{load_error} Using dummy segmenter.")
        model = DummySegmenter(num_classes=DEFAULT_NUM_CLASSES)
        setattr(model, "_load_error", load_error)
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
        load_error = f"Failed to load checkpoint ({exc})."
        print(f"{load_error} Using dummy segmenter.")
        model = DummySegmenter(num_classes=DEFAULT_NUM_CLASSES)
        setattr(model, "_load_error", load_error)
        model.to(device)
        model.eval()
        return model


def _export_onnx_model(model: torch.nn.Module, onnx_path: Path, opset: int) -> bool:
    try:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()
        dummy_input = torch.zeros(1, 3, MODEL_SIZE, MODEL_SIZE, dtype=torch.float32)
        torch.onnx.export(
            model_cpu,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        return True
    except Exception as exc:
        print(f"Failed to export ONNX model ({exc}).")
        return False


def _collect_calibration_images(calibration_dir: str, limit: int) -> list[Path]:
    if not calibration_dir:
        return []
    base = Path(calibration_dir)
    if not base.exists():
        return []
    if base.is_file():
        if base.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            return [base]
        return []
    image_paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(base.rglob(ext))
    image_paths = sorted(image_paths)
    if limit > 0:
        image_paths = image_paths[:limit]
    return image_paths


def _quantize_onnx_int8(
    onnx_path: Path,
    int8_path: Path,
    calibration_dir: str,
    max_samples: int,
) -> bool:
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except Exception as exc:
        print(f"ONNX Runtime quantization unavailable ({exc}).")
        return False

    image_paths = _collect_calibration_images(calibration_dir, max_samples)
    if not image_paths:
        print("No calibration images found; skipping INT8 quantization.")
        return False

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
    except Exception:
        input_name = "input"

    class _ImageCalibrationDataReader(CalibrationDataReader):
        def __init__(self, paths: list[Path], name: str):
            self.paths = paths
            self.input_name = name
            self._iter = None

        def get_next(self):
            if self._iter is None:
                self._iter = iter(self._iter_data())
            return next(self._iter, None)

        def _iter_data(self):
            from PIL import Image

            for path in self.paths:
                try:
                    img = Image.open(path).convert("RGB")
                    image_np = np.array(img)
                except Exception:
                    continue
                tensor = preprocess_for_model(image_np, torch.device("cpu"))
                yield {self.input_name: tensor.cpu().numpy()}

    data_reader = _ImageCalibrationDataReader(image_paths, input_name)
    try:
        int8_path.parent.mkdir(parents=True, exist_ok=True)
        quantize_static(
            model_input=str(onnx_path),
            model_output=str(int8_path),
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        return True
    except Exception as exc:
        print(f"Failed to quantize ONNX model ({exc}).")
        return False


def _load_ort_model(onnx_path: Path) -> OrtSegmenter | None:
    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"ONNX Runtime not available ({exc}).")
        return None
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        return OrtSegmenter(session, input_name, output_name)
    except Exception as exc:
        print(f"Failed to create ONNX Runtime session ({exc}).")
        return None


def _load_state_dict(model: torch.nn.Module, checkpoint: dict) -> None:
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")


def load_segmentation_model(
    checkpoint_path: str, device: torch.device
) -> torch.nn.Module | OrtSegmenter:
    """
    Load a segmentation model checkpoint or ONNX Runtime session. Falls back to a dummy model if loading fails.
    """
    quant_mode = SEGMENTATION_QUANTIZATION
    runtime = SEGMENTATION_RUNTIME
    if runtime not in _SUPPORTED_RUNTIMES:
        print(f"Unknown runtime '{runtime}'. Falling back to torch.")
        runtime = "torch"

    if runtime == "onnx":
        export_mode = SEGMENTATION_ONNX_EXPORT
        if export_mode not in _SUPPORTED_ONNX_EXPORT_MODES:
            print(f"Unknown ONNX export mode '{export_mode}'. Using 'auto'.")
            export_mode = "auto"

        quantize_mode = SEGMENTATION_ONNX_QUANTIZE
        if quantize_mode not in _SUPPORTED_ONNX_QUANTIZE_MODES:
            print(f"Unknown ONNX quantization mode '{quantize_mode}'. Using 'none'.")
            quantize_mode = "none"

        onnx_path: Path | None = SEGMENTATION_ONNX_PATH
        int8_path = SEGMENTATION_ONNX_INT8_PATH

        exported = False
        should_export = export_mode == "always" or (
            export_mode == "auto" and not onnx_path.exists()
        )
        if should_export:
            torch_model = _load_torch_model(checkpoint_path, torch.device("cpu"))
            if _export_onnx_model(torch_model, onnx_path, SEGMENTATION_ONNX_OPSET):
                exported = True
            else:
                onnx_path = None

        if export_mode == "never" and (onnx_path is None or not onnx_path.exists()):
            print("ONNX export disabled and no ONNX model found.")
            onnx_path = None

        if onnx_path is not None and onnx_path.exists():
            model_path = onnx_path
            if quantize_mode == "int8":
                if exported or not int8_path.exists():
                    if _quantize_onnx_int8(
                        onnx_path,
                        int8_path,
                        SEGMENTATION_ONNX_CALIBRATION_DIR,
                        SEGMENTATION_ONNX_CALIBRATION_SAMPLES,
                    ):
                        model_path = int8_path
                else:
                    model_path = int8_path
            ort_model = _load_ort_model(model_path)
            if ort_model is not None:
                return ort_model

        print("Falling back to torch model.")

    model = _load_torch_model(checkpoint_path, device)
    model = _maybe_quantize_model(model, device, quant_mode)
    model.eval()
    return model


def _run_model(
    model: torch.nn.Module | OrtSegmenter, image_np: np.ndarray, device: torch.device
) -> np.ndarray:
    """
    Run the model and return logits as numpy array [H, W, C].
    """
    if isinstance(model, OrtSegmenter):
        tensor = preprocess_for_model(image_np, torch.device("cpu"))
        logits_np = model.run(tensor.cpu().numpy())
        if logits_np.ndim == 3:
            logits_np = np.expand_dims(logits_np, axis=0)
        logits_np = logits_np[0]  # [C, H, W]
        return np.transpose(logits_np, (1, 2, 0))  # [H, W, C]

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


def predict_segmentation(
    model: torch.nn.Module | OrtSegmenter, image_np: np.ndarray, device: torch.device
) -> np.ndarray:
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
