"""FastAPI backend for the local training UI.

Run from the repository root:

    python -m uvicorn training.interface.training_server:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "training"
TRAINING_UI_PATH = TRAINING_ROOT / "interface" / "training_ui.html"
TRAINING_SELECTION_ROOT = TRAINING_ROOT / ".generated"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from training.src.config import get_data_paths

    _PATHS = get_data_paths()
except Exception:
    _PATHS = {
        "unet_dataset": TRAINING_ROOT / "datasets" / "Unet_dataset",
        "regression_dataset": TRAINING_ROOT / "datasets" / "Regression_dataset",
        "checkpoints": TRAINING_ROOT / "trained_models",
    }

UNET_MODULE = "training.src.unet_trainer"
REGRESSION_MODULE = "training.src.regression_trainer"
VALID_DATASET_TYPES = {"unet", "regression"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg"}


app = FastAPI(title="Soil Segment Training Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
@app.get("/training_ui.html", include_in_schema=False)
def training_ui():
    if not TRAINING_UI_PATH.is_file():
        raise HTTPException(404, f"Training UI not found: {TRAINING_UI_PATH}")
    return FileResponse(TRAINING_UI_PATH)


_state: Dict[str, Any] = {
    "running": False,
    "mode": None,
    "metrics": [],
    "log": [],
    "process": None,
    "fold": None,
    "total_folds": None,
    "phase": None,
    "batch": None,
    "total_batches": None,
    "stopped": False,
    "error": None,
    "completed": False,
    "selected_datasets": [],
}
_state_lock = threading.Lock()

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)"
    r"\s+\|\s+Train L:([\d.]+)\s+D:([\d.]+)\s+IoU:([\d.]+)"
    r"\s+\|\s+Val L:([\d.]+)\s+D:([\d.]+)\s+IoU:([\d.]+)"
    r"\s+\|\s+LR:([\d.e+-]+)"
    r".*?(\[BEST\])?"
)
_FOLD_RE = re.compile(r"Fold\s+(\d+)/(\d+)")
_BATCH_RE = re.compile(r"\[BATCH\]\s+(train|val)\s+(\d+)/(\d+)", re.IGNORECASE)
_NOISY_LOG_RE = re.compile(r"^(Training|Validation|Testing|Computing NPK features):")
MAX_LOG_LINES = 400


def _stream_process(proc: subprocess.Popen[str], mode: str) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n\r").strip()
        if not line:
            continue

        m_fold = _FOLD_RE.search(line)
        if m_fold:
            with _state_lock:
                _state["fold"] = int(m_fold.group(1))
                _state["total_folds"] = int(m_fold.group(2))

        m_batch = _BATCH_RE.search(line)
        if m_batch:
            with _state_lock:
                _state["phase"] = m_batch.group(1).lower()
                _state["batch"] = int(m_batch.group(2))
                _state["total_batches"] = int(m_batch.group(3))
            continue

        if _NOISY_LOG_RE.match(line):
            continue

        with _state_lock:
            if not _state["log"] or _state["log"][-1] != line:
                _state["log"].append(line)
            if len(_state["log"]) > MAX_LOG_LINES:
                _state["log"] = _state["log"][-MAX_LOG_LINES:]

        m_ep = _EPOCH_RE.search(line)
        if m_ep:
            entry = {
                "epoch": int(m_ep.group(1)),
                "max_epoch": int(m_ep.group(2)),
                "train_loss": float(m_ep.group(3)),
                "train_dice": float(m_ep.group(4)),
                "train_iou": float(m_ep.group(5)),
                "val_loss": float(m_ep.group(6)),
                "val_dice": float(m_ep.group(7)),
                "val_iou": float(m_ep.group(8)),
                "lr": float(m_ep.group(9)),
                "best": bool(m_ep.group(10)),
                "fold": _state.get("fold"),
            }
            with _state_lock:
                _state["metrics"].append(entry)
                _state["phase"] = None
                _state["batch"] = None
                _state["total_batches"] = None

    proc.wait()
    with _state_lock:
        _state["running"] = False
        _state["process"] = None
        _state["phase"] = None
        _state["batch"] = None
        _state["total_batches"] = None
        _state["completed"] = proc.returncode == 0
        if proc.returncode not in (0, -9, -15):
            _state["error"] = f"Process exited with code {proc.returncode}"


class UNetParams(BaseModel):
    uncoated: bool = False
    datasets: List[str] = Field(default_factory=list)
    seed: int = 42
    batch_size: int = 4
    img_size: int = 512
    epochs: int = 300
    patience: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patch_size: int = 512
    patches_per_image: int = 32
    uncoated_batch_size: int = 4
    uncoated_epochs: int = 160
    uncoated_patience: int = 30
    uncoated_learning_rate: float = 1e-4
    uncoated_weight_decay: float = 1e-4
    uncoated_init_checkpoint: Optional[str] = None


class RegressionParams(BaseModel):
    datasets: List[str] = Field(default_factory=list)
    checkpoint: Optional[str] = None
    uncoated: bool = False
    img_size: int = 512
    degree: int = 1
    alpha: float = 1.0
    no_ridge: bool = False
    no_scaling: bool = False
    test_size: float = 0.2
    seed: int = 42
    max_images: Optional[int] = None
    cpu: bool = False


def _build_unet_cmd(params: UNetParams, dataset_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        UNET_MODULE,
        "--seed",
        str(params.seed),
        "--dataset",
        str(dataset_path),
    ]
    if params.uncoated:
        cmd += [
            "--uncoated",
            "--uncoated-patch-size",
            str(params.patch_size),
            "--uncoated-patches-per-image",
            str(params.patches_per_image),
            "--uncoated-batch-size",
            str(params.uncoated_batch_size),
            "--uncoated-epochs",
            str(params.uncoated_epochs),
            "--uncoated-patience",
            str(params.uncoated_patience),
            "--uncoated-learning-rate",
            str(params.uncoated_learning_rate),
            "--uncoated-weight-decay",
            str(params.uncoated_weight_decay),
        ]
        if params.uncoated_init_checkpoint:
            cmd += ["--uncoated-init-checkpoint", params.uncoated_init_checkpoint]
        return cmd

    return cmd + [
        "--batch-size",
        str(params.batch_size),
        "--img-size",
        str(params.img_size),
        "--epochs",
        str(params.epochs),
        "--patience",
        str(params.patience),
        "--learning-rate",
        str(params.learning_rate),
        "--weight-decay",
        str(params.weight_decay),
    ]


def _build_regression_cmd(params: RegressionParams, dataset_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        REGRESSION_MODULE,
        "--dataset",
        str(dataset_path),
        "--img-size",
        str(params.img_size),
        "--degree",
        str(params.degree),
        "--alpha",
        str(params.alpha),
        "--test-size",
        str(params.test_size),
        "--seed",
        str(params.seed),
    ]
    checkpoint_name = _single_file_name(params.checkpoint, label="Checkpoint name")
    if checkpoint_name:
        checkpoint_path = Path(_PATHS["checkpoints"]) / checkpoint_name
        if not checkpoint_path.is_file():
            raise HTTPException(400, f"Checkpoint not found: {checkpoint_name}")
        cmd += ["--checkpoint", str(checkpoint_path)]
    if params.uncoated:
        cmd.append("--uncoated")
    if params.no_ridge:
        cmd.append("--no-ridge")
    if params.no_scaling:
        cmd.append("--no-scaling")
    if params.cpu:
        cmd.append("--cpu")
    if params.max_images is not None:
        cmd += ["--max-images", str(params.max_images)]
    return cmd


def _build_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    root = str(PROJECT_ROOT)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not current else root + os.pathsep + current
    return env


def _count_dir(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def _count_named_files(path: Path, extensions: set[str]) -> int:
    if not path.is_dir():
        return 0
    return sum(
        1
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in extensions
    )


def _list_formula_datasets(base: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not base.exists():
        return rows

    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue

        images = _count_named_files(child / "images", IMAGE_EXTENSIONS)
        masks = _count_named_files(child / "masks", MASK_EXTENSIONS)
        rows.append(
            {
                "name": child.name,
                "base": str(child),
                "images": images,
                "masks": masks,
                "exists": child.exists(),
                "trainable": images > 0 and masks > 0,
                "uncoated": "uncoated" in child.name.lower(),
            }
        )
    return rows


def _list_regression_datasets(base: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not base.exists():
        return rows

    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue

        images = _count_named_files(child, IMAGE_EXTENSIONS)
        rows.append(
            {
                "name": child.name,
                "base": str(child),
                "images": images,
                "exists": child.exists(),
            }
        )
    return rows


def _validate_relative_path(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise HTTPException(400, "Upload archive contains an invalid path.")
    return path


def _single_folder_name(raw_value: Optional[str], *, label: str) -> Optional[str]:
    if raw_value is None or not raw_value.strip():
        return None
    value = Path(raw_value.strip())
    value = _validate_relative_path(value)
    if len(value.parts) != 1:
        raise HTTPException(
            400,
            f"{label} must be a single folder name like 15-15-15.",
        )
    return value.name


def _single_file_name(raw_value: Optional[str], *, label: str) -> Optional[str]:
    if raw_value is None or not raw_value.strip():
        return None
    value = Path(raw_value.strip())
    value = _validate_relative_path(value)
    if len(value.parts) != 1:
        raise HTTPException(400, f"{label} must be a single file name.")
    return value.name


def _resolve_selected_unet_datasets(selected: List[str]) -> List[str]:
    datasets = _list_formula_datasets(Path(_PATHS["unet_dataset"]))
    available = {item["name"]: item for item in datasets if item["trainable"]}
    known = {item["name"] for item in datasets}

    if not available:
        raise HTTPException(400, "No trainable UNet datasets are available.")

    if not selected:
        return sorted(available)

    resolved: List[str] = []
    seen: set[str] = set()
    missing: List[str] = []
    incomplete: List[str] = []

    for raw_name in selected:
        name = _single_folder_name(raw_name, label="Dataset name")
        if not name or name in seen:
            continue
        seen.add(name)
        if name in available:
            resolved.append(name)
        elif name in known:
            incomplete.append(name)
        else:
            missing.append(name)

    if missing:
        raise HTTPException(400, f"Unknown dataset(s): {', '.join(missing)}")
    if incomplete:
        raise HTTPException(
            400,
            f"Dataset(s) are missing images or masks: {', '.join(incomplete)}",
        )
    if not resolved:
        raise HTTPException(400, "Select at least one dataset to train.")
    return resolved


def _resolve_mode_training_datasets(
    selected: List[str],
    *,
    uncoated: bool,
) -> tuple[List[str], List[str]]:
    resolved = _resolve_selected_unet_datasets(selected)
    dataset_meta = {
        item["name"]: item for item in _list_formula_datasets(Path(_PATHS["unet_dataset"]))
    }

    effective: List[str] = []
    ignored: List[str] = []
    for name in resolved:
        is_uncoated = bool(dataset_meta.get(name, {}).get("uncoated"))
        if is_uncoated == uncoated:
            effective.append(name)
        else:
            ignored.append(name)

    if not effective:
        mode_label = "uncoated" if uncoated else "coated"
        raise HTTPException(400, f"Select at least one {mode_label} dataset to train.")

    return effective, ignored


def _prepare_unet_selection_root(dataset_names: List[str]) -> Path:
    source_root = Path(_PATHS["unet_dataset"])
    selection_root = TRAINING_SELECTION_ROOT / "unet_selected"
    if selection_root.exists():
        shutil.rmtree(selection_root)

    images_dir = selection_root / "images"
    masks_dir = selection_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_masks = 0
    for dataset_name in dataset_names:
        dataset_dir = source_root / dataset_name
        prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name).strip("_") or "dataset"

        for image_file in sorted((dataset_dir / "images").iterdir()):
            if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(image_file, images_dir / f"{prefix}__{image_file.name}")
                copied_images += 1

        for mask_file in sorted((dataset_dir / "masks").iterdir()):
            if mask_file.is_file() and mask_file.suffix.lower() in MASK_EXTENSIONS:
                shutil.copy2(mask_file, masks_dir / f"{prefix}__{mask_file.name}")
                copied_masks += 1

    if copied_images == 0 or copied_masks == 0:
        raise HTTPException(400, "The selected datasets do not contain trainable image/mask pairs.")

    return selection_root


def _prepare_regression_selection_root(dataset_names: List[str]) -> Path:
    source_root = Path(_PATHS["unet_dataset"])
    selection_root = TRAINING_SELECTION_ROOT / "regression_selected"
    if selection_root.exists():
        shutil.rmtree(selection_root)

    selection_root.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    for dataset_name in dataset_names:
        source_images_dir = source_root / dataset_name / "images"
        target_dir = selection_root / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_file in sorted(source_images_dir.iterdir()):
            if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(image_file, target_dir / image_file.name)
                copied_images += 1

    if copied_images == 0:
        raise HTTPException(400, "The selected datasets do not contain regression source images.")

    return selection_root


@app.get("/api/dataset/info")
def dataset_info():
    unet_root = Path(_PATHS["unet_dataset"])
    regression_root = Path(_PATHS["regression_dataset"])
    return {
        "unet_root": str(unet_root),
        "regression_root": str(regression_root),
        "datasets": _list_formula_datasets(unet_root),
        "regression_datasets": _list_regression_datasets(regression_root),
        "checkpoints": {
            "base": str(_PATHS["checkpoints"]),
            "files": [
                item.name
                for item in sorted(Path(_PATHS["checkpoints"]).iterdir())
                if item.is_file()
            ]
            if Path(_PATHS["checkpoints"]).exists()
            else [],
        },
    }


@app.post("/api/dataset/upload")
async def upload_dataset(
    files: List[UploadFile] = File(...),
    dataset_type: str = Form("unet"),
    dataset_name: Optional[str] = Form(None),
    file_role: str = Form("images"),
):
    if dataset_type not in VALID_DATASET_TYPES:
        raise HTTPException(400, f"Unsupported dataset type: {dataset_type}")

    is_regression = dataset_type == "regression"
    if file_role not in {"images", "masks"}:
        raise HTTPException(400, f"Unsupported file role: {file_role}")
    if is_regression and file_role == "masks":
        raise HTTPException(400, "Regression datasets accept only image files.")

    base_path = Path(_PATHS["regression_dataset"] if is_regression else _PATHS["unet_dataset"])
    base_path.mkdir(parents=True, exist_ok=True)

    dataset_folder = _single_folder_name(
        dataset_name,
        label="Dataset name",
    )
    if not dataset_folder:
        raise HTTPException(400, "Choose or type a dataset name before uploading.")

    dataset_dir = base_path / dataset_folder
    target_dir = dataset_dir if is_regression else (dataset_dir / file_role)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []

    for upload in files:
        data = await upload.read()
        filename = upload.filename or "file"

        if filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for member in archive.infolist():
                        if member.is_dir():
                            continue
                        rel_path = _validate_relative_path(Path(member.filename))
                        if is_regression:
                            dest = dataset_dir / (Path(*rel_path.parts) if len(rel_path.parts) > 1 else rel_path.name)
                        else:
                            if len(rel_path.parts) < 2 or rel_path.parts[0] not in {"images", "masks"}:
                                continue
                            dest = dataset_dir / rel_path.parts[0] / Path(*rel_path.parts[1:])

                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(archive.read(member))
                        saved.append(str(dest.relative_to(dataset_dir)))
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, f"Bad zip: {exc}") from exc
            continue

        if is_regression:
            dest = dataset_dir / filename
        else:
            dest = dataset_dir / file_role / filename

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        saved.append(str(dest.relative_to(dataset_dir)))

    return {
        "saved": saved,
        "target": str(target_dir),
        "dataset": dataset_folder,
    }


@app.get("/api/train/status")
def train_status():
    with _state_lock:
        return {
            "running": _state["running"],
            "mode": _state["mode"],
            "fold": _state["fold"],
            "total_folds": _state["total_folds"],
            "phase": _state["phase"],
            "batch": _state["batch"],
            "total_batches": _state["total_batches"],
            "metrics": list(_state["metrics"]),
            "log": list(_state["log"]),
            "completed": _state["completed"],
            "error": _state["error"],
            "selected_datasets": list(_state["selected_datasets"]),
        }


@app.post("/api/train/unet/start")
def start_unet(params: UNetParams):
    selected_datasets, ignored_datasets = _resolve_mode_training_datasets(
        params.datasets,
        uncoated=params.uncoated,
    )
    dataset_path = _prepare_unet_selection_root(selected_datasets)

    with _state_lock:
        if _state["running"]:
            raise HTTPException(409, "Training already running")
        _state.update(
            {
                "running": True,
                "mode": "unet_uncoated" if params.uncoated else "unet",
                "metrics": [],
                "log": [
                    f"[DATASET] Using {len(selected_datasets)} selected dataset(s)",
                    f"[DATASET] {', '.join(selected_datasets)}",
                ]
                + (
                    [f"[DATASET] Ignored mode-mismatched datasets: {', '.join(ignored_datasets)}"]
                    if ignored_datasets
                    else []
                ),
                "fold": None,
                "total_folds": None,
                "phase": None,
                "batch": None,
                "total_batches": None,
                "stopped": False,
                "error": None,
                "completed": False,
                "selected_datasets": selected_datasets,
            }
        )

    cmd = _build_unet_cmd(params, dataset_path)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env=_build_subprocess_env(),
    )
    with _state_lock:
        _state["process"] = proc

    threading.Thread(target=_stream_process, args=(proc, "unet"), daemon=True).start()
    return {"started": True, "cmd": " ".join(cmd), "datasets": selected_datasets}


@app.post("/api/train/regression/start")
def start_regression(params: RegressionParams):
    selected_datasets, ignored_datasets = _resolve_mode_training_datasets(
        params.datasets,
        uncoated=params.uncoated,
    )
    dataset_path = _prepare_regression_selection_root(selected_datasets)

    with _state_lock:
        if _state["running"]:
            raise HTTPException(409, "Training already running")
        _state.update(
            {
                "running": True,
                "mode": "regression",
                "metrics": [],
                "log": [
                    f"[DATASET] Using {len(selected_datasets)} selected dataset(s)",
                    f"[DATASET] {', '.join(selected_datasets)}",
                ]
                + (
                    [f"[DATASET] Ignored mode-mismatched datasets: {', '.join(ignored_datasets)}"]
                    if ignored_datasets
                    else []
                ),
                "fold": None,
                "total_folds": None,
                "phase": None,
                "batch": None,
                "total_batches": None,
                "stopped": False,
                "error": None,
                "completed": False,
                "selected_datasets": selected_datasets,
            }
        )

    cmd = _build_regression_cmd(params, dataset_path)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env=_build_subprocess_env(),
    )
    with _state_lock:
        _state["process"] = proc

    threading.Thread(target=_stream_process, args=(proc, "regression"), daemon=True).start()
    return {"started": True, "cmd": " ".join(cmd), "datasets": selected_datasets}


@app.post("/api/train/stop")
def stop_training():
    with _state_lock:
        proc = _state.get("process")
        if proc is None or not _state["running"]:
            raise HTTPException(400, "No training process is running")
        proc.terminate()
        _state["running"] = False
        _state["stopped"] = True
        _state["process"] = None
        _state["phase"] = None
        _state["batch"] = None
        _state["total_batches"] = None
    return {"stopped": True}


@app.delete("/api/train/clear")
def clear_metrics():
    with _state_lock:
        _state["metrics"] = []
        _state["log"] = []
        _state["error"] = None
        _state["completed"] = False
        _state["selected_datasets"] = []
        _state["phase"] = None
        _state["batch"] = None
        _state["total_batches"] = None
    return {"cleared": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("training.interface.training_server:app", host="0.0.0.0", port=8000, reload=True)
