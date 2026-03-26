"""FastAPI backend for the local training UI.

Run from the repository root:

    python -m uvicorn training.interface.training_server:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "training"
TRAINING_UI_PATH = TRAINING_ROOT / "interface" / "training_ui.html"

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
VALID_DATASET_TYPES = {"unet", "unet_uncoated", "regression", "regression_uncoated"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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
    "stopped": False,
    "error": None,
    "completed": False,
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
MAX_LOG_LINES = 400


def _stream_process(proc: subprocess.Popen[str], mode: str) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n\r")
        with _state_lock:
            _state["log"].append(line)
            if len(_state["log"]) > MAX_LOG_LINES:
                _state["log"] = _state["log"][-MAX_LOG_LINES:]

        m_fold = _FOLD_RE.search(line)
        if m_fold:
            with _state_lock:
                _state["fold"] = int(m_fold.group(1))
                _state["total_folds"] = int(m_fold.group(2))

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

    proc.wait()
    with _state_lock:
        _state["running"] = False
        _state["process"] = None
        _state["completed"] = proc.returncode == 0
        if proc.returncode not in (0, -9, -15):
            _state["error"] = f"Process exited with code {proc.returncode}"


class UNetParams(BaseModel):
    uncoated: bool = False
    seed: int = 42
    batch_size: int = 4
    img_size: int = 512
    epochs: int = 300
    patience: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patch_size: int = 512
    patches_per_image: int = 64
    uncoated_batch_size: int = 4
    uncoated_epochs: int = 160
    uncoated_patience: int = 30
    uncoated_learning_rate: float = 1e-4
    uncoated_weight_decay: float = 1e-4
    uncoated_init_checkpoint: Optional[str] = None


class RegressionParams(BaseModel):
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


def _build_unet_cmd(params: UNetParams) -> List[str]:
    cmd = [sys.executable, "-m", UNET_MODULE, "--seed", str(params.seed)]
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


def _build_regression_cmd(params: RegressionParams) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        REGRESSION_MODULE,
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


def _regression_groups(base: Path) -> List[List[Path]]:
    if not base.exists():
        return []

    groups: List[List[Path]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue

        image_files = [
            item
            for item in sorted(child.iterdir())
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_files:
            images_dir = child / "images"
            if images_dir.is_dir():
                image_files = [
                    item
                    for item in sorted(images_dir.iterdir())
                    if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
                ]
        if image_files:
            groups.append(image_files)
    return groups


def _dataset_info(base: Path, *, kind: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "base": str(base),
        "exists": base.exists(),
        "kind": kind,
    }
    if kind == "unet":
        info["images"] = _count_dir(base / "images")
        info["masks"] = _count_dir(base / "masks")
        return info

    groups = _regression_groups(base)
    info["images"] = sum(len(group) for group in groups)
    info["formulas"] = len(groups)
    return info


def _validate_relative_path(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise HTTPException(400, "Upload archive contains an invalid path.")
    return path


def _formula_dir_path(raw_value: Optional[str]) -> Optional[Path]:
    if raw_value is None or not raw_value.strip():
        return None
    value = Path(raw_value.strip())
    value = _validate_relative_path(value)
    if len(value.parts) != 1:
        raise HTTPException(400, "Formula directory must be a single folder name like 15-15-15.")
    return value


@app.get("/api/dataset/info")
def dataset_info():
    return {
        "unet": _dataset_info(Path(_PATHS["unet_dataset"]), kind="unet"),
        "unet_uncoated": _dataset_info(
            Path(str(_PATHS["unet_dataset"]) + "_uncoated"), kind="unet"
        ),
        "regression": _dataset_info(Path(_PATHS["regression_dataset"]), kind="regression"),
        "regression_uncoated": _dataset_info(
            Path(str(_PATHS["regression_dataset"]) + "_uncoated"), kind="regression"
        ),
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
    file_role: str = Form("images"),
    formula_dir: Optional[str] = Form(None),
):
    if dataset_type not in VALID_DATASET_TYPES:
        raise HTTPException(400, f"Unsupported dataset type: {dataset_type}")

    is_regression = "regression" in dataset_type
    suffix = "_uncoated" if "uncoated" in dataset_type else ""
    if file_role not in {"images", "masks"}:
        raise HTTPException(400, f"Unsupported file role: {file_role}")
    if is_regression and file_role == "masks":
        raise HTTPException(400, "Regression datasets accept only image files.")

    base_path = Path(_PATHS["regression_dataset"] if is_regression else _PATHS["unet_dataset"])
    base_path = Path(str(base_path) + suffix)
    base_path.mkdir(parents=True, exist_ok=True)

    formula_path = _formula_dir_path(formula_dir) if is_regression else None
    target_dir = base_path if is_regression else (base_path / file_role)
    if formula_path is not None:
        target_dir = base_path / formula_path
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
                            if len(rel_path.parts) == 1:
                                if formula_path is None:
                                    raise HTTPException(
                                        400,
                                        "Loose regression files need a formula directory like 15-15-15.",
                                    )
                                dest = base_path / formula_path / rel_path.name
                            else:
                                dest = base_path / rel_path
                        else:
                            if len(rel_path.parts) < 2 or rel_path.parts[0] not in {"images", "masks"}:
                                continue
                            dest = base_path / rel_path.parts[0] / Path(*rel_path.parts[1:])

                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(archive.read(member))
                        saved.append(str(dest.relative_to(base_path)))
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, f"Bad zip: {exc}") from exc
            continue

        if is_regression:
            if formula_path is None:
                raise HTTPException(
                    400,
                    "Regression uploads need a formula directory like 15-15-15 for loose files.",
                )
            dest = base_path / formula_path / filename
        else:
            dest = base_path / file_role / filename

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        saved.append(str(dest.relative_to(base_path)))

    return {
        "saved": saved,
        "target": str(target_dir),
    }


@app.get("/api/train/status")
def train_status():
    with _state_lock:
        return {
            "running": _state["running"],
            "mode": _state["mode"],
            "fold": _state["fold"],
            "total_folds": _state["total_folds"],
            "metrics": list(_state["metrics"]),
            "log": list(_state["log"]),
            "completed": _state["completed"],
            "error": _state["error"],
        }


@app.post("/api/train/unet/start")
def start_unet(params: UNetParams):
    with _state_lock:
        if _state["running"]:
            raise HTTPException(409, "Training already running")
        _state.update(
            {
                "running": True,
                "mode": "unet_uncoated" if params.uncoated else "unet",
                "metrics": [],
                "log": [],
                "fold": None,
                "total_folds": None,
                "stopped": False,
                "error": None,
                "completed": False,
            }
        )

    cmd = _build_unet_cmd(params)
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
    return {"started": True, "cmd": " ".join(cmd)}


@app.post("/api/train/regression/start")
def start_regression(params: RegressionParams):
    with _state_lock:
        if _state["running"]:
            raise HTTPException(409, "Training already running")
        _state.update(
            {
                "running": True,
                "mode": "regression",
                "metrics": [],
                "log": [],
                "fold": None,
                "total_folds": None,
                "stopped": False,
                "error": None,
                "completed": False,
            }
        )

    cmd = _build_regression_cmd(params)
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
    return {"started": True, "cmd": " ".join(cmd)}


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
    return {"stopped": True}


@app.delete("/api/train/clear")
def clear_metrics():
    with _state_lock:
        _state["metrics"] = []
        _state["log"] = []
        _state["error"] = None
        _state["completed"] = False
    return {"cleared": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("training.interface.training_server:app", host="0.0.0.0", port=8000, reload=True)
