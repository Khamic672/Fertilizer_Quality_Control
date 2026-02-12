"""
Flask Backend API for Fertilizer Quality Control
Exposes endpoints for image processing and NPK prediction
"""
import base64
import csv
import datetime
import io
import logging
import os
import hashlib
import sys
import threading
import time
from openpyxl import Workbook

import numpy as np
import torch
from PIL import Image
from flask import Flask, g, jsonify, request, send_file
from flask_cors import CORS
from logging.handlers import RotatingFileHandler
from soil_segment.config import (
    HISTORY_FILE,
    INFERENCE_LOG_FILE,
    LOG_DIR,
    MODELS_DIR,
    RUNTIME_LOG_FILE,
    SEGMENTATION_MODEL_SIZE,
)
from soil_segment.storage import append_history, delete_history_by_id, load_history

from soil_segment.inference import (
    CLASS_COLORS,
    load_segmentation_model,
    predict_segmentation,
)
from soil_segment.model import DummySegmenter
from soil_segment.npk_predictor import NPKPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Vue frontend

inference_logger = logging.getLogger("soil_segment.inference")
if not inference_logger.handlers:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(INFERENCE_LOG_FILE, maxBytes=5_000_000, backupCount=3)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    inference_logger.addHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    inference_logger.addHandler(console_handler)
    inference_logger.setLevel(logging.INFO)
    inference_logger.propagate = False

runtime_logger = logging.getLogger("soil_segment.runtime")
if not runtime_logger.handlers:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    runtime_handler = RotatingFileHandler(RUNTIME_LOG_FILE, maxBytes=5_000_000, backupCount=3)
    runtime_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    runtime_logger.addHandler(runtime_handler)
    runtime_console = logging.StreamHandler()
    runtime_console.setFormatter(logging.Formatter("%(message)s"))
    runtime_logger.addHandler(runtime_console)
    runtime_logger.setLevel(logging.INFO)
    runtime_logger.propagate = False

# Global variables for models
unet_model = None
npk_predictor = None
device = None
history_items = []
history_counter = 0
HISTORY_CSV = HISTORY_FILE

CHECKPOINT_DIR = MODELS_DIR
UNET_CHECKPOINT = CHECKPOINT_DIR / "best_model.pth"
REGRESSION_CHECKPOINT = CHECKPOINT_DIR / "regression_model.pkl"
TARGET_SIZE = (SEGMENTATION_MODEL_SIZE, SEGMENTATION_MODEL_SIZE)
APP_START_MONO = time.monotonic()
_REQUEST_LOCK = threading.Lock()
_REQUEST_COUNT = 0
_REQUESTS_IN_FLIGHT = 0
_ENDPOINT_COUNTS = {}
_CACHE_LOCK = threading.Lock()
_INFERENCE_CACHE = {}
_INFERENCE_CACHE_ORDER = []
_INFERENCE_CACHE_MAX_ITEMS = int(os.environ.get("INFERENCE_CACHE_MAX_ITEMS", "32"))
_INFERENCE_CACHE_HITS = 0
_INFERENCE_CACHE_MISSES = 0


def _cache_get(key: str):
    global _INFERENCE_CACHE_HITS, _INFERENCE_CACHE_MISSES
    with _CACHE_LOCK:
        if key not in _INFERENCE_CACHE:
            _INFERENCE_CACHE_MISSES += 1
            return None
        _INFERENCE_CACHE_ORDER.remove(key)
        _INFERENCE_CACHE_ORDER.append(key)
        _INFERENCE_CACHE_HITS += 1
        return _INFERENCE_CACHE[key]


def _cache_set(key: str, value: dict) -> None:
    with _CACHE_LOCK:
        if key in _INFERENCE_CACHE:
            _INFERENCE_CACHE_ORDER.remove(key)
        _INFERENCE_CACHE[key] = value
        _INFERENCE_CACHE_ORDER.append(key)
        while len(_INFERENCE_CACHE_ORDER) > _INFERENCE_CACHE_MAX_ITEMS:
            oldest = _INFERENCE_CACHE_ORDER.pop(0)
            _INFERENCE_CACHE.pop(oldest, None)


def _record_endpoint_count(endpoint_name: str) -> int:
    global _REQUEST_COUNT, _REQUESTS_IN_FLIGHT
    with _REQUEST_LOCK:
        _REQUEST_COUNT += 1
        _REQUESTS_IN_FLIGHT += 1
        _ENDPOINT_COUNTS[endpoint_name] = _ENDPOINT_COUNTS.get(endpoint_name, 0) + 1
        return _ENDPOINT_COUNTS[endpoint_name]


def _finish_request_count() -> int:
    global _REQUESTS_IN_FLIGHT
    with _REQUEST_LOCK:
        _REQUESTS_IN_FLIGHT = max(0, _REQUESTS_IN_FLIGHT - 1)
        return _REQUESTS_IN_FLIGHT


def _memory_rss_mb() -> float:
    """Best-effort RSS (MB) without optional deps."""
    try:
        if os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm", "r", encoding="utf-8") as handle:
                parts = handle.read().strip().split()
            if len(parts) > 1:
                rss_pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (rss_pages * page_size) / (1024 * 1024)
    except Exception:
        pass

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024.0
    except Exception:
        return 0.0


def _request_size_bytes() -> int:
    if request.content_length is not None:
        return int(request.content_length)
    return 0


def _validate_required_model_files() -> None:
    required_models = {
        "UNet segmentation model": UNET_CHECKPOINT,
        "NPK regression model": REGRESSION_CHECKPOINT,
    }
    missing = []
    for model_name, model_path in required_models.items():
        if not model_path.exists():
            runtime_logger.error("%s checkpoint is missing: %s", model_name, model_path)
            missing.append(f"{model_name} ({model_path})")

    if missing:
        raise RuntimeError(f"Missing required model checkpoint(s): {', '.join(missing)}")


def initialize_models():
    """Load models on startup"""
    global unet_model, npk_predictor, device

    # Reset globals so partial initialization does not look healthy.
    unet_model = None
    npk_predictor = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_logger.info("Using device: %s", device)

    _validate_required_model_files()

    runtime_logger.info("Loading UNet segmentation model from %s", UNET_CHECKPOINT)
    try:
        loaded_unet = load_segmentation_model(str(UNET_CHECKPOINT), device)
    except Exception as exc:
        runtime_logger.exception(
            "UNet segmentation model load failed from %s: %s",
            UNET_CHECKPOINT,
            exc,
        )
        raise RuntimeError("UNet segmentation model failed to load.") from exc

    if isinstance(loaded_unet, DummySegmenter):
        load_error = getattr(loaded_unet, "_load_error", "unknown error")
        runtime_logger.error(
            "UNet segmentation model load failed from %s (fallback DummySegmenter detected): %s",
            UNET_CHECKPOINT,
            load_error,
        )
        raise RuntimeError(f"UNet segmentation model failed to load ({load_error})")
    unet_model = loaded_unet

    runtime_logger.info("Loading NPK regression model from %s", REGRESSION_CHECKPOINT)
    try:
        loaded_predictor = NPKPredictor(str(REGRESSION_CHECKPOINT), strict=True)
    except Exception as exc:
        runtime_logger.exception(
            "NPK regression model load failed from %s: %s",
            REGRESSION_CHECKPOINT,
            exc,
        )
        raise RuntimeError("NPK regression model failed to load.") from exc
    npk_predictor = loaded_predictor

    runtime_logger.info("All models loaded successfully.")
    initialize_history()


def initialize_history():
    """Load history items for API responses."""
    global history_items, history_counter
    loaded_history, max_id = load_history(limit=25)
    if loaded_history:
        history_items = loaded_history
        history_counter = max_id
        return

    history_items = []
    history_counter = 0


def preprocess_image(image_data):
    """Convert uploaded image to TARGET_SIZE numpy array"""
    # Decode base64 or file
    if isinstance(image_data, str):
        # Base64 encoded
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_data, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_data))
    else:
        # File object
        image = Image.open(image_data)
    
    # Convert to RGB and resize
    image = image.convert('RGB')
    image_resized = image.resize(TARGET_SIZE, Image.BILINEAR)
    return np.array(image_resized)


def create_segmentation_overlay(original, mask):
    """Blend segmentation colors on top of the original image."""
    base = original.astype(np.float32)
    overlay = base.copy()
    alpha = 0.45  # color opacity

    for class_id in np.unique(mask):
        if class_id == 0 or class_id >= len(CLASS_COLORS):
            continue
        class_mask = mask == class_id
        color = np.array(CLASS_COLORS[class_id], dtype=np.float32)
        overlay[class_mask] = base[class_mask] * (1 - alpha) + color * alpha

    return overlay.astype(np.uint8)


def numpy_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def parse_float(value, default=None):
    """Parse a float-like value safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_npk_formula(formula: str):
    """Parse NPK formula string (e.g., '15-15-15' or '15,15,15') into a dict."""
    if not formula or not isinstance(formula, str):
        raise ValueError("สูตรปุ๋ยที่ต้องการวิเคราะห์จำเป็นต้องระบุ (NPK required).")

    separators = ["-", ",", " "]
    parts = [formula]
    for sep in separators:
        if sep in formula:
            parts = formula.replace(" ", "").split(sep) if sep != " " else formula.split()
            break

    if len(parts) != 3:
        raise ValueError("ระบุสูตร NPK เป็นรูปแบบ N-P-K เช่น 15-15-15.")

    try:
        n, p, k = (float(x) for x in parts)
    except ValueError as exc:
        raise ValueError("สูตร NPK ต้องเป็นตัวเลข เช่น 15-15-15.") from exc

    return {"N": n, "P": p, "K": k}


def evaluate_npk_against_threshold(predicted, target, threshold_percent):
    """
    Compare predicted NPK values against target within the provided threshold (%).
    Returns (status_level, message, errors_dict).
    """
    threshold = 5 if threshold_percent is None else float(threshold_percent)
    errors = {}
    exceeded = False

    for key in ("N", "P", "K"):
        pred_val = float(predicted.get(key, 0.0))
        target_val = float(target.get(key, 0.0))

        if target_val > 0:
            diff = abs(pred_val - target_val) / target_val * 100.0
        else:
            diff = abs(pred_val - target_val)  # fallback to absolute diff when target is zero

        errors[key] = diff
        if diff > threshold:
            exceeded = True

    if exceeded:
        return "bad", f"NPK prediction exceeds the {threshold}% threshold.", errors
    return "ok", f"NPK prediction within the {threshold}% threshold.", errors


def parse_ddmmyyyy(date_str):
    """Parse dd/mm/yyyy to a date object."""
    try:
        return datetime.datetime.strptime(date_str, "%d/%m/%Y").date()
    except Exception:
        raise ValueError("Invalid date format, use dd/mm/yyyy.")


def load_history_full():
    """Load all history rows from CSV without truncation."""
    if not HISTORY_CSV.exists():
        return []

    rows = []
    try:
        with HISTORY_CSV.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(row)
    except Exception as exc:
        print(f"Warning: failed to read history for export ({exc})")
    return rows


def filter_history_by_date(rows, start_date=None, end_date=None):
    """Filter rows whose date is between start_date and end_date (inclusive)."""
    filtered = []
    for row in rows:
        raw_date = row.get("date")
        try:
            row_date = datetime.datetime.strptime(raw_date, "%Y-%m-%d").date()
        except Exception:
            continue

        if start_date and row_date < start_date:
            continue
        if end_date and row_date > end_date:
            continue
        filtered.append(row)
    return filtered


def next_history_id():
    """Incrementing ID for history rows (not limited to in-memory)."""
    global history_counter
    history_counter += 1
    return history_counter



def extract_common_inputs():
    """Extract shared form/json inputs."""
    form_payload = request.form or {}
    json_payload = request.json if request.is_json else {}

    formula = (
        form_payload.get("formula")
        or form_payload.get("npk")
        or (json_payload or {}).get("formula")
        or (json_payload or {}).get("npk")
    )
    lot_number = (
        form_payload.get("lot_number")
        or form_payload.get("lotNumber")
        or (json_payload or {}).get("lot_number")
        or (json_payload or {}).get("lotNumber")
    )
    threshold = parse_float(form_payload.get("threshold") or (json_payload or {}).get("threshold"))

    return formula, lot_number, threshold


def process_image_payload(image_payload, target_npk, threshold):
    """Run segmentation + NPK prediction for a single image payload."""
    filename = getattr(image_payload, "filename", "upload")
    start_time = time.perf_counter()
    image_hash = None
    inference_logger.info("image_processing_start filename=%s", filename)
    status_level = "error"
    try:
        if isinstance(image_payload, str):
            image_bytes = base64.b64decode(image_payload.split(",", 1)[1])
        else:
            image_bytes = image_payload.read()
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        cached = _cache_get(image_hash)
        if cached is None:
            image_np = preprocess_image(image_bytes)
            mask = predict_segmentation(unet_model, image_np, device)
            overlay = create_segmentation_overlay(image_np, mask)
            npk_values = npk_predictor.predict(image_np, mask)
            mask_pixels = int(np.sum(mask > 0))
            cached = {
                "segmentation": numpy_to_base64(overlay),
                "npk": {
                    "N": float(npk_values["N"]),
                    "P": float(npk_values["P"]),
                    "K": float(npk_values["K"]),
                },
                "metadata": {
                    "classes_detected": int(len(np.unique(mask)) - 1),
                    "pixels_analyzed": mask_pixels,
                    "image_size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
                },
            }
            _cache_set(image_hash, cached)
            inference_logger.info("inference_cache miss")
        else:
            inference_logger.info("inference_cache hit")

        threshold_level, threshold_message, npk_errors = evaluate_npk_against_threshold(
            cached["npk"],
            target_npk,
            threshold,
        )
        status_level = threshold_level

        return {
            "filename": filename,
            "segmentation": cached["segmentation"],
            "npk": cached["npk"],
            "status_level": status_level,
            "status_message": threshold_message,
            "passed": status_level != "bad",
            "target_npk": target_npk,
            "npk_errors": npk_errors,
            "metadata": cached["metadata"],
        }
    finally:
        duration_s = time.perf_counter() - start_time
        inference_logger.info(
            "image_processing_end filename=%s duration_s=%.3f status=%s",
            filename,
            duration_s,
            status_level,
        )


def add_history_record(name, formula, lot_number, threshold, total_images, passed_images, avg_npk, status_level):
    """Push a new history entry to the in-memory list."""
    record_id = next_history_id()
    record = {
        "id": record_id,
        "name": name or "upload",
        "lot_number": lot_number or "N/A",
        "formula": formula or "N/A",
        "threshold": threshold,
        "total_images": total_images,
        "passed_images": passed_images,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "n": avg_npk["N"],
        "p": avg_npk["P"],
        "k": avg_npk["K"],
        "status": status_level,
    }
    history_items.insert(0, record)
    append_history(record)
    if len(history_items) > 25:
        history_items.pop()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if server and models are ready"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': unet_model is not None and npk_predictor is not None,
        'device': str(device),
        'model_size': SEGMENTATION_MODEL_SIZE,
    })


@app.route('/api/history', methods=['GET'])
def history_api():
    """Return processed history."""
    return jsonify({"items": history_items})


@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def delete_history_item(record_id):
    """Delete a history record by ID."""
    global history_items
    existing = next((item for item in history_items if item.get("id") == record_id), None)
    if not existing:
        return jsonify({"success": False, "error": "History record not found."}), 404

    history_items = [item for item in history_items if item.get("id") != record_id]
    persisted = delete_history_by_id(record_id)
    if not persisted and HISTORY_CSV.exists():
        runtime_logger.warning("History delete failed for id=%s (CSV not updated)", record_id)

    return jsonify({"success": True, "persisted": persisted})


@app.route('/api/history/export', methods=['GET'])
def export_history_csv():
    """Export filtered history as XLSX."""
    start_raw = request.args.get("start")
    end_raw = request.args.get("end")

    try:
        start_date = parse_ddmmyyyy(start_raw) if start_raw else None
        end_date = parse_ddmmyyyy(end_raw) if end_raw else None
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    rows = load_history_full()
    filtered = filter_history_by_date(rows, start_date, end_date)

    if not filtered:
        return jsonify({"success": False, "error": "No history found for the selected range."}), 404

    filtered.sort(key=lambda r: r.get("date", ""))

    # Create XLSX in-memory
    wb = Workbook()
    ws = wb.active
    ws.title = "History"

    headers = list(filtered[0].keys()) if filtered else []
    if headers:
        ws.append(headers)
        for row in filtered:
            ws.append([row.get(h, "") for h in headers])

    mem = io.BytesIO()
    wb.save(mem)
    mem.seek(0)

    date_label = ""
    if start_raw and end_raw:
        date_label = f"{start_raw}_to_{end_raw}"
    elif start_raw:
        date_label = f"from_{start_raw}"
    elif end_raw:
        date_label = f"until_{end_raw}"
    else:
        date_label = "history"

    filename = f"{date_label}.xlsx"
    return send_file(
        mem,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename,
    )


@app.route('/api/upload', methods=['POST'])
def upload_and_process():
    """
    Main endpoint: Upload image and get segmentation + NPK prediction
    
    Request:
        - file: image file OR
        - image: base64 encoded image
        - formula: fertilizer NPK formula string (optional)
        - lot_number: lot identifier (optional)
        - threshold: allowable percent error (optional)
    
    Response:
        {
            "success": true,
            "segmentation": "base64_image",
            "npk": {
                "N": 12.34,
                "P": 5.67,
                "K": 8.90
            },
            "metadata": {
                "classes_detected": 3,
                "pixels_analyzed": 262144
            }
        }
    """
    try:
        formula, lot_number, threshold = extract_common_inputs()
        target_npk = parse_npk_formula(formula)
        threshold_value = 5 if threshold is None else threshold

        # Get image from request
        if 'file' in request.files:
            file = request.files['file']
            processed = process_image_payload(file, target_npk, threshold_value)
        elif request.is_json and 'image' in request.json:
            image_data = request.json['image']
            processed = process_image_payload(image_data, target_npk, threshold_value)
        else:
            return jsonify({'error': 'No image provided'}), 400

        response = {
            'success': True,
            'mode': 'single',
            **processed,
            'inputs': {
                'formula': formula,
                'target_npk': target_npk,
                'lot_number': lot_number,
                'threshold': threshold_value
            }
        }

        add_history_record(
            name=processed.get("filename"),
            formula=formula,
            lot_number=lot_number,
            threshold=threshold_value,
            total_images=1,
            passed_images=1 if processed["passed"] else 0,
            avg_npk=processed["npk"],
            status_level=processed["status_level"],
        )

        return jsonify(response)
    
    except ValueError as e:
        inference_logger.exception("inference_error type=value_error message=%s", str(e))
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        inference_logger.exception("inference_error type=server_error message=%s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch-upload', methods=['POST'])
def batch_upload():
    """Process multiple images at once"""
    try:
        formula, lot_number, threshold = extract_common_inputs()
        target_npk = parse_npk_formula(formula)
        threshold_value = 5 if threshold is None else threshold
        files = request.files.getlist('files')

        if not files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400

        results = []
        for file in files:
            processed = process_image_payload(file, target_npk, threshold_value)
            results.append(processed)

        status_level = (
            "bad"
            if any(item["status_level"] == "bad" for item in results)
            else ("warn" if any(item["status_level"] == "warn" for item in results) else "ok")
        )
        passed_images = sum(1 for item in results if item["passed"])
        avg_npk = {
            "N": float(np.mean([item["npk"]["N"] for item in results])),
            "P": float(np.mean([item["npk"]["P"] for item in results])),
            "K": float(np.mean([item["npk"]["K"] for item in results])),
        }

        add_history_record(
            name=files[0].filename or "batch upload",
            formula=formula,
            lot_number=lot_number,
            threshold=threshold_value,
            total_images=len(results),
            passed_images=passed_images,
            avg_npk=avg_npk,
            status_level=status_level,
        )

        return jsonify({
            'success': True,
            'mode': 'batch',
            'items': results,
            'summary': {
                'total_images': len(results),
                'passed_images': passed_images,
                'status': status_level,
                'formula': formula,
                'lot_number': lot_number,
                'threshold': threshold_value,
                'target_npk': target_npk
            }
        })

    except ValueError as e:
        inference_logger.exception("inference_error type=value_error message=%s", str(e))
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        inference_logger.exception("inference_error type=server_error message=%s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# STARTUP
# ============================================================================

def run_dev():
    """Run the Flask app directly for local development."""
    print("\n" + "=" * 50)
    print("Fertilizer QC Backend API")
    print("=" * 50)

    # Initialize models
    try:
        initialize_models()
    except Exception as exc:
        runtime_logger.error("Backend startup aborted: %s", exc)
        raise SystemExit(1) from exc

    # Start server
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    display_host = "localhost" if host in ("0.0.0.0", "::") else host
    print("\n" + "=" * 50)
    print("Starting Flask server...")
    print(f"API available at: http://{display_host}:{port}")
    print("=" * 50 + "\n")

    app.run(
        host=host,
        port=port,
        debug=True,
    )


@app.before_request
def _log_request_start():
    endpoint = request.endpoint or "unknown"
    g.request_start = time.perf_counter()
    g.request_cpu_start = time.process_time()
    g.endpoint = endpoint
    g.endpoint_count = _record_endpoint_count(endpoint)
    runtime_logger.info(
        "request_start method=%s path=%s endpoint=%s size_bytes=%s",
        request.method,
        request.path,
        endpoint,
        _request_size_bytes(),
    )


@app.after_request
def _log_request_end(response):
    elapsed_s = time.perf_counter() - getattr(g, "request_start", time.perf_counter())
    cpu_s = time.process_time() - getattr(g, "request_cpu_start", time.process_time())
    in_flight = _finish_request_count()
    uptime_s = time.monotonic() - APP_START_MONO
    runtime_logger.info(
        "request_end method=%s path=%s endpoint=%s status=%s duration_s=%.4f cpu_s=%.4f "
        "uptime_s=%.1f rss_mb=%.1f in_flight=%s total_requests=%s endpoint_count=%s bytes_out=%s",
        request.method,
        request.path,
        getattr(g, "endpoint", "unknown"),
        response.status_code,
        elapsed_s,
        cpu_s,
        uptime_s,
        _memory_rss_mb(),
        in_flight,
        _REQUEST_COUNT,
        getattr(g, "endpoint_count", 0),
        response.content_length or 0,
    )
    g.request_count_finished = True
    return response


@app.teardown_request
def _log_request_teardown(error=None):
    if getattr(g, "request_count_finished", False):
        return
    if not hasattr(g, "request_start"):
        return
    in_flight = _finish_request_count()
    runtime_logger.info(
        "request_teardown method=%s path=%s endpoint=%s error=%s in_flight=%s",
        request.method,
        request.path,
        getattr(g, "endpoint", "unknown"),
        type(error).__name__ if error else "none",
        in_flight,
    )


if __name__ == "__main__":
    run_dev()
