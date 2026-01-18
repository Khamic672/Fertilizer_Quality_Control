"""
Flask Backend API for Fertilizer Quality Control
Exposes endpoints for image processing and NPK prediction
"""
import base64
import csv
import datetime
import io
import os
from openpyxl import Workbook

import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from soil_segment.config import HISTORY_FILE, MODELS_DIR
from soil_segment.storage import append_history, load_history

from soil_segment.inference import (
    CLASS_COLORS,
    load_segmentation_model,
    predict_segmentation,
)
from soil_segment.npk_predictor import NPKPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Vue frontend

# Global variables for models
unet_model = None
npk_predictor = None
device = None
history_items = []
sample_items = []
history_counter = 0
HISTORY_CSV = HISTORY_FILE

CHECKPOINT_DIR = MODELS_DIR
UNET_CHECKPOINT = CHECKPOINT_DIR / "best_model.pth"
REGRESSION_CHECKPOINT = CHECKPOINT_DIR / "regression_model.pkl"
TARGET_SIZE = (1024, 1024)


def initialize_models():
    """Load models on startup"""
    global unet_model, npk_predictor, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load UNet
    print("Loading UNet segmentation model...")
    unet_model = load_segmentation_model(str(UNET_CHECKPOINT), device)
    
    # Load regression model
    print("Loading NPK regression model...")
    npk_predictor = NPKPredictor(str(REGRESSION_CHECKPOINT))
    
    print("All models loaded successfully.")
    initialize_samples()


def initialize_samples():
    """Seed sample gallery items."""
    global sample_items, history_items, history_counter
    if sample_items:
        return
    sample_items = [
        {"id": 1, "label": "Sample A", "image": generate_placeholder_base64()},
        {"id": 2, "label": "Sample B", "image": generate_placeholder_base64((44, 92, 138))},
        {"id": 3, "label": "Sample C", "image": generate_placeholder_base64((80, 160, 90))},
    ]
    loaded_history, max_id = load_history(limit=25)
    if loaded_history:
        history_items = loaded_history
        history_counter = max_id
        return

    history_items = [
        {
            "id": 1,
            "name": "Lot A",
            "lot_number": "Lot A",
            "formula": "15-15-15",
            "threshold": 5,
            "total_images": 1,
            "passed_images": 1,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "moisture": 10.0,
            "ph": 6.7,
            "n": 12.3,
            "p": 6.1,
            "k": 7.9,
            "status": "ok",
        }
    ]
    history_counter = max(item["id"] for item in history_items)


def preprocess_image(image_data):
    """Convert uploaded image to 1024x1024 numpy array"""
    # Decode base64 or file
    if isinstance(image_data, str):
        # Base64 encoded
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
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


def generate_placeholder_base64(color=(242, 140, 40)):
    """Create a simple placeholder banner as base64 PNG."""
    width, height = 300, 150
    img = Image.new("RGB", (width, height), (245, 248, 255))
    stripe = Image.new("RGB", (width, 50), color)
    img.paste(stripe, (0, height // 2 - 25))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def build_status_messages(npk_values, mask_pixels):
    """Generate UI-friendly status list based on thresholds."""
    statuses = []
    if mask_pixels < 1000:
        statuses.append({"level": "bad", "message": "Segmentation confidence is low. Retry with a clearer image."})
    else:
        statuses.append({"level": "good", "message": "Mask detected successfully."})

    if npk_values["N"] < 5 or npk_values["P"] < 3:
        statuses.append({"level": "warn", "message": "Nitrogen or phosphorus below desired range."})
    else:
        statuses.append({"level": "good", "message": "N and P within acceptable range."})

    if npk_values["K"] < 4:
        statuses.append({"level": "warn", "message": "Potassium slightly low."})
    else:
        statuses.append({"level": "good", "message": "Potassium level looks stable."})

    return statuses


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


def status_level_from_messages(status_messages):
    """Map list of status messages to a single level."""
    if any(s["level"] == "bad" for s in status_messages):
        return "bad"
    if any(s["level"] == "warn" for s in status_messages):
        return "warn"
    return "ok"


def process_image_payload(image_payload, target_npk, threshold):
    """Run segmentation + NPK prediction for a single image payload."""
    image_np = preprocess_image(image_payload)
    mask = predict_segmentation(unet_model, image_np, device)
    overlay = create_segmentation_overlay(image_np, mask)
    npk_values = npk_predictor.predict(image_np, mask)
    status_messages = build_status_messages(npk_values, int(np.sum(mask > 0)))
    threshold_level, threshold_message, npk_errors = evaluate_npk_against_threshold(npk_values, target_npk, threshold)
    status_messages.append({"level": threshold_level, "message": threshold_message})
    status_level = status_level_from_messages(status_messages)

    return {
        "filename": getattr(image_payload, "filename", "upload"),
        "original": numpy_to_base64(image_np),
        "segmentation": numpy_to_base64(overlay),
        "npk": {
            "N": float(npk_values["N"]),
            "P": float(npk_values["P"]),
            "K": float(npk_values["K"])
        },
        "status": status_messages,
        "status_level": status_level,
        "passed": status_level != "bad",
        "target_npk": target_npk,
        "npk_errors": npk_errors,
        "metadata": {
            "classes_detected": int(len(np.unique(mask)) - 1),
            "pixels_analyzed": int(np.sum(mask > 0)),
            "image_size": "1024x1024"
        }
    }


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
        'device': str(device)
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model metadata"""
    return jsonify({
        'unet': {
            'checkpoint': str(UNET_CHECKPOINT),
            'loaded': unet_model is not None
        },
        'regression': {
            'checkpoint': str(REGRESSION_CHECKPOINT),
            'loaded': npk_predictor is not None
        }
    })


@app.route('/api/samples', methods=['GET'])
def samples_api():
    """Return sample gallery items."""
    return jsonify({"items": sample_items})


@app.route('/api/history', methods=['GET'])
def history_api():
    """Return processed history."""
    return jsonify({"items": history_items})


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
            "original": "base64_image",
            "segmentation": "base64_image",
            "npk": {
                "N": 12.34,
                "P": 5.67,
                "K": 8.90
            },
            "metadata": {
                "classes_detected": 3,
                "pixels_analyzed": 1048576
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
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
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
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
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
    initialize_models()

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


if __name__ == "__main__":
    run_dev()
