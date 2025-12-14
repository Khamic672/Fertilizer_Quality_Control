"""
Flask Backend API for Fertilizer Quality Control
Exposes endpoints for image processing and NPK prediction
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import sys
import torch
import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from soil_segment.inference import (
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

CHECKPOINT_DIR = Path(__file__).parent / "models"
UNET_CHECKPOINT = CHECKPOINT_DIR / "unet_best.pth"
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
    global sample_items, history_items
    if sample_items:
        return
    sample_items = [
        {"id": 1, "label": "Sample A", "image": generate_placeholder_base64()},
        {"id": 2, "label": "Sample B", "image": generate_placeholder_base64((44, 92, 138))},
        {"id": 3, "label": "Sample C", "image": generate_placeholder_base64((80, 160, 90))},
    ]
    history_items = [
        {
            "id": 1,
            "name": "Lot A",
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "moisture": 10.0,
            "ph": 6.7,
            "n": 12.3,
            "p": 6.1,
            "k": 7.9,
            "status": "ok",
        }
    ]


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
    """Create colored overlay of segmentation"""
    colors = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
    ]
    
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in np.unique(mask):
        if class_id == 0:
            continue
        color_idx = (class_id - 1) % len(colors)
        colored_mask[mask == class_id] = colors[color_idx]
    
    # Black background
    overlay = np.zeros_like(original)
    overlay[mask > 0] = colored_mask[mask > 0]
    
    return overlay


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


@app.route('/api/upload', methods=['POST'])
def upload_and_process():
    """
    Main endpoint: Upload image and get segmentation + NPK prediction
    
    Request:
        - file: image file OR
        - image: base64 encoded image
    
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
        # Get image from request
        if 'file' in request.files:
            file = request.files['file']
            image_np = preprocess_image(file)
        elif request.is_json and 'image' in request.json:
            image_data = request.json['image']
            image_np = preprocess_image(image_data)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Step 1: Segmentation
        mask = predict_segmentation(unet_model, image_np, device)
        
        # Step 2: Create overlay
        overlay = create_segmentation_overlay(image_np, mask)
        
        # Step 3: NPK Prediction
        npk_values = npk_predictor.predict(image_np, mask)
        status_messages = build_status_messages(npk_values, int(np.sum(mask > 0)))
        
        # Step 4: Prepare response
        response = {
            'success': True,
            'original': numpy_to_base64(image_np),
            'segmentation': numpy_to_base64(overlay),
            'npk': {
                'N': float(npk_values['N']),
                'P': float(npk_values['P']),
                'K': float(npk_values['K'])
            },
            'status': status_messages,
            'metadata': {
                'classes_detected': int(len(np.unique(mask)) - 1),
                'pixels_analyzed': int(np.sum(mask > 0)),
                'image_size': '1024x1024'
            }
        }
        
        # Update in-memory history
        history_items.insert(0, {
            "id": len(history_items) + 1,
            "name": request.files.get('file', type('obj', (object,), {'filename': 'upload.png'})()).filename if request.files else "upload",
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "moisture": round(np.random.uniform(8, 14), 2),
            "ph": round(np.random.uniform(5.5, 7.5), 2),
            "n": npk_values['N'],
            "p": npk_values['P'],
            "k": npk_values['K'],
            "status": "bad" if any(s['level'] == 'bad' for s in status_messages) else ("warn" if any(s['level'] == 'warn' for s in status_messages) else "ok"),
        })
        if len(history_items) > 25:
            history_items.pop()

        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch-upload', methods=['POST'])
def batch_upload():
    """Process multiple images at once"""
    try:
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            image_np = preprocess_image(file)
            mask = predict_segmentation(unet_model, image_np, device)
            overlay = create_segmentation_overlay(image_np, mask)
            npk_values = npk_predictor.predict(image_np, mask)
            
            results.append({
                'filename': file.filename,
                'npk': npk_values,
                'classes_detected': int(len(np.unique(mask)) - 1)
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Fertilizer QC Backend API")
    print("="*50)
    
    # Initialize models
    initialize_models()
    
    # Start server
    print("\n" + "="*50)
    print("Starting Flask server...")
    print("API available at: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
