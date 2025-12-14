"""
Flask Backend API for Fertilizer Quality Control
Exposes endpoints for image processing and NPK prediction
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import cv2
import io
import base64
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from soil_segment.model import UNet
from soil_segment.inference import load_model, predict_segmentation
from soil_segment.npk_predictor import NPKPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Vue frontend

# Global variables for models
unet_model = None
npk_predictor = None
device = None

# Configuration
CHECKPOINT_DIR = Path(__file__).parent / "models"
UNET_CHECKPOINT = CHECKPOINT_DIR / "unet_best.pth"
REGRESSION_CHECKPOINT = CHECKPOINT_DIR / "regression_model.pkl"


def initialize_models():
    """Load models on startup"""
    global unet_model, npk_predictor, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load UNet
    print("Loading UNet segmentation model...")
    unet_model = load_model(str(UNET_CHECKPOINT), device)
    
    # Load regression model
    print("Loading NPK regression model...")
    npk_predictor = NPKPredictor(str(REGRESSION_CHECKPOINT))
    
    print("✓ All models loaded successfully!")


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
    image_resized = image.resize((1024, 1024), Image.BILINEAR)
    
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
        elif 'image' in request.json:
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
            'metadata': {
                'classes_detected': int(len(np.unique(mask)) - 1),
                'pixels_analyzed': int(np.sum(mask > 0)),
                'image_size': '1024x1024'
            }
        }
        
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
    print("🌾 Fertilizer QC Backend API")
    print("="*50)
    
    # Verify checkpoints exist
    if not UNET_CHECKPOINT.exists():
        print(f"ERROR: UNet checkpoint not found at {UNET_CHECKPOINT}")
        print("Please place your checkpoint in backend/models/")
        sys.exit(1)
    
    if not REGRESSION_CHECKPOINT.exists():
        print(f"ERROR: Regression model not found at {REGRESSION_CHECKPOINT}")
        print("Please place your model in backend/models/")
        sys.exit(1)
    
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