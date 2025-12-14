# 🌾 Fertilizer Quality Control System

A full-stack web application for automated fertilizer image segmentation and NPK analysis with **Vue.js frontend** and **Flask backend**.

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Vue Frontend   │ ◄────► │  Flask Backend   │
│  Port: 5173     │  HTTP   │  Port: 5000      │
│                 │  REST   │                  │
│  - Upload UI    │  API    │  - UNet Model    │
│  - Display      │         │  - NPK Predictor │
└─────────────────┘         └──────────────────┘
```

## 📁 Project Structure

```
fertilizer-qc-app/
│
├── backend/                        # Flask API Backend
│   ├── app.py                     # Main Flask server
│   ├── requirements.txt           # Python dependencies
│   │
│   ├── models/                    # Model checkpoints
│   │   ├── best_model.pth
│   │   └── regression_model.plk
│   │
│   ├── src/
│   │   └── soil_segment/
│   │       ├── __init__.py
│   │       ├── model.py           # UNet architecture
│   │       ├── inference.py       # Segmentation
│   │       └── npk_predictor.py   # NPK regression
│   │
│   └── utils/
│       └── __init__.py
│
└── frontend/                       # Vue.js Frontend
    ├── package.json
    ├── vite.config.js
    ├── index.html
    │
    └── src/
        ├── main.js
        ├── App.vue                # Main component
        │
        ├── assets/
        │   └── style.css
        │
        └── components/
            ├── ImageUpload.vue
            ├── ResultDisplay.vue
            └── LoadingSpinner.vue
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Your trained model checkpoints

### Step 1: Backend Setup

```bash
# Navigate to backend
cd backend

# Create models directory and add your checkpoints
mkdir -p models
# Place your models:
# - models/best_model.pth
# - models/regression_model.plk

# Install dependencies
pip install -r requirements.txt

# Start backend server
python app.py
```

Backend will run on **http://localhost:5000**

### Step 2: Frontend Setup

```bash
# Navigate to frontend (in a new terminal)
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run on **http://localhost:5173**

### Step 3: Access the App

Open your browser to: **http://localhost:5173**

## 🔌 API Endpoints

### `GET /api/health`
Check backend status and model loading

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda:0"
}
```

### `POST /api/upload`
Upload image for analysis

**Request:**
- FormData with `file` field

**Response:**
```json
{
  "success": true,
  "original": "data:image/png;base64,...",
  "segmentation": "data:image/png;base64,...",
  "npk": {
    "N": 12.34,
    "P": 5.67,
    "K": 8.90
  },
  "metadata": {
    "classes_detected": 3,
    "pixels_analyzed": 1048576,
    "image_size": "1024x1024"
  }
}
```

### `POST /api/batch-upload`
Process multiple images

**Request:**
- FormData with multiple `files`

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "filename": "img1.jpg",
      "npk": {...},
      "classes_detected": 3
    }
  ]
}
```

## 📊 Pipeline Flow

```
1. User uploads image via Vue frontend
   ↓
2. Frontend sends to Flask /api/upload endpoint
   ↓
3. Backend resizes image to 1024×1024
   ↓
4. UNet model performs segmentation
   ↓
5. Features extracted from segmented regions
   ↓
6. Regression model predicts NPK values
   ↓
7. Backend returns JSON response
   ↓
8. Frontend displays results with visualizations
```

## 🔧 Configuration

### Backend - Change Model Paths

Edit `backend/app.py`:
```python
UNET_CHECKPOINT = CHECKPOINT_DIR / "best_model.pth"
REGRESSION_CHECKPOINT = CHECKPOINT_DIR / "regression_model.plk"
```

### Frontend - Change API URL

Edit `frontend/src/App.vue`:
```javascript
apiUrl: 'http://localhost:5000/api'
```

### Backend - Adjust Number of Classes

Edit `backend/src/soil_segment/inference.py`:
```python
num_classes = checkpoint.get('num_classes', 4)
```

## 🎨 Features

### Frontend
- ✅ Drag & drop image upload
- ✅ Image preview before analysis
- ✅ Real-time loading states
- ✅ Side-by-side image comparison
- ✅ Animated NPK progress bars
- ✅ Export results to JSON
- ✅ Responsive design

### Backend
- ✅ RESTful API endpoints
- ✅ Automatic image preprocessing
- ✅ Multi-class segmentation
- ✅ Feature extraction
- ✅ NPK prediction
- ✅ Error handling
- ✅ CORS support

## 🐛 Troubleshooting

### "Cannot connect to backend"

Check if Flask is running:
```bash
curl http://localhost:5000/api/health
```

### "Models not loaded"

Verify checkpoints exist:
```bash
ls -lh backend/models/
```

### "CORS error"

Ensure `flask-cors` is installed:
```bash
pip install flask-cors
```

### Port already in use

Change ports in:
- Backend: `app.py` → `app.run(port=5001)`
- Frontend: `vite.config.js` → `server: { port: 5174 }`

## 📝 Development

### Run in Production Mode

**Backend:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Frontend:**
```bash
npm run build
npm run preview
```

### Add New API Endpoint

1. Add route in `backend/app.py`
2. Create corresponding method in Vue component
3. Update API calls in `frontend/src/App.vue`

## 🔒 Security Notes

- Runs locally only by default
- No external data transmission
- CORS restricted to localhost
- File upload size limits recommended

## 📦 Model Requirements

### UNet Checkpoint Format
```python
{
    'model_state_dict': OrderedDict(...),
    'num_classes': 4,
    'epoch': 100
}
```

### Regression Model
- Sklearn model (joblib format)
- Input: Feature vector
- Output: [N, P, K] array

## 🎓 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vue 3 + Vite |
| Backend | Flask + PyTorch |
| Models | UNet + scikit-learn |
| Styling | CSS3 |
| API | REST |

## 📄 License

Proprietary - For internal use only

---

**Ready to analyze fertilizer quality! 🔬**

For issues, check the troubleshooting section or review the API endpoint documentation.