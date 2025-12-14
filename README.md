Fertilizer Quality Control (Local)
==================================

Vue frontend + Flask backend for running a local fertilizer quality control pipeline. The app accepts an image, resizes to 1024x1024, segments it (main file → mask segment/initial predict), runs regression for NPK, postprocesses, and returns images + metadata through REST endpoints.

Requirements
------------
- Python 3.9+ (torch, flask)
- Node 18+ (Vite + Vue)
- Trained checkpoints placed in `backend/models/`:
  - `unet_best.pth` (segmentation)
  - `regression_model.pkl` (NPK regression)
  If checkpoints are missing or cannot load, the backend uses deterministic fallbacks so the UI still works.

Backend setup
-------------
1) Create env + install deps:
   - `cd backend`
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2) Place your checkpoints in `backend/models/` (names above).
3) Run the API: `python app.py`
   - Health: `GET http://localhost:5000/api/health`
   - Upload: `POST http://localhost:5000/api/upload` (`file` form field or JSON `image` base64)
   - Batch: `POST http://localhost:5000/api/batch-upload`
   - Model info: `GET http://localhost:5000/api/model-info`

Frontend setup
--------------
1) `cd frontend`
2) `npm install`
3) `npm run dev` (defaults to http://localhost:5173)
   - The frontend calls the backend at `http://localhost:5000/api` by default. Override with `VITE_API_URL`.

Data flow
---------
- Accept image → resize to 1024x1024 (RGB).
- Mask segment via UNet (or fallback dummy mask).
- Regression on extracted features → N/P/K.
- Postprocess + return: base64 original, base64 overlay, NPK numbers, metadata (classes detected, pixels analyzed).

Key files
---------
- `backend/app.py` — Flask API endpoints and pipeline glue.
- `backend/src/soil_segment/` — UNet model, inference helpers, regression wrapper.
- `backend/requirements.txt` — backend dependencies.
- `frontend/src/App.vue` — main Vue layout and API wiring.
- `frontend/src/components/` — upload, results, and spinner components.
- `frontend/vite.config.js` — Vite config (port 5173).
