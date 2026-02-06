# Fertilizer Quality Control

Simple steps to run the app (Flask API + Vue frontend).

## What you need
- Python 3.11+
- Node.js 18+
- Model files in `backend/models/`:
  - `best_model.pth`
  - `regression_model.pkl`

## Model download
- You can download the models from: https://drive.google.com/drive/folders/1KbrBNBra7BiPkGN_agqiyhIEKc8-egJq?usp=sharing

## Backend (API)
From the repo root:
```bash
pip install -e .
cd backend
python app.py
```
The API runs at `http://localhost:5000/api`.

## Frontend (UI)
In a new terminal:
```bash
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (default `http://localhost:5173`). If your API is elsewhere, set `VITE_API_URL` in `frontend/.env` (example: `VITE_API_URL=http://localhost:5000/api`).

## Exporting history
Click “ดาวน์โหลด Excel” in the UI, choose a date range, and it will download a filtered Excel file generated from `backend/history.csv`.

## Notes
- Large model files are ignored by git; keep them in `backend/models/`.
- History is stored in `backend/history.csv` and served via `/api/history` and `/api/history/export`.
- Default image size processed: 512x512. Adjust via `SEGMENTATION_MODEL_SIZE` in `backend/src/soil_segment/config.py` (or set the env var).
