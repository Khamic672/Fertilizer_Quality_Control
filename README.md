# Fertilizer Quality Control

Flask + Vue application for fertilizer quality control from images. The system segments pellet classes from an input image, estimates NPK values, compares the prediction against a target formula, and stores each run in CSV-backed history.

## What the repo contains

The handwritten application code lives in:

- `backend/src/soil_segment/`: model loading, inference, NPK prediction, config, storage
- `backend/src/soil_segment/api/app.py`: Flask API and request lifecycle
- `frontend/src/App.vue`: single-page UI for upload, review, and history
- `test/test_model_predictions.py`: snapshot-style model pipeline test

Compatibility wrappers exist in `backend/app.py`, `backend/history.py`, and `backend/wsgi.py`.

Generated build output in `frontend/dist/` is not the source of truth.

## How it works

### Backend flow

1. The backend starts and loads:
   - `backend/models/best_model.pth`
   - `backend/models/regression_model.pkl`
   - optionally `backend/models/best_uncoated_model.pth`
2. Each uploaded image is converted to RGB and resized to `SEGMENTATION_MODEL_SIZE` (default `512`).
3. The segmentation model predicts a pellet-class mask.
4. The backend overlays class colors on the resized image and returns that overlay as base64 PNG.
5. The NPK predictor uses the mask to estimate N, P, and K:
   - first from class composition heuristics
   - then from the regression checkpoint if available
6. Predicted NPK is compared against the target formula using fixed absolute allowances.
7. The result is cached in memory by `sha256(image_bytes)` plus model variant.
8. A summary row is appended to `backend/history.csv`.

### Frontend flow

The Vue app has two screens:

- `Upload`: pick one image, multiple images from the same lot, or capture from the camera
- `History`: review recent runs, delete rows, and export filtered history to Excel

The frontend calls:

- `GET /api/health` on load
- `POST /api/upload` for one image
- `POST /api/batch-upload` for multiple images
- `GET /api/history` for the recent history table
- `DELETE /api/history/<id>` to remove a row
- `GET /api/history/export` to download `.xlsx`

## Segmentation classes

The overlay legend and heuristic NPK logic use these pellet classes:

- `Black_DAP`
- `Red_MOP`
- `White_AMP`
- `White_Boron`
- `White_Mg`
- `Yellow_Urea`

## QC status logic

The API compares predicted NPK against the requested formula with fixed allowances:

- `N` and `P`
  - target `< 8.0`: allowance `0.4`
  - `8.0` to `16.0`: allowance `0.6`
  - `16.0` to `24.0`: allowance `0.8`
  - `> 24.0`: allowance `1.0`
- `K`
  - target `< 8.0`: allowance `0.5`
  - `8.0` to `16.0`: allowance `0.8`
  - `16.0` to `24.0`: allowance `1.0`
  - `> 24.0`: allowance `1.2`

If any nutrient exceeds its allowance, the result is marked `bad`; otherwise it is `ok`.

## Requirements

- Python `3.11+`
- Node.js `18+`
- Model files in `backend/models/`

Required model files:

- `backend/models/best_model.pth`
- `backend/models/regression_model.pkl`

Optional model file:

- `backend/models/best_uncoated_model.pth`

If the optional uncoated checkpoint is missing and the UI requests `Uncoated`, the backend falls back to `best_model.pth` and returns a warning note in the response.

## Installation

Use the root `pyproject.toml` as the main install path.

```bash
python3 -m pip install -e .
```

For development and tests:

```bash
python3 -m pip install -e ".[dev]"
```

For ONNX export/runtime support:

```bash
python3 -m pip install -e ".[onnx]"
```

For both:

```bash
python3 -m pip install -e ".[dev,onnx]"
```

The repo also includes `setup.sh`, but the maintained install path is the `pyproject.toml` flow above.

## Run locally

### Backend

From the repo root:

```bash
python3 backend/app.py
```

Default backend URL:

```text
http://localhost:5000/api
```

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Default frontend URL:

```text
http://localhost:5173
```

To point the frontend at a different backend, create `frontend/.env`:

```bash
VITE_API_URL=http://localhost:5000/api
```

## Docker

The backend directory is containerized separately and runs behind Gunicorn.
Docker Compose (inside `backend/`) builds with repo-root context and uses `backend/Dockerfile`.

From `backend/`:

```bash
docker compose up --build
```

This exposes the API at:

```text
http://localhost:9000/api
```

The compose file mounts `backend/models/` into the container read-only.

## Runtime configuration

The backend reads these environment variables from `backend/src/soil_segment/config.py` and `backend/src/soil_segment/api/app.py`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Flask bind host in local dev |
| `PORT` | `5000` | Flask port in local dev |
| `INFERENCE_CACHE_MAX_ITEMS` | `32` | Max cached image/model results |
| `SEGMENTATION_MODEL_SIZE` | `512` | Input resize width and height |
| `SEGMENTATION_RUNTIME` | `torch` | `torch` or `onnx` |
| `SEGMENTATION_QUANTIZATION` | `none` | Torch dynamic quantization mode |
| `SEGMENTATION_ONNX_PATH` | `backend/models/segmentation.onnx` | ONNX export path |
| `SEGMENTATION_ONNX_INT8_PATH` | `backend/models/segmentation.int8.onnx` | INT8 ONNX path |
| `SEGMENTATION_ONNX_EXPORT` | `auto` | `auto`, `always`, `never` |
| `SEGMENTATION_ONNX_QUANTIZE` | `int8` | `none` or `int8` |
| `SEGMENTATION_ONNX_CALIBRATION_DIR` | empty | Images for ONNX INT8 calibration |
| `SEGMENTATION_ONNX_CALIBRATION_SAMPLES` | `16` | Max calibration images |
| `SEGMENTATION_ONNX_OPSET` | `17` | ONNX export opset |

Notes:

- The backend requires the main segmentation and regression checkpoints at startup.
- The backend can export ONNX or run ONNX Runtime when the optional dependencies are installed.
- The Flask API keeps only the latest 25 history rows in memory for `/api/history`, but export reads the full CSV.

## API summary

### `GET /api/health`

Returns backend readiness, model state, selected device, and model size.

Example response fields:

- `status`
- `models_loaded`
- `uncoated_model_loaded`
- `device`
- `model_size`

### `POST /api/upload`

Single-image inference.

Accepted input:

- multipart form with `file`
- or JSON with `image` as base64 data URL

Other fields:

- `formula` or `npk` (required, format like `15-15-15`)
- `lot_number` or `lotNumber` (optional)
- `uncoated` (optional boolean-like flag)

Example:

```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@test/20-3-3.JPG" \
  -F "formula=15-15-15" \
  -F "lot_number=LOT-001"
```

### `POST /api/batch-upload`

Multi-image inference for a single lot.

Accepted input:

- multipart form with repeated `files`
- `formula` or `npk` (required)
- `lot_number` (optional)
- `uncoated` (optional)

Example:

```bash
curl -X POST http://localhost:5000/api/batch-upload \
  -F "files=@test/20-3-3.JPG" \
  -F "files=@test/14-7-35.JPG" \
  -F "formula=15-15-15" \
  -F "lot_number=LOT-001"
```

### `GET /api/history`

Returns the in-memory recent history list:

```json
{ "items": [...] }
```

### `DELETE /api/history/<id>`

Deletes the row from memory and from `backend/history.csv`.

### `GET /api/history/export`

Exports filtered history as `.xlsx`.

Query parameters:

- `start=dd/mm/yyyy`
- `end=dd/mm/yyyy`

Example:

```bash
curl -OJ "http://localhost:5000/api/history/export?start=01/03/2026&end=31/03/2026"
```

## Response shape highlights

Upload responses include fields like:

- `success`
- `mode`
- `segmentation`: base64 PNG overlay
- `npk`: predicted N/P/K
- `status_level`
- `status_message`
- `passed`
- `npk_errors`
- `npk_allowances`
- `model_variant`
- `metadata.classes_detected`
- `metadata.pixels_analyzed`
- `metadata.image_size`

Batch responses also include:

- `items`
- `summary.total_images`
- `summary.passed_images`
- `summary.status`

## Persistence and logs

- History CSV: `backend/history.csv`
- Runtime log: `backend/logs/runtime.log`
- Inference log: `backend/logs/inference.log`

History export writes an in-memory Excel workbook and returns it as a download; it does not create an `.xlsx` file on disk.

## Testing

The test suite includes a model snapshot test that loads the real checkpoints and validates:

- segmentation class counts on `test/20-3-3.JPG`
- regression output within tolerance

Run it from the repo root:

```bash
python3 -m pytest -q test/test_model_predictions.py
```

Run all tests:

```bash
python3 -m pytest -q
```

Test notes:

- The test skips if the sample image or checkpoints are missing.
- Snapshot values are tied to the current model files and should be refreshed if checkpoints change.

## Frontend behavior notes

- Single-page Vue app with hash-based page switching between `#upload` and `#history`
- Camera capture uses `navigator.mediaDevices.getUserMedia`
- Batch upload is supported only when multiple image files are selected
- History export is triggered from the UI modal and downloads `.xlsx`
- The UI shows a class legend that matches the backend segmentation palette

## Repo structure

```text
.
├── backend/
│   ├── app.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── history.csv
│   ├── models/
│   ├── src/soil_segment/
│   │   ├── api/app.py
│   │   ├── config.py
│   │   ├── inference.py
│   │   ├── model.py
│   │   ├── npk_predictor.py
│   │   └── storage/history.py
│   └── wsgi.py
├── frontend/
│   ├── package.json
│   ├── src/App.vue
│   └── src/assets/style.css
├── test/
│   ├── *.JPG
│   └── test_model_predictions.py
├── pyproject.toml
└── README.md
```
