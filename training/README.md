# Training Interface

This folder contains the training-only assets that were moved out of the backend runtime path.

## Layout

- `interface/training_server.py`: FastAPI server for the local training UI
- `interface/training_ui.html`: browser UI for dataset upload and training control
- `src/`: UNet and regression trainers plus their training config
- `datasets/Unet_dataset/`: coated segmentation dataset root
- `datasets/Regression_dataset/`: coated regression dataset root
- `trained_models/`: training outputs such as `best_model.pth`, `regression_model.pkl`, plots, and CV summaries

Uncoated runs use sibling folders created automatically on demand:

- `datasets/Unet_dataset_uncoated/`
- `datasets/Regression_dataset_uncoated/`
- `trained_models_uncoated/`

## Install

From the repository root:

```bash
python3 -m pip install -e ".[training]"
```

## Run the UI

From the repository root:

```bash
python3 -m uvicorn training.interface.training_server:app --reload --port 8000
```

Open `http://localhost:8000`.

## Dataset expectations

UNet datasets must look like:

```text
training/datasets/Unet_dataset/
├── images/
└── masks/
```

Regression datasets must be grouped by NPK formula:

```text
training/datasets/Regression_dataset/
├── 13-6-27/
│   └── *.JPG
└── 15-5-20/
    └── *.JPG
```

The UI accepts zip uploads too:

- UNet zips should contain top-level `images/` and `masks/` folders.
- Regression zips should contain formula folders like `15-5-20/`.
- Loose regression files require a formula directory entry in the UI.

## Outputs

- UNet coated: `training/trained_models/best_model.pth`
- UNet uncoated: `training/trained_models_uncoated/best_model_uncoated.pth` plus LOO-CV summary JSON
- Regression coated: `training/trained_models/regression_model.pkl`
- Regression uncoated: `training/trained_models_uncoated/regression_model_uncoated.pkl`

To promote a trained model into the production backend, copy the chosen checkpoint into `backend/app_models/`.
