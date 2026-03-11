# Chess Piece Detection and Opening Search

This project does two things:

1. Trains a YOLO model to detect chess pieces from board images.
2. Runs a Streamlit app that:
   - detects pieces in an image
   - highlights filtered pieces and/or squares
   - identifies built-in chess openings from the detected board state
   - searches folders of images for matching positions

The opening detection is rule-based logic on top of the YOLO detections. It is not a separate ML model.

## Published model

The repository is intended to ship with a trained model weight file tracked with Git LFS:

- `runs/detect/chess-pieces-gpu-90min-safe/weights/best.pt`

That is the default weight path used by the app.

## What is in this repo

- `chess-pieces.ndjson`
  - Ultralytics dataset export for the chess-pieces dataset
- `scripts/prepare_ultralytics_ndjson.py`
  - downloads images from the NDJSON export and converts them into a normal YOLO dataset layout
- `train_yolo.py`
  - trains the detector with Ultralytics
- `app.py`
  - Streamlit app for image analysis and folder search
- `src/chessvision/core.py`
  - square mapping and highlight logic
- `src/chessvision/openings.py`
  - built-in opening definitions and matching logic
- `notebooks/train_chess_yolo.ipynb`
  - notebook workflow for Colab, Kaggle, or Jupyter

## Current runtime reality

At the time this README was last updated, the local environment had:

- `ultralytics` installed
- `streamlit` installed
- `torch` installed as CPU-only

That means:

- the app can run locally
- YOLO training can run locally
- training will be slow on CPU
- GPU training requires installing a CUDA-enabled PyTorch build separately

If you want GPU training, install a CUDA-compatible PyTorch build for your machine before training. Without that, Ultralytics will run on CPU.

## Features

### Piece detection

- Detects chess pieces in an existing image
- Supports uploaded images and local image paths
- Uses YOLO object detection

### Highlighting

- Highlights only the pieces that match the selected filters
- Uses a fixed color per piece family
- Draws a thick white outline for selected board squares
- If a square is selected but no matching piece is present, the square outline still appears

### Opening recognition

- Identifies openings from the detected board state
- Uses piece color, piece type, and square occupancy
- Does not add any highlight overlays
- Can be used as a filter during folder search

### Folder search

- Scans a folder of images
- Returns only images that match your selected filters
- Supports piece filters, square filters, and opening filters together

## Built-in openings

The app currently includes 12 rule-based opening templates:

- Ruy Lopez
- Italian Game
- Sicilian Defense
- French Defense
- Caro-Kann Defense
- Scandinavian Defense
- Pirc Defense
- Queen's Gambit
- Slav Defense
- King's Indian Defense
- English Opening
- London System

## Quick start

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Convert the NDJSON export into a YOLO dataset

```powershell
python scripts/prepare_ultralytics_ndjson.py --input chess-pieces.ndjson --output data/chess-pieces
```

When this finishes successfully, you should have:

- `data/chess-pieces/images/train`
- `data/chess-pieces/images/val`
- `data/chess-pieces/images/test`
- `data/chess-pieces/labels/train`
- `data/chess-pieces/labels/val`
- `data/chess-pieces/labels/test`
- `data/chess-pieces/data.yaml`
- `data/chess-pieces/summary.json`

Important:

- this step downloads thousands of images, so it can take a long time
- if it is interrupted, run the same command again
- the script skips files that already exist, so rerunning it is the correct recovery path
- `data.yaml` is written at the end, so if `data/chess-pieces/data.yaml` does not exist yet, the dataset preparation is not finished

### 4. Train the model

```powershell
python train_yolo.py --data data/chess-pieces/data.yaml --model yolo11n.pt --epochs 40 --imgsz 640
```

This creates a run directory similar to:

- `runs/detect/chess-pieces/weights/best.pt`
- `runs/detect/chess-pieces/weights/last.pt`

If you train with a different `--name`, your weights will be saved under that run name instead.
The published trained model in this repo is:

- `runs/detect/chess-pieces-gpu-90min-safe/weights/best.pt`

Important:

- on CPU, training can take a long time
- if you have only CPU PyTorch installed, this is expected
- if you want GPU training, install CUDA-enabled PyTorch first

### Optional: quick-start training on a smaller subset

If you want a faster local setup that produces a usable checkpoint for the app, create a smaller dataset first:

```powershell
python scripts/create_dataset_subset.py --source data/chess-pieces --output data/chess-pieces-quickstart --train 512 --val 128 --test 16
python train_yolo.py --data data/chess-pieces-quickstart/data.yaml --model yolo11n.pt --epochs 3 --imgsz 416 --batch 8 --workers 0 --device cpu --project runs/detect --name chess-pieces
```

This is the quickest way to get:

- `runs/detect/chess-pieces/weights/best.pt`

Tradeoff:

- the app will run
- detection quality will be lower than a longer full-dataset training run

### 5. Start the app

```powershell
python -m streamlit run app.py
```

## How to use the app

### Single image mode

Use this when you want to inspect one image.

You can provide the image in two ways:

- upload an image through the app
- paste a local file path into the app

The app then:

- detects pieces
- maps them to board squares
- highlights matching pieces and/or squares
- reports whether the detected position matches one of the built-in openings

### Folder search mode

Use this when you want to search many images.

Point the app to a folder and it will scan the images there. A common default is:

- `data/chess-pieces/images/val`

You can filter by:

- piece type
- piece color
- board square
- opening

### Highlight behavior

- piece filters highlight only matching pieces
- square filters always draw a white square outline
- opening filters do not draw highlights
- opening filters only affect search/match results

## Different ways to use the project

### 1. Notebook-only workflow

Use [notebooks/train_chess_yolo.ipynb](notebooks/train_chess_yolo.ipynb).

Use this if you want:

- Colab
- Kaggle
- Jupyter
- a cell-by-cell training workflow

### 2. Terminal/local workflow

Use the Python scripts directly:

```powershell
python scripts/prepare_ultralytics_ndjson.py --input chess-pieces.ndjson --output data/chess-pieces
python train_yolo.py --data data/chess-pieces/data.yaml --model yolo11n.pt --epochs 40 --imgsz 640
python -m streamlit run app.py
```

Use this if you want:

- a local VS Code workflow
- reproducible CLI commands
- no notebook dependency

### 3. App-only inference workflow

If you already have trained weights, you can skip training and just run the app.

Requirements:

- a valid YOLO weights file
- ideally `runs/detect/chess-pieces-gpu-90min-safe/weights/best.pt`

Then run:

```powershell
python -m streamlit run app.py
```

Note:

- if you point the app to a generic YOLO checkpoint instead of trained chess weights, the app may still run, but chess-piece detection quality will not be reliable
- the default app weights path is `runs/detect/chess-pieces-gpu-90min-safe/weights/best.pt`

## Troubleshooting

### The model seems confused or inaccurate

This is expected if you are using the quick-start checkpoint.

Why:

- the quick-start model was trained on a small subset
- it was only trained for a few epochs
- it is good enough to make the app runnable, but not good enough for high-quality chess detection

Best optimizations, in order of impact:

1. Train on the full dataset instead of the quick-start subset.
2. Train longer.
3. Use a larger model.
4. Use GPU training instead of CPU.
5. Use a slightly higher inference confidence in the app.
6. Keep the board orientation and board margin aligned correctly.

Recommended quality training command on CPU:

```powershell
python train_yolo.py --data data/chess-pieces/data.yaml --model yolo11s.pt --epochs 50 --imgsz 640 --batch 8 --workers 0 --device cpu --project runs/detect --name chess-pieces
```

Recommended quality training command on GPU:

```powershell
python train_yolo.py --data data/chess-pieces/data.yaml --model yolo11s.pt --epochs 50 --imgsz 640 --batch 16 --workers 8 --device 0 --project runs/detect --name chess-pieces
```

Practical notes:

- `yolo11n.pt` is the fastest option but also the weakest
- `yolo11s.pt` is a better quality/speed tradeoff for this project
- the app now uses stricter inference defaults and keeps only the highest-confidence detection per board square
- if the model still over-detects, raise the app confidence slider to `0.50` or `0.60`

### The app says it cannot load YOLO weights

Cause:

- `runs/detect/chess-pieces-gpu-90min-safe/weights/best.pt` does not exist yet
- or the weights path in the sidebar is wrong

Fix:

- train the model first
- or set the sidebar weights path to an existing `.pt` file

### `streamlit` is not recognized or looks uninstalled

Cause:

- the `streamlit` package is installed, but the Scripts directory is not on PATH
- or VS Code / the terminal is using a different Python interpreter than the one where `streamlit` was installed

Fix:

- run the app with:

```powershell
python -m streamlit run app.py
```

- check which interpreter is active:

```powershell
python -c "import sys; print(sys.executable)"
```

- verify Streamlit in that same interpreter:

```powershell
python -m pip show streamlit
```

- if it is missing in that interpreter, reinstall requirements:

```powershell
python -m pip install -r requirements.txt
```

### Dataset preparation was interrupted

Fix:

- rerun the same preparation command

```powershell
python scripts/prepare_ultralytics_ndjson.py --input chess-pieces.ndjson --output data/chess-pieces
```

Check that this file exists when the process is done:

- `data/chess-pieces/data.yaml`

### Training is very slow

Cause:

- PyTorch is running on CPU

Fix:

- use the current CPU setup and wait longer
- or install a CUDA-enabled PyTorch build and rerun training

### Squares do not line up correctly

Fix:

- change the board orientation in the app
- increase the board margin slider until the square mapping matches the board in the image

### Opening detection is wrong

Possible causes:

- the detector missed a piece
- a piece was assigned to the wrong square
- the position does not exactly match one of the built-in opening templates

Important:

- opening recognition is template-based
- it only matches the built-in opening rules in `src/chessvision/openings.py`

## Notes for another user

If someone else needs to run this project on a new machine, the minimum correct order is:

1. Clone the repo.
2. Create and activate a virtual environment.
3. Install `requirements.txt`.
4. Run dataset preparation until `data/chess-pieces/data.yaml` exists.
5. Train until you have a valid `best.pt`, or use the published `runs/detect/chess-pieces-gpu-90min-safe/weights/best.pt`.
6. Run `python -m streamlit run app.py`.

If steps 4 or 5 are skipped, the app may open but it will not have the trained chess model it expects.
