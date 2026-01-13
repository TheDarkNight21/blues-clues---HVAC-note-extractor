# Blues Clues — HVAC Note Extractor

Detectron2 + OCR pipeline to detect and extract handwritten / printed HVAC notes from blueprint PDF pages.

- Primary language: Python
- Repo ID: 1021198103

---

Table of contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure & Key Files](#repository-structure--key-files)
- [Architecture & Data Flow](#architecture--data-flow)
- [Dataset Layout](#dataset-layout)
- [Quickstart / Setup](#quickstart--setup)
- [Training](#training)
- [Inference (single image / PDF)](#inference-single-image--pdf)
- [API (FastAPI)](#api-fastapi)
- [Utilities & Preprocessing](#utilities--preprocessing)
- [Troubleshooting & Notes](#troubleshooting--notes)
- [Example usage](#example-usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project overview

This project is a pipeline for detecting "notes" (textual annotations, callouts, and handwritten notes) on building blueprints and extracting their text using OCR. It uses Detectron2 for object detection (to find note bounding boxes) and EasyOCR for extracting text inside detected boxes. The project includes dataset preparation helpers, training scripts, inference/prediction utilities, PDF-to-image conversion routines, and a FastAPI wrapper to expose an HTTP endpoint.

Goals:
- Detect regions on blueprint pages that contain notes.
- Crop those regions and run OCR for text extraction.
- Provide an API that accepts a PDF (URL or upload) and returns extracted text per page.

---

## Features

- Detectron2-based object detection training and inference
- PDF -> image conversion (PyMuPDF / fitz)
- OCR extraction using EasyOCR
- Scripts for dataset preparation from Label Studio exports
- Upscaling / rescaling utilities to convert annotations back to original image sizes
- Visualization helpers to save annotated images for debugging
- FastAPI endpoint to run the end-to-end pipeline on PDFs

---

## Repository structure & key files

High-level important files and their purpose:

- `main.py` — FastAPI app exposing `/api/extract-notes` which accepts a PDF (URL or file), converts pages to images, runs the detection model, extracts OCR text, and returns JSON.
- `predict.py` — Load Detectron2 model and run predictions on images; includes visualization helpers.
- `train_detectron.py` — Register COCO datasets and start training using Detectron2's `DefaultTrainer`.
- `visualize.py` — Run inference on the validation set and save visualized outputs.
- `functions_for_api.py` — Helper functions used by `main.py`: PDF-to-image conversion, image storage, downloading PDF from URL, etc.
- `functions_for_api.py` (contains `pdf_bytes_to_images_with_save`) — creates images using PyMuPDF (fitz).
- `pre_process_data/prepare_detectron_dataset.py` — Convert Label Studio JSON export (`result.json`) into train/val COCO-style dirs and annotation files.
- `test_model.py` — Upscale predicted/rescaled bounding boxes back to original image sizes and visualize results.
- `extract_text_from_bounding_boxes.py` — Standalone script to run EasyOCR on bounding boxes stored in an upscaled COCO annotation file.
- `run.py` — Helper script (installers / automation).
- `code.txt` — Step-by-step guide (environment setup, reproducible steps) and notes (useful commands and recommendations).

---

## Architecture & data flow

1. Input PDF (uploaded or URL).
2. Convert PDF pages to images (PyMuPDF).
3. (Optional) Pre-process images and annotations into COCO format for training.
4. Train Detectron2 model to detect note regions (two classes used in repo).
5. Run inference on images -> produce bounding boxes + class + score.
6. Crop box regions and run EasyOCR to extract text.
7. Return structured JSON per page with OCR results and optionally save visualized images.

---

## Dataset layout

Recommended repository dataset layout (based on guides in repo):

HVAC_project/
- datasets/
  - blueprint_notes_rescaled/
    - annotations/
      - instances_train.json
      - instances_val.json
    - train/
      - image1.jpg
      - ...
    - val/
      - image2.jpg
      - ...
- datasets/blueprint_notes/ (originals)
  - train/
  - val/

Files such as `prepare_detectron_dataset.py` expect a Label Studio export `result.json` and use a source image directory (configurable at top of the script).

---

## Quickstart / Setup

The repository includes a detailed environment setup in `code.txt`. A condensed, reproducible setup:

1. Create and activate Conda environment (recommended):
   - conda create -n d2_project_env python=3.9 -y
   - conda activate d2_project_env

2. Install PyTorch (choose the version that matches your CUDA). Example for CUDA 11.8:
   - pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

   Or CPU-only:
   - pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

3. Install Detectron2 (follow official instructions matching your torch/cuda):
   - pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torchX.Y/index.html
   (Replace cu118 and torchX.Y with your CUDA and torch versions; using the official page is recommended.)

4. Install other dependencies:
   - pip install opencv-python easyocr pymupdf tqdm pillow fastapi uvicorn requests

   If using macOS, set cfg.DATALOADER.NUM_WORKERS = 0 in training configs as the repo suggests.

5. Optional: Install Tesseract for fallback OCR or additional tools (not required if you use EasyOCR).

Notes:
- EasyOCR can use GPU; set `gpu=True` when creating the reader in `main.py` or `extract_text_from_bounding_boxes.py` if CUDA is available.
- Keep NumPy at <=1.25 or stable versions compatible with Detectron2 / PyTorch if you run into issues (repo notes mention avoiding NumPy 2.0 regressions).

---

## Training

1. Prepare datasets in COCO format using `pre_process_data/prepare_detectron_dataset.py` (edit config top variables to match your environment).
2. Use the rescaled dataset paths registered in `train_detectron.py`:
   - `register_coco_instances("blueprint_train", {}, "datasets/blueprint_notes_rescaled/annotations/instances_train.json", "datasets/blueprint_notes_rescaled/train")`
3. Edit training hyperparameters directly in `train_detectron.py`:
   - `cfg.SOLVER.MAX_ITER`, `cfg.SOLVER.IMS_PER_BATCH`, `cfg.SOLVER.BASE_LR`, `cfg.MODEL.ROI_HEADS.NUM_CLASSES` (set to number of classes).
4. Run:
   - python train_detectron.py
5. Outputs:
   - Trained weights and logs saved to `cfg.OUTPUT_DIR` (defaults to `./output_blueprints` or similar).
   - Use TensorBoard to visualize:
     - tensorboard --logdir ./output_blueprints

Important hints found in the repo:
- The code sets devices with `cfg.MODEL.DEVICE = "cuda"` (or CPU as needed).
- For macOS MPS quirks, the repo comments suggest forcing CPU or CUDA for stability in some setups.

---

## Inference (single image / PDF)

- Single image (scripted):
  - Use `predict.py` to load the model and predict bounding boxes for a given image.
  - `visualize_predictions` will draw boxes on the image and save output.

- PDF (end-to-end via API):
  - `main.py` exposes `/api/extract-notes`. It:
    - accepts `file` (multipart upload) or `pdf_url` (URL to PDF),
    - converts PDF pages to images using `functions_for_api.py`,
    - runs model/`predict_image`,
    - uses EasyOCR to extract text from detected boxes,
    - returns JSON with `pages` array containing per-page `ocr_results`.

Returned JSON structure (example):
{
  "status": "success",
  "size_bytes": 12345,
  "pages": [
    {
      "page_number": 1,
      "image_path": "path/to/page_1.jpg",
      "ocr_results": [
        {"bbox": [x1,y1,x2,y2], "text": "example note", "score": 0.92, "class": 1},
        ...
      ]
    }, ...
  ]
}

---

## API (FastAPI)

Start the server:
- uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
- POST /api/extract-notes
  - Params:
    - pdf_url: string (optional) — URL pointing to a PDF
    - file: multipart file (optional) — PDF upload
  - Returns JSON with per-page OCR results and page metadata.

Example multipart upload (curl):
- curl -F "file=@blueprint.pdf" http://localhost:8000/api/extract-notes

Notes:
- `main.py` checks `file.content_type == "application/pdf"` for uploads.
- `functions_for_api.py` contains helper to convert PDF bytes to images using PyMuPDF (`fitz`).

---

## Utilities & preprocessing

- prepare_detectron_dataset.py
  - Converts Label Studio JSON `result.json` into train/val splits, copies images and builds COCO-style annotations.

- test_model.py
  - Locates original images for rescaled predictions and upscales bounding boxes back to original size. Also saves an upscaled COCO JSON.

- extract_text_from_bounding_boxes.py
  - Walks through upscaled annotations and runs EasyOCR to extract text from each bounding box, saving results to `ocr_output.txt`.

- visualize.py / visualize_validation_results()
  - Re-register validation dataset, loads `model_final.pth`, runs predictions and saves visualizations for manual inspection.

---

## Troubleshooting & notes

- NumPy compatibility: the repository guides mention avoiding breaking NumPy versions (NumPy 2.0 issues). If Detectron2 or PyTorch fails after pip installs, check NumPy and PyTorch compatibility.
- macOS: `cfg.DATALOADER.NUM_WORKERS = 0` recommended for stability.
- Device selection: Some files hardcode `cfg.MODEL.DEVICE = "cuda"`. If CUDA not available, set to `"cpu"`.
- Detectron2 install: Must match PyTorch + CUDA. Use official Detectron2 wheel URLs.
- If EasyOCR is slow on CPU, consider using GPU (set `gpu=True`).
- Paths in examples are Windows absolute paths — update to local paths before running (e.g., `output_image_storage`).
- If you get MPS / macOS GPU errors with Detectron2, fallback to CPU training/inference.

---

## Example usage

- Train model:
  - python train_detectron.py

- Run FastAPI server locally:
  - uvicorn main:app --host 0.0.0.0 --port 8000 --reload

- Run inference on a saved image (scripted):
  - python predict.py  # depending on how predict.py is written you may pass an image path

- Run OCR extraction on upscaled annotation file:
  - python extract_text_from_bounding_boxes.py

---

## Contributing

- Tidy up the repo: add a `requirements.txt` or `environment.yml` capturing tested versions.
- Add CI checks for linting and basic unit tests.
- Add example dataset subset and sample PDF for integration tests.
- Improve docs: include a small reproducible demo and model checkpoint for quick validation.

---

## License

Add a license file (e.g., MIT) if you want to allow reuse. Currently no license file is included in the repo; add `LICENSE` to make terms explicit.

---

## Contact

Author: TheDarkNight21 (GitHub)
- Repo: https://github.com/TheDarkNight21/blues-clues---HVAC-note-extractor

---

Changelog / Notes:
- This README was generated from analysis of the repository files: `main.py`, `train_detectron.py`, `predict.py`, `visualize.py`, `functions_for_api.py`, `pre_process_data/prepare_detectron_dataset.py`, `extract_text_from_bounding_boxes.py`, `test_model.py`, and `code.txt` which contains environment setup steps and guidance.
- Next recommended steps: add `requirements.txt`, include a small `sample_data/` with at least one PDF and pre-trained model exemplar and add `LICENSE`.
