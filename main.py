from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import cv2
import easyocr

from functions_for_api import *
from predict import *

# === FastAPI App ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Initialize EasyOCR Reader ===
reader = easyocr.Reader(['en'], gpu=True)  # Load once for speed


# === Utility Function: OCR Extraction ===
def extract_text_from_boxes(image_path, boxes, classes, scores):
    """
    Extract text from bounding boxes using EasyOCR.
    """
    image = cv2.imread(image_path)
    extracted = []

    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores), 1):
        # Box comes in as [x1, y1, x2, y2]
        x1, y1, x2, y2 = [int(v) for v in box]
        cropped = image[y1:y2, x1:x2]

        results = reader.readtext(cropped)
        text = " ".join([res[1] for res in results]) if results else "[No text found]"

        extracted.append({
            "box_id": i,
            "class_id": int(cls),
            "score": float(score),
            "bbox": [x1, y1, x2, y2],
            "text": text
        })

    return extracted


# === API Endpoint ===
@app.post("/api/extract-notes")
async def extract_notes(pdf_url: str = None, file: UploadFile = File(None)):
    # Validate input
    if not pdf_url and not file:
        raise HTTPException(status_code=400, detail="Either PDF URL or file must be provided")

    if pdf_url and file:
        raise HTTPException(status_code=400, detail="Provide either URL or file, not both")

    try:
        # === Load PDF ===
        if pdf_url:
            if not pdf_url.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="URL must point to a PDF file")
            pdf_bytes = await download_pdf_from_url(pdf_url)
        else:
            if file.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail="Only PDF files are supported")
            pdf_bytes = await file.read()

        # Convert PDF to images
        images = store_images(pdf_bytes)

        # Load model once
        predictor = load_model()

        pages_output = []

        # Process each page
        for page_num, image in enumerate(images, 1):
            # Run detection model
            boxes_rescaled, scores, classes = predict_image(image, predictor)

            # Save annotated visualization
            output_path = fr"C:\Users\Owner\PycharmProjects\HVAC-notes-extractorRound2\output_image_storage/{image}"
            visualize_predictions(image, boxes_rescaled, classes, scores, output_path=output_path)

            # OCR extraction
            ocr_results = extract_text_from_boxes(image, boxes_rescaled, classes, scores)

            pages_output.append({
                "page_number": page_num,
                "image_path": image,
                "ocr_results": ocr_results
            })

        # === Final JSON Response ===
        return JSONResponse(content={
            "status": "success",
            "size_bytes": len(pdf_bytes),
            "pages": pages_output
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


# === Entry Point ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)