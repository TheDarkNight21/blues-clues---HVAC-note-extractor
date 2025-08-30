from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import uvicorn

from get_text_from_blueprint_pipeline import get_text_from_blueprint

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def download_pdf_from_url(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx, 5xx)
        return response.content  # Returns PDF as bytes
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")

@app.post("/api/extract-notes")
async def extract_notes(pdf_url: str = None, file: UploadFile = File(None)):
    # Check if either URL or file is provided
    if not pdf_url and not file:
        raise HTTPException(status_code=400, detail="Either PDF URL or file must be provided")

    if pdf_url and file:
        raise HTTPException(status_code=400, detail="Provide either URL or file, not both")

    try:
        if pdf_url:
            # Handle PDF from URL
            if not pdf_url.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="URL must point to a PDF file")
            pdf_bytes = await download_pdf_from_url(pdf_url)
            images = get_text_from_blueprint(pdf_bytes)
        else:
            # Handle uploaded file
            if file.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail="Only PDF files are supported")
            pdf_bytes = await file.read()
            images = get_text_from_blueprint(pdf_bytes)

        # Now you can process pdf_bytes with your function
        # For example: result = process_pdf(pdf_bytes)

        return JSONResponse(content={
            "status": "success",
            "size_bytes": len(pdf_bytes),
            "image1": images[0],
            "image2": images[1]
            # Add your processed data here
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
