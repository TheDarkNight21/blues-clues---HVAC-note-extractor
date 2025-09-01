# goal: make function that gets a blueprint, runs the model and gets the output, and returns the output text.
import io
import PyPDF2
import os
from pathlib import Path
import fitz

import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException


def pdf_bytes_to_images_with_save(
        pdf_bytes,
        output_folder,
        pdf_name="document",
        dpi=300,
        fmt='jpeg'
):
    """
    Convert PDF bytes to images and save them to the specified output folder using PyMuPDF.

    Args:
        pdf_bytes (bytes): PDF file as bytes
        output_folder (str): Path to the output folder
        pdf_name (str): Base name for the output files
        dpi (int): Rendering resolution for the output images
        fmt (str): Output format ('png' or 'jpeg')

    Returns:
        list: List of paths to the saved images
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Open PDF from bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    saved_paths = []

    for i, page in enumerate(doc, 1):
        # Render page to a pixmap
        zoom = dpi / 72  # 72 is the default resolution in PDFs
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # File naming convention: {pdf_name}_page_{number}.{ext}
        filename = f"{pdf_name}_page_{i}.{fmt.lower()}"
        filepath = os.path.join(output_folder, filename)

        # Save as chosen format
        pix.save(filepath)
        saved_paths.append(filepath)

    return saved_paths


def get_text_from_blueprint(pdf_bytes):
    image_paths_list = pdf_bytes_to_images_with_save(
        pdf_bytes,
        output_folder="./image_storage",
        pdf_name="file_name",
        dpi=300,
        fmt='jpeg'
    )
    return image_paths_list


async def download_pdf_from_url(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx, 5xx)
        return response.content  # Returns PDF as bytes
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading PDF: {str(e)}")


MAX_WIDTH = 2000
# MAX_HEIGHT = 2000  # optional, can remove if you only care about width

def downscale_image(input_path: str, output_path: str) -> float:
    """
    Downscales a JPEG image so its width/height do not exceed MAX_WIDTH/HEIGHT.
    Saves the downscaled image to `output_path`.
    """
    with Image.open(input_path) as img:
        w, h = img.size
        ratio = 1.0

        # compute resize ratio if image exceeds limits
        if w > MAX_WIDTH:
            ratio = MAX_WIDTH / w
            new_w, new_h = int(w * ratio), int(h * ratio)

            # high-quality downscale
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img.save(output_path, format="JPEG")

        return ratio  # return scaling ratio (1.0 if unchanged)


