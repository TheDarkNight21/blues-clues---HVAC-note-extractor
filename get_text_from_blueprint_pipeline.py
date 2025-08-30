# goal: make function that gets a blueprint, runs the model and gets the output, and returns the output text.

import io
import PyPDF2
import os
from pathlib import Path
import fitz


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


