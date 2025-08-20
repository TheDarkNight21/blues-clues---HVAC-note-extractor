import os
from pdf2image import convert_from_path


def convert_pdfs_to_images(folder_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_name = os.path.splitext(filename)[0]  # Remove .pdf extension

            # Convert PDF pages to images
            pages = convert_from_path(pdf_path)

            for i, page in enumerate(pages):
                image_name = f"{pdf_name}_page_{i + 1}.jpg"
                image_path = os.path.join(output_folder, image_name)
                page.save(image_path, "JPEG")
                print(f"Saved: {image_path}")


# Example usage
convert_pdfs_to_images("/Users/darkknight/desktop", "/Users/darkknight/desktop/pages")
