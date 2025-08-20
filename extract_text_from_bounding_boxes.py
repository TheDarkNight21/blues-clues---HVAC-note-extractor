import os
import json
import cv2
import easyocr
from tqdm import tqdm

# === Paths ===
UPSCALED_JSON_PATH = "datasets/blueprint_notes_upscaled/annotations/instances_val_upscaled.json"
ORIGINAL_IMAGE_DIRS = [
    "datasets/blueprint_notes/val",
    "datasets/blueprint_notes/train"
]

# === Output Path ===
OUTPUT_TEXT_PATH = "ocr_output.txt"

# === Initialize EasyOCR Reader ===
reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=True if CUDA is available

# === Utility Functions ===
def find_image(image_name):
    for root_dir in ORIGINAL_IMAGE_DIRS:
        full_path = os.path.join(root_dir, image_name)
        if os.path.exists(full_path):
            return full_path
    return None

def extract_text_from_box(image, bbox):
    x, y, w, h = [int(coord) for coord in bbox]
    cropped = image[y:y + h, x:x + w]
    results = reader.readtext(cropped)
    if results:
        return " ".join([res[1] for res in results])
    return "[No text found]"

# === Main ===
def main():
    with open(UPSCALED_JSON_PATH, "r") as f:
        data = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    results = []

    grouped_annotations = {}
    for ann in data["annotations"]:
        grouped_annotations.setdefault(ann["image_id"], []).append(ann)

    print("Starting OCR extraction...")
    for image_id, anns in tqdm(grouped_annotations.items()):
        image_name = image_id_to_file[image_id]
        image_path = find_image(image_name)
        if image_path is None:
            print(f"Image not found: {image_name}")
            continue

        image = cv2.imread(image_path)
        results.append(f"{image_name}")

        for i, ann in enumerate(anns, 1):
            bbox = ann["bbox"]
            text = extract_text_from_box(image, bbox)
            results.append(f"  bounding box {i}: {text if text else '[No text found]'}")

        results.append("")

    print(f"Saving OCR results to: {OUTPUT_TEXT_PATH}")
    with open(OUTPUT_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(results))


if __name__ == "__main__":
    main()