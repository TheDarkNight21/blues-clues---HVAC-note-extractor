import os
import json
import cv2
from tqdm import tqdm
from PIL import Image

DOWNSCALED_JSON_PATH = "datasets/blueprint_notes_downscaled/annotations/instances_val.json"
DOWNSCALED_IMAGE_DIR = "datasets/blueprint_notes_downscaled/val"
ORIGINAL_DIRS = [
    "datasets/blueprint_notes/train",
    "datasets/blueprint_notes/val"
]
OUTPUT_JSON_PATH = "datasets/blueprint_notes_upscaled/annotations/instances_val_upscaled.json"
VISUALIZED_DIR = "datasets/blueprint_notes_upscaled/visualized"

os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
os.makedirs(VISUALIZED_DIR, exist_ok=True)

def find_original_image(filename):
    for dir_path in ORIGINAL_DIRS:
        for root, _, files in os.walk(dir_path):
            for f in files:
                if filename in f or f in filename:
                    return os.path.join(root, f)
    return None

def load_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  # returns (width, height)

def upscale_bbox(bbox, scale_x, scale_y):
    x, y, w, h = bbox
    return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]

def visualize(image_path, bboxes, categories, output_path):
    img = cv2.imread(image_path)
    for bbox, cat in zip(bboxes, categories):
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(cat), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)

def main():
    print("Loading downscaled annotations...")
    with open(DOWNSCALED_JSON_PATH, "r") as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    image_id_to_annotations = {}
    for ann in data["annotations"]:
        image_id_to_annotations.setdefault(ann["image_id"], []).append(ann)

    new_images = []
    new_annotations = []
    ann_id_counter = 1

    print("Processing annotations and upscaling...")
    for img in tqdm(data["images"]):
        downscaled_path = os.path.join(DOWNSCALED_IMAGE_DIR, img["file_name"])
        original_path = find_original_image(img["file_name"])

        if original_path is None:
            print(f"Original image not found for: {img['file_name']}")
            continue

        down_w, down_h = img["width"], img["height"]
        orig_w, orig_h = load_image_dimensions(original_path)

        scale_x = orig_w / down_w
        scale_y = orig_h / down_h

        new_img_id = img["id"]
        new_images.append({
            "id": new_img_id,
            "file_name": os.path.basename(original_path),
            "width": orig_w,
            "height": orig_h
        })

        annotations = image_id_to_annotations.get(img["id"], [])
        upscaled_bboxes = []
        categories = []

        for ann in annotations:
            up_bbox = upscale_bbox(ann["bbox"], scale_x, scale_y)
            new_annotations.append({
                "id": ann_id_counter,
                "image_id": new_img_id,
                "category_id": ann["category_id"],
                "bbox": up_bbox,
                "area": up_bbox[2] * up_bbox[3],
                "iscrowd": ann.get("iscrowd", 0)
            })
            ann_id_counter += 1
            upscaled_bboxes.append(up_bbox)
            categories.append(ann["category_id"])

        vis_output = os.path.join(VISUALIZED_DIR, f"vis_{os.path.basename(original_path)}")
        visualize(original_path, upscaled_bboxes, categories, vis_output)

    print("Saving upscaled annotations...")
    new_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": data["categories"]
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Saved new COCO JSON to {OUTPUT_JSON_PATH}")
    print(f"Saved visualizations to {VISUALIZED_DIR}")

if __name__ == "__main__":
    main()
