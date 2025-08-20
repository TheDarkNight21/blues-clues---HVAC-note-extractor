import os
import shutil
import json
import random
from pathlib import Path

# === CONFIG ===
LABEL_STUDIO_JSON = "result.json"
IMAGE_SOURCE_DIR = "/Users/darkknight/Library/Application Support/label-studio/media/upload/1/"
DEST_ROOT = "datasets/blueprint_notes"
TRAIN_RATIO = 0.8

# === SETUP ===
os.makedirs(f"{DEST_ROOT}/annotations", exist_ok=True)
os.makedirs(f"{DEST_ROOT}/train", exist_ok=True)
os.makedirs(f"{DEST_ROOT}/val", exist_ok=True)

# === LOAD AND CLEAN JSON ===
with open(LABEL_STUDIO_JSON, "r") as f:
    coco = json.load(f)

all_images = coco["images"]
random.shuffle(all_images)
split_idx = int(len(all_images) * TRAIN_RATIO)

train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def process_split(images, split_name):
    new_images = []
    new_annotations = []

    image_ids = {img["id"] for img in images}

    for img in images:
        old_path = os.path.join(IMAGE_SOURCE_DIR, os.path.basename(img["file_name"]))
        new_path = f"{DEST_ROOT}/{split_name}/{os.path.basename(img['file_name'])}"
        img["file_name"] = os.path.basename(img["file_name"])

        if not os.path.exists(old_path):
            print(f"⚠️ Image not found: {old_path}")
            continue

        shutil.copyfile(old_path, new_path)
        new_images.append(img)

    for ann in coco["annotations"]:
        if ann["image_id"] in image_ids:
            new_annotations.append(ann)

    out_json = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"]
    }

    with open(f"{DEST_ROOT}/annotations/instances_{split_name}.json", "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"✅ Wrote {split_name}: {len(new_images)} images, {len(new_annotations)} annotations")

process_split(train_images, "train")
process_split(val_images, "val")

