from PIL import Image
import json
import os
from collections import defaultdict

# --- Configuration ---
MAX_WIDTH = 2000
ORIGINAL_DATASET_ROOT = "datasets/blueprint_notes"
RESCALED_DATASET_ROOT = "datasets/blueprint_notes_downscaled"  # Save new data here


# --- Main Logic ---

def process_split(split):
    """
    Resizes images in a split and scales their corresponding COCO annotations.
    """
    # Define paths for original and new data
    original_ann_path = os.path.join(ORIGINAL_DATASET_ROOT, "annotations", f"instances_{split}.json")
    rescaled_img_dir = os.path.join(RESCALED_DATASET_ROOT, split)
    rescaled_ann_dir = os.path.join(RESCALED_DATASET_ROOT, "annotations")

    # Create directories for the new rescaled dataset if they don't exist
    os.makedirs(rescaled_img_dir, exist_ok=True)
    os.makedirs(rescaled_ann_dir, exist_ok=True)

    # Load the original COCO JSON
    with open(original_ann_path, "r") as f:
        coco = json.load(f)

    # Create a map from image_id to its annotations for easy lookup
    # This is much more efficient than searching the whole list every time.
    annotations_by_img_id = defaultdict(list)
    if "annotations" in coco:
        for ann in coco["annotations"]:
            annotations_by_img_id[ann["image_id"]].append(ann)

    # Process each image
    for img_info in coco["images"]:
        original_img_path = os.path.join(ORIGINAL_DATASET_ROOT, split, img_info["file_name"])
        rescaled_img_path = os.path.join(rescaled_img_dir, img_info["file_name"])

        img = Image.open(original_img_path)
        w, h = img.size

        # Assume no scaling is needed initially
        ratio = 1.0

        # If image is too wide, calculate new dimensions and scaling ratio
        if w > MAX_WIDTH:
            ratio = MAX_WIDTH / w
            new_w = MAX_WIDTH
            new_h = int(h * ratio)

            # Resize and save the image to the new directory
            # Use Image.Resampling.LANCZOS for high-quality downsampling
            resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_img.save(rescaled_img_path)

            # Update image info in the JSON
            img_info["width"] = new_w
            img_info["height"] = new_h
        else:
            # If not resizing, just copy the original image to the new location
            img.save(rescaled_img_path)

        # *** THIS IS THE CRITICAL FIX ***
        # If the image was resized (ratio < 1.0), scale its annotations
        if ratio < 1.0:
            image_id = img_info["id"]
            if image_id in annotations_by_img_id:
                for ann in annotations_by_img_id[image_id]:
                    # Scale the bounding box [x, y, width, height]
                    if "bbox" in ann:
                        bbox = ann["bbox"]
                        ann["bbox"] = [
                            bbox[0] * ratio,
                            bbox[1] * ratio,
                            bbox[2] * ratio,
                            bbox[3] * ratio
                        ]

                    # Scale the segmentation polygons [[x1, y1, x2, y2, ...]]
                    if "segmentation" in ann:
                        new_seg = []
                        for seg_part in ann["segmentation"]:
                            # seg_part is a list of coordinates like [x1, y1, x2, y2, ...]
                            new_seg_part = [(coord * ratio) for coord in seg_part]
                            new_seg.append(new_seg_part)
                        ann["segmentation"] = new_seg

    # Save the updated COCO JSON to the new annotations directory
    new_json_path = os.path.join(rescaled_ann_dir, f"instances_{split}.json")
    with open(new_json_path, "w") as f:
        # We can dump the whole coco dictionary because we modified its contents in-place
        json.dump(coco, f, indent=2)

    print(f"Processed '{split}' split.")
    print(f"Rescaled images saved to: {rescaled_img_dir}")
    print(f"Updated JSON saved to: {new_json_path}\n")


# --- Run the processing ---
if __name__ == "__main__":
    process_split("train")
    process_split("val")
    print("All splits processed successfully.")