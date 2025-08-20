from PIL import Image
import json
import os
from collections import defaultdict

# --- Configuration ---
RESCALED_DATASET_ROOT = "../datasets/blueprint_notes_rescaled"  # Source (downscaled data)
ORIGINAL_DATASET_ROOT = "../datasets/blueprint_notes"  # Reference for original dimensions
UPSCALED_DATASET_ROOT = "../datasets/blueprint_notes_upscaled"  # Destination (upscaled data)


# --- Main Logic ---

def process_split(split):
    """
    Upscales images in a split back to their original dimensions and scales their corresponding COCO annotations.
    """
    # Define paths for rescaled, original, and upscaled data
    rescaled_ann_path = os.path.join(RESCALED_DATASET_ROOT, "annotations", f"instances_{split}.json")
    original_ann_path = os.path.join(ORIGINAL_DATASET_ROOT, "annotations", f"instances_{split}.json")
    upscaled_img_dir = os.path.join(UPSCALED_DATASET_ROOT, split)
    upscaled_ann_dir = os.path.join(UPSCALED_DATASET_ROOT, "annotations")

    # Create directories for the new upscaled dataset if they don't exist
    os.makedirs(upscaled_img_dir, exist_ok=True)
    os.makedirs(upscaled_ann_dir, exist_ok=True)

    # Load the rescaled COCO JSON (current state)
    with open(rescaled_ann_path, "r") as f:
        rescaled_coco = json.load(f)

    # Load the original COCO JSON to get original dimensions
    with open(original_ann_path, "r") as f:
        original_coco = json.load(f)

    # Create a map from image filename to original dimensions
    original_dimensions = {}
    for img_info in original_coco["images"]:
        original_dimensions[img_info["file_name"]] = {
            "width": img_info["width"],
            "height": img_info["height"]
        }

    # Create a map from image_id to its annotations for easy lookup
    annotations_by_img_id = defaultdict(list)
    if "annotations" in rescaled_coco:
        for ann in rescaled_coco["annotations"]:
            annotations_by_img_id[ann["image_id"]].append(ann)

    # Process each image
    for img_info in rescaled_coco["images"]:
        rescaled_img_path = os.path.join(RESCALED_DATASET_ROOT, split, img_info["file_name"])
        upscaled_img_path = os.path.join(upscaled_img_dir, img_info["file_name"])

        # Get original dimensions
        original_dims = original_dimensions[img_info["file_name"]]
        original_w = original_dims["width"]
        original_h = original_dims["height"]

        # Get current (rescaled) dimensions
        current_w = img_info["width"]
        current_h = img_info["height"]

        # Calculate the upscaling ratio
        # If the image was downscaled, we need to upscale it back
        ratio = 1.0
        if current_w != original_w or current_h != original_h:
            # Calculate ratio based on width (since downscaling was width-based)
            ratio = original_w / current_w

        # Load and process the image
        img = Image.open(rescaled_img_path)

        if ratio > 1.0:
            # Upscale the image back to original dimensions
            # Use Image.Resampling.LANCZOS for high-quality upsampling
            upscaled_img = img.resize((original_w, original_h), Image.Resampling.LANCZOS)
            upscaled_img.save(upscaled_img_path)

            # Update image info in the JSON to original dimensions
            img_info["width"] = original_w
            img_info["height"] = original_h

            print(
                f"Upscaled {img_info['file_name']}: {current_w}x{current_h} -> {original_w}x{original_h} (ratio: {ratio:.3f})")
        else:
            # If not upscaling, just copy the image to the new location
            img.save(upscaled_img_path)
            print(f"Copied {img_info['file_name']}: {current_w}x{current_h} (no upscaling needed)")

        # *** SCALE ANNOTATIONS BACK TO ORIGINAL SIZE ***
        # If the image was upscaled (ratio > 1.0), scale its annotations
        if ratio > 1.0:
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
    new_json_path = os.path.join(upscaled_ann_dir, f"instances_{split}.json")
    with open(new_json_path, "w") as f:
        # We can dump the whole coco dictionary because we modified its contents in-place
        json.dump(rescaled_coco, f, indent=2)

    print(f"Processed '{split}' split.")
    print(f"Upscaled images saved to: {upscaled_img_dir}")
    print(f"Updated JSON saved to: {new_json_path}\n")


def verify_upscaling(split):
    """
    Verify that upscaling worked correctly by comparing dimensions.
    """
    print(f"Verifying upscaling for '{split}' split...")

    # Load original and upscaled annotations
    original_ann_path = os.path.join(ORIGINAL_DATASET_ROOT, "annotations", f"instances_{split}.json")
    upscaled_ann_path = os.path.join(UPSCALED_DATASET_ROOT, "annotations", f"instances_{split}.json")

    with open(original_ann_path, "r") as f:
        original_coco = json.load(f)

    with open(upscaled_ann_path, "r") as f:
        upscaled_coco = json.load(f)

    # Create maps for comparison
    original_dims = {img["file_name"]: (img["width"], img["height"]) for img in original_coco["images"]}
    upscaled_dims = {img["file_name"]: (img["width"], img["height"]) for img in upscaled_coco["images"]}

    mismatches = 0
    for filename in original_dims:
        if filename in upscaled_dims:
            orig_dims = original_dims[filename]
            upsc_dims = upscaled_dims[filename]
            if orig_dims != upsc_dims:
                print(f"MISMATCH: {filename} - Original: {orig_dims}, Upscaled: {upsc_dims}")
                mismatches += 1
        else:
            print(f"MISSING: {filename} not found in upscaled dataset")
            mismatches += 1

    if mismatches == 0:
        print(f"✓ All images in '{split}' split have correct dimensions!")
    else:
        print(f"✗ Found {mismatches} dimension mismatches in '{split}' split")

    print()


# --- Run the processing ---
if __name__ == "__main__":
    print("Starting upscaling process...")
    print(f"Source (rescaled): {RESCALED_DATASET_ROOT}")
    print(f"Reference (original): {ORIGINAL_DATASET_ROOT}")
    print(f"Destination (upscaled): {UPSCALED_DATASET_ROOT}")
    print("-" * 50)

    process_split("train")
    process_split("val")

    print("All splits processed successfully.")
    print("-" * 50)

    # Verify the upscaling worked correctly
    verify_upscaling("train")
    verify_upscaling("val")

    print("Upscaling process completed!")