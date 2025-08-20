import os
import cv2
import random
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo


def visualize_validation_results():
    """
    This function loads a trained model and runs it on images from the validation set,
    saving the predicted bounding boxes to files.
    """
    # === 1. Re-register your validation dataset ===
    BASE_DIR = "datasets/blueprint_notes_downscaled"
    VAL_PATH = os.path.join(BASE_DIR, "val")
    VAL_JSON = os.path.join(BASE_DIR, "annotations", "instances_val.json")
    register_coco_instances("blueprint_val_visualize", {}, VAL_JSON, VAL_PATH)

    # === 2. Set up the configuration for inference ===
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = os.path.join("model_output", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    # === 3. Create the Predictor ===
    predictor = DefaultPredictor(cfg)

    # === 4. Get the data for visualization ===
    metadata = MetadataCatalog.get("blueprint_val_visualize")
    dataset_dicts = DatasetCatalog.get("blueprint_val_visualize")

    # Create model_output directory for saved images
    output_dir = "./visualization_output"
    os.makedirs(output_dir, exist_ok=True)

    # === 5. Loop over random images and save predictions ===
    print("Processing 5 random images from the validation set.")
    print(f"Saving results to {output_dir}")

    for i, d in enumerate(random.sample(dataset_dicts, 5)):
        print(f"Processing image {i + 1}/5...")

        # Load the image
        im = cv2.imread(d["file_name"])

        # Pass the image to the predictor
        outputs = predictor(im)

        # Use Detectron2's Visualizer to draw the predictions on the image
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Save the resulting image
        output_filename = os.path.join(output_dir, f"prediction_{i + 1}.jpg")
        cv2.imwrite(output_filename, out.get_image()[:, :, ::-1])
        print(f"Saved: {output_filename}")

        # Print prediction details
        instances = outputs["instances"]
        num_detections = len(instances)
        print(f"  Found {num_detections} detections")

        if num_detections > 0:
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            for j, (score, cls) in enumerate(zip(scores, classes)):
                class_name = metadata.thing_classes[cls]
                print(f"    Detection {j + 1}: {class_name} (confidence: {score:.2f})")

    print("Visualization finished. Check the model_output directory for saved images.")


if __name__ == '__main__':
    visualize_validation_results()