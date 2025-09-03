import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

from functions_for_api import *

def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Use same training settings
    cfg.MODEL.DEVICE = "cpu"   # change to "cuda" if GPU available
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  

    # Path to trained weights
    cfg.MODEL.WEIGHTS = os.path.join("./output_blueprints", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold for inference

    predictor = DefaultPredictor(cfg)
    return predictor

def predict_image(image_path, predictor):
    """
    Downscale an image before inference, run the model, then
    rescale predicted boxes back to the original image size.
    """
    # --- 1. Load original image dimensions ---
    orig_im = cv2.imread(image_path)
    orig_h, orig_w = orig_im.shape[:2]
    print("original height: " + str(orig_h) + "\n" + "original width: " + str(orig_w))

    # --- 2. Downscale the image (overwrite same path for simplicity) ---
    # Make a copy in memory
    output_path = "/output_image_storage"
    downscale_ratio = downscale_image(image_path, output_path)  # modifies in place
    im = cv2.imread(output_path)
    down_h, down_w = im.shape[:2]
    print("downscaled height: " + str(down_h) + "\n" + "downscaled width: " + str(down_w))

    # --- 3. Run model on downscaled image ---
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    # --- 4. Compute rescale factors ---
    # If downscale_ratio = 0.5, then original was 2x bigger
    scale_x = orig_w / down_w
    scale_y = orig_h / down_h
    print("rescale factors" + "\n" + "x: " +str(scale_x) + "\n" + "y:" + str(scale_y))

    # --- 5. Rescale boxes back to original resolution ---
    # boxes are in format [x1, y1, x2, y2]
    boxes_rescaled = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_rescaled.append([
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        ])

    return boxes_rescaled, scores, classes

def visualize_predictions(image_path, boxes, classes, scores, output_path="/output_image_storage", class_names=None, score_threshold=0.5):
    """
    Visualize predictions (rescaled boxes) on the original image.

    Args:
        image_path (str): Path to original image.
        boxes (list): List of [x1, y1, x2, y2] bounding boxes.
        classes (list): List of class indices.
        scores (list): List of confidence scores.
        output_path (str, optional): If given, saves the visualized image.
        class_names (list, optional): Map from class ID -> class name.
        score_threshold (float): Only show boxes above this confidence.
    """
    img = cv2.imread(image_path)
    copy = img.copy()

    for bbox, cls, score in zip(boxes, classes, scores):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Draw rectangle
        cv2.rectangle(copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text: class name if available, else ID
        label = str(cls)
        if class_names and 0 <= cls < len(class_names):
            label = class_names[cls]
        label = f"{label}: {score:.2f}"

        # Draw label
        cv2.putText(copy, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save or show
    if output_path:
        cv2.imwrite(output_path, copy)
        print(f"Saved visualization to {output_path}")
    else:
        cv2.imshow("Predictions", copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    predictor = load_model()
    boxes, scores, classes = predict_image(r"C:\Users\Owner\PycharmProjects\HVAC-notes-extractorRound2\datasets\blueprint_notes\val\Firehouse Subs Tayor, MI_2024-10-8 Bid & Permit Submission_page_20.jpg", predictor)

    # visualize results on original image
    visualize_predictions(
        r"C:\Users\Owner\PycharmProjects\HVAC-notes-extractorRound2\datasets\blueprint_notes\val\Firehouse Subs Tayor, MI_2024-10-8 Bid & Permit Submission_page_20.jpg",
        boxes,
        classes,
        scores,
        output_path="vis_test_image.jpg",
        class_names=["General Note Section", "Sheet Name"],  # example if you have 2 classes
        score_threshold=0.6
    )