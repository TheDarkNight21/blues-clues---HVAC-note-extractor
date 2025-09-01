from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import torch

def main():
    """
    Main function to set up config and run training.
    """
    # --- 1. Register Datasets ---
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("blueprint_train", {}, "datasets/blueprint_notes_rescaled/annotations/instances_train.json", "datasets/blueprint_notes_rescaled/train")
    register_coco_instances("blueprint_val", {}, "datasets/blueprint_notes_rescaled/annotations/instances_val.json", "datasets/blueprint_notes_rescaled/val")

    # --- 2. Setup Configuration ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # *** THE FIX: Force CPU to avoid MPS backend bugs ***
    cfg.MODEL.DEVICE = "cuda"
    print(f"Using device: {cfg.MODEL.DEVICE}")

    cfg.DATASETS.TRAIN = ("blueprint_train",)
    cfg.DATASETS.TEST = ("blueprint_val",)
    cfg.DATALOADER.NUM_WORKERS = 0 # Keep at 0 for macOS stability

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    cfg.SOLVER.LOG_PERIOD = 50
    cfg.TEST.EVAL_PERIOD = 500

    cfg.OUTPUT_DIR = "./output_blueprints"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # --- 3. Train the Model ---
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("\nTraining finished.")
    print(f"To see training graphs, run the following command in your terminal:")
    print(f"tensorboard --logdir {cfg.OUTPUT_DIR}")


if __name__ == '__main__':
    main()