import os
import json 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


register_coco_instances("my_dataset_train", {},
                        "/home/ubuntu/Documents/IDLE/Detectron 2/Pecan Detection/Dataset/small-cell1/small-cell1.json",
                        "/home/ubuntu/Documents/IDLE/Detectron 2/Pecan Detection/Dataset/small-cell1")
register_coco_instances("my_dataset_test", {},
                        "/home/ubuntu/Documents/IDLE/Detectron 2/Pecan Detection/Dataset/small-cell1/small-cell1.json",
                        "/home/ubuntu/Documents/IDLE/Detectron 2/Pecan Detection/Dataset/small-cell1")


# visualize training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

#configuring the training settings
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 0

# Set the model and basic configurations
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")
cfg.INPUT.MAX_SIZE_TRAIN = 6500
cfg.INPUT.MAX_SIZE_TEST = 6500
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 3500 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.TEST.EVAL_PERIOD = 500
cfg.SOLVER.STEPS = (1200, 2400)
cfg.SOLVER.WARMUP_ITERS = 50
cfg.OUTPUT_DIR = "/home/ubuntu/Documents/IDLE/Detectron 2/Pecan Detection/output/small_cell"

# Start training
os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
trainer =  DefaultTrainer(cfg)
trainer.resume_or_load(resume = False)
trainer.train()

# Testing the model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TEST = ("my_dataset_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("my_dataset_test")

from detectron2.utils.visualizer import ColorMode
import glob

for imageName in random.sample(glob.glob('/home/ubuntu/Documents/IDLE/Detectron 2/Pecan Detection/data/test/*JPG'),5):
    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata,
                   scale=0.8)
    
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Output",out.get_image()[:, :, ::-1])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
