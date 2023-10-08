from dotenv import find_dotenv, load_dotenv
import os
import sys
from pathlib import Path
import json
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Read input data
DATA_FOLDER = os.getenv('DATA_FOLDER')
HOME_FOLDER = os.getenv('HOME_FOLDER')

sys.stdout = open(os.path.join(os.getenv('HOME_FOLDER'), 'reports', 'training_log.txt'), 'w')

register_coco_instances("my_dataset_train", {}, Path(DATA_FOLDER) / 'external' / 'train' / 'annotations' / 'train.json', Path(DATA_FOLDER) / 'external' / 'train' / 'images')
register_coco_instances("my_dataset_val", {}, Path(DATA_FOLDER) / 'external' / 'train' / 'annotations' / 'val.json', Path(DATA_FOLDER) / 'external' / 'val' / 'images')


model_str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_str))

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.SOLVER.MAX_ITER = 8800  # 50 epochs, considering batch size of 16 and 2816 images
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

cfg.OUTPUT_DIR = os.path.join(HOME_FOLDER,'output')


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'),'wt') as f:
          f.write(cfg.dump())

#cfg.INPUT.MAX_SIZE_TEST = 800
#cfg.INPUT.MAX_SIZE_TRAIN = 800
#cfg.INPUT.MIN_SIZE_TEST = 600
#cfg.INPUT.MIN_SIZE_TRAIN = 600
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Check for existing checkpoints
checkpoints = [ckpt for ckpt in os.listdir(cfg.OUTPUT_DIR) if ckpt.endswith(".pth")]
checkpoints.sort()

if checkpoints:
    # If there are checkpoints available, resume from the latest checkpoint
    last_checkpoint = checkpoints[-1]
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, last_checkpoint)
    print(f"Resuming training from {cfg.MODEL.WEIGHTS}")
    resume_flag = True
else:
    # If no checkpoints are found, start training from the pre-trained model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_str)
    print(f"Starting training from scratch with weights from {cfg.MODEL.WEIGHTS}")
    resume_flag = False
    
    
class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Define a sequence of augmentations:
        augs = [
            T.ResizeShortestEdge(800),
            T.RandomBrightness(0.5, 2),
            T.RandomContrast(0.5, 2),
            T.RandomSaturation(0.5, 2),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomRotation([0,90,180,270], expand=True, center=None, sample_style='choice', interp=None)
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper, num_workers=0)


setup_logger(output=cfg.OUTPUT_DIR)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=resume_flag)
trainer.train()