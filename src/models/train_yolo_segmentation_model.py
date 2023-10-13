import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import wandb
from ultralytics import YOLO

def train():
    
    # Write command line outputs to training log file
    sys.stdout = open(os.path.join(os.getenv('HOME_FOLDER'), 'reports', 'yolo_seg_training_log.txt'), 'w')
    # Load environment variables from .env file
    load_dotenv(find_dotenv())

    DATA_FOLDER = os.getenv('DATA_FOLDER')
    HOME_FOLDER = os.getenv('HOME_FOLDER')

    # Initialize W&B from environment variables
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_ENTITY = os.getenv('WANDB_ENTITY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    if not all([WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT]):
        raise ValueError("Please set WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT environment variables.")

    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, resume="allow")

    # Paths
    YOLO_CONFIG = 'yolov8n-seg.yaml'
    CHECKPOINT_DIR = Path(HOME_FOLDER) / 'models' / 'yolo_seg' / 'checkpoints'
    LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'last.pt')

    # Load YOLO model
    if os.path.exists(LAST_CHECKPOINT_PATH):
        print("Resuming training from last checkpoint.")
        model = YOLO(YOLO_CONFIG).load(LAST_CHECKPOINT_PATH)  # Resume from saved weights
    else:
        print("Starting training from scratch.")
        model = YOLO(YOLO_CONFIG)  # Start from pre-trained weights

    # Configure training parameters
    epochs = 20
    imgsz = 224
    batch_size = 64
    data_config = Path(HOME_FOLDER) / 'models' / 'yolo_seg' / 'cardd.yaml'

    # Train the model
    results = model.train(data=data_config, epochs=epochs, imgsz=imgsz, weights=CHECKPOINT_DIR, resume=True, batch_size=batch_size)

    # Finish W&B run
    run.finish()

if __name__ == '__main__':
    train()
