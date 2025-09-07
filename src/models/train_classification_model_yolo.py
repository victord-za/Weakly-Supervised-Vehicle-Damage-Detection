import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import wandb
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, multilabel_confusion_matrix, classification_report
import yaml
from PIL import Image
import cv2

def train():
    # Timer and timeout setup
    start_time = time.time()
    timeout_flag = False
    TIMEOUT_THRESHOLD = 9.25 * 60 * 60  # 9.25 hours in seconds
    PREDICTION_THRESHOLD = 0.5
    
    # Load environment variables from .env file
    load_dotenv(find_dotenv())
    DATA_FOLDER = os.getenv('DATA_FOLDER')
    HOME_FOLDER = os.getenv('HOME_FOLDER')
    
    # Redirect stdout to log file 
    log_path = os.path.join(HOME_FOLDER, 'reports', 'yolo_class_training_log.txt')
    sys.stdout = open(log_path, 'w')
    
    print("Starting YOLO Multi-Label Classification Training")
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Home folder: {HOME_FOLDER}")
    
    # Set up weights and biases for logging
    os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')
    os.environ['WANDB_ENTITY'] = os.getenv('WANDB_ENTITY')
    os.environ['WANDB_PROJECT'] = os.getenv('WANDB_PROJECT')
    os.environ["WANDB_MODE"] = "offline"
    
    if not all([os.getenv('WANDB_API_KEY'), os.getenv('WANDB_ENTITY'), os.getenv('WANDB_PROJECT')]):
        raise ValueError("Please set WANDB_API_KEY, WANDB_ENTITY, and WANDB_PROJECT environment variables.")
    
    wandb.init(project=os.environ['WANDB_PROJECT'], 
               entity=os.environ['WANDB_ENTITY'], 
               resume="allow",
               name="yolo_multilabel_damage_classification")
    
    # Paths and configuration
    YOLO_CONFIG = 'yolov8n-cls.pt' 
    CHECKPOINT_DIR = Path(HOME_FOLDER) / 'models' / 'yolo_class' / 'checkpoints'
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    BEST_MODEL_PATH = CHECKPOINT_DIR / 'best.pt'
    LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / 'last.pt'
    
    # Data configuration - should point to your generated YOLO labels
    data_config = Path(DATA_FOLDER) / 'data.yaml'
    
    # Verify data config exists
    if not data_config.exists():
        raise FileNotFoundError(f"Data config not found: {data_config}")
    
    # Load class names from data config
    with open(data_config, 'r') as f:
        config = yaml.safe_load(f)
    class_names = list(config['names'].values())
    num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Training parameters
    epochs = 30
    imgsz = 224 
    batch_size = 32 
    patience = 10 
    
    # Load or initialize YOLO model
    if os.path.exists(LAST_CHECKPOINT_PATH):
        print(f"Resuming training from: {LAST_CHECKPOINT_PATH}")
        model = YOLO(LAST_CHECKPOINT_PATH)
    else:
        print("Starting training from pre-trained weights")
        model = YOLO(YOLO_CONFIG)
    
    # Configure model for multi-label
    print("Configuring model for multi-label classification")
    
    # Training arguments
    train_args = {
        'data': str(data_config),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'patience': patience,
        'save_period': 1,  # Save checkpoint every epoch
        'project': str(CHECKPOINT_DIR.parent),
        'name': 'training_run',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'Adam', 
        'lr0': 0.001, 
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'cos_lr': True,  # Cosine learning rate 
        'multi_scale': True,  # Data augmentation
        'mixup': 0.1,  # Additional augmentation
        'copy_paste': 0.1,
        'degrees': 15.0,  # rotation
        'translate': 0.1,  # translation
        'scale': 0.1,  # scale
        'hsv_h': 0.05,  # hue
        'hsv_s': 0.1,   # saturation
        'hsv_v': 0.2,   # brightness
        'flipud': 0.0,
        'fliplr': 0.5,  # horizontal flip
        'mosaic': 0.0,  # Disable mosaic
        'verbose': True,
        'save': True,
        'save_txt': True,
        'plots': True
    }
    
    # Log hyperparameters to wandb
    wandb.config.update({
        'model': 'YOLOv8-cls',
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': train_args['lr0'],
        'optimizer': train_args['optimizer'],
        'patience': patience,
        'num_classes': num_classes,
        'image_size': imgsz,
        'prediction_threshold': PREDICTION_THRESHOLD
    })
    
    print("Starting training with the following parameters:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    try:
        # Train the model
        print("Initiating training...")
        results = model.train(**train_args)
        
        print("Training completed successfully!")
        print(f"Results: {results}")
        
        # Load best model for evaluation
        if BEST_MODEL_PATH.exists():
            print("Loading best model for evaluation")
            best_model = YOLO(BEST_MODEL_PATH)
        else:
            print("Using final model for evaluation")
            best_model = model
        
        # Evaluation on test set
        print("Starting test set evaluation...")
        test_results = evaluate_model(best_model, data_config, class_names, HOME_FOLDER, PREDICTION_THRESHOLD)
        
        # Log test results to wandb
        wandb.log({
            "Test_Loss": test_results.get('test_loss', 0),
            "Test_Accuracy": test_results.get('test_accuracy', 0),
            "Test_Macro_F1": test_results.get('macro_f1', 0),
            "Test_Micro_F1": test_results.get('micro_f1', 0),
            "Test_Weighted_F1": test_results.get('weighted_f1', 0)
        })
        
        # Log per-class metrics
        if 'per_class_metrics' in test_results:
            for i, class_name in enumerate(class_names):
                wandb.log({
                    f"Test_Precision_{class_name}": test_results['per_class_metrics']['precision'][i],
                    f"Test_Recall_{class_name}": test_results['per_class_metrics']['recall'][i],
                    f"Test_F1_{class_name}": test_results['per_class_metrics']['f1'][i]
                })
        
        # Generate saliency maps
        print("Generating saliency maps...")
        generate_saliency_maps(best_model, data_config, class_names, HOME_FOLDER)
        
        print("Training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        timeout_flag = True
        
    finally:
        wandb.finish()
        sys.stdout.close()
    
    return not timeout_flag

def evaluate_model(model, data_config, class_names, home_folder, threshold=0.5):
    """Evaluate model on test set"""
    
    # Load test data paths from config
    with open(data_config, 'r') as f:
        config = yaml.safe_load(f)
    
    test_file = config.get('test', config.get('val'))  # Fallback to val if no test
    if not test_file:
        print("No test set specified in config")
        return {}
    
    # Read test image paths
    with open(test_file, 'r') as f:
        test_image_paths = [line.strip() for line in f.readlines()]
    
    print(f"Evaluating on {len(test_image_paths)} test images")
    
    all_labels = []
    all_predictions = []
    
    # Process test images
    for img_path in test_image_paths:
        if not os.path.exists(img_path):
            continue
            
        # Get corresponding label file
        img_name = Path(img_path).stem
        label_path = Path(data_config).parent / 'labels' / f'{img_name}.txt'
        
        if not label_path.exists():
            continue
        
        # Load ground truth labels
        with open(label_path, 'r') as f:
            gt_classes = [int(line.strip()) for line in f.readlines()]
        
        # Create multi-label ground truth vector
        gt_vector = np.zeros(len(class_names))
        for class_id in gt_classes:
            if class_id < len(class_names):
                gt_vector[class_id] = 1
        
        # Get model predictions
        results = model.predict(img_path, verbose=False)
        
        if results and len(results) > 0:
            # Extract prediction probabilities
            probs = results[0].probs.data.cpu().numpy() if hasattr(results[0], 'probs') else np.zeros(len(class_names))
            pred_vector = (probs > threshold).astype(int)
        else:
            pred_vector = np.zeros(len(class_names))
        
        all_labels.append(gt_vector)
        all_predictions.append(pred_vector)
    
    if not all_labels:
        print("No valid test samples found")
        return {}
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Calculate metrics
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    accuracy = np.mean(all_predictions == all_labels)
    
    # Per-class metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    # Generate confusion matrices
    conf_matrices = multilabel_confusion_matrix(all_labels, all_predictions)
    
    # Save confusion matrix plots
    results_dir = Path(home_folder) / 'reports' / 'figures' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (conf_matrix, class_name) in enumerate(zip(conf_matrices, class_names)):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix: {class_name}')
        plt.savefig(results_dir / f'yolo_confusion_matrix_{class_name}.png')
        plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    return {
        'test_accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'per_class_metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1_per_class,
            'support': support
        }
    }

def generate_saliency_maps(model, data_config, class_names, home_folder):
    """Generate GradCAM saliency maps for YOLO"""
    
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from torchvision import transforms
    import torch
    
    # Load test data
    with open(data_config, 'r') as f:
        config = yaml.safe_load(f)
    
    test_file = config.get('test', config.get('val'))
    if not test_file:
        return
    
    # Read a sample of test images
    with open(test_file, 'r') as f:
        test_image_paths = [line.strip() for line in f.readlines()]
    
    # Create saliency output directory
    saliency_dir = Path(home_folder) / 'reports' / 'figures' / 'saliency' / 'yolo_test'
    saliency_dir.mkdir(parents=True, exist_ok=True)
    
    # Get YOLO's backbone model for GradCAM
    yolo_model = model.model
    
    # Find the last convolutional layer in YOLO backbone
    target_layers = []
    
    # For YOLOv8 classification, find the last conv layer before classifier
    for name, module in yolo_model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.modules.conv.Conv2d)):
            target_layers = [module]  # Keep updating to get the last one
    
    if not target_layers:
        print("No suitable convolutional layers found for GradCAM")
        return
    
    print(f"Using target layer: {target_layers[0]}")
    
    # Initialize GradCAM
    cam = GradCAM(model=yolo_model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process sample images
    sample_images = test_image_paths[:50]
    print(f"Generating GradCAM saliency maps for {len(sample_images)} sample images")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for img_path in sample_images:
        if not os.path.exists(img_path):
            continue
            
        img_name = Path(img_path).stem
        
        # Create directory for this image
        img_saliency_dir = saliency_dir / img_name
        img_saliency_dir.mkdir(exist_ok=True)
        
        try:
            # Load and preprocess image
            original_img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(original_img).unsqueeze(0).to(device)
            
            # Convert to numpy for visualization
            original_img_np = np.array(original_img.resize((224, 224))) / 255.0
            
            # Generate GradCAM for each class
            for class_idx, class_name in enumerate(class_names):
                
                # Generate GradCAM
                grayscale_cam = cam(input_tensor=img_tensor, target_category=class_idx)
                
                if len(grayscale_cam) > 0:
                    # Create visualization
                    visualization = show_cam_on_image(original_img_np, grayscale_cam[0], use_rgb=True)
                    
                    # Save visualization
                    plt.figure(figsize=(10, 5))
                    
                    plt.subplot(1, 2, 1)
                    plt.imshow(original_img_np)
                    plt.title('Original Image')
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(visualization)
                    plt.title(f'GradCAM: {class_name}')
                    plt.axis('off')
                    
                    plt.savefig(img_saliency_dir / f'{class_name}_gradcam.jpg', 
                              bbox_inches='tight', dpi=100)
                    plt.close()
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print("GradCAM saliency maps generated for YOLO model")

if __name__ == '__main__':
    success = train()
    if success:
        print("Training completed successfully!")
    else:
        print("Training was interrupted or failed.")