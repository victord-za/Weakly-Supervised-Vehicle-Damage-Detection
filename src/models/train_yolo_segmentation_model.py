import os
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import wandb

# Fix Windows CUDA memory issues
if os.name == 'nt':  # Windows
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from ultralytics import YOLO
import yaml
from datetime import datetime
import numpy as np


def validate_setup():
    """Basic setup validation"""
    load_dotenv(find_dotenv())
    HOME_FOLDER = os.getenv('HOME_FOLDER')
    DATA_FOLDER = os.getenv('DATA_FOLDER')
    
    if not HOME_FOLDER:
        raise ValueError("HOME_FOLDER environment variable not set")
    
    required_dirs = [
        Path(HOME_FOLDER) / 'models' / 'yolo11_seg',
        Path(HOME_FOLDER) / 'data' / 'external',
        Path(HOME_FOLDER) / 'reports',
        Path(DATA_FOLDER) / 'external'
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Setup validation completed")
    return HOME_FOLDER, DATA_FOLDER


def setup_logging_and_wandb(HOME_FOLDER):
    """Setup logging and Weights & Biases"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(HOME_FOLDER, 'reports', f'yolo11_seg_training_log_{timestamp}.txt')
    
    sys.stdout = open(log_path, 'w')
    
    print(f"Training started at: {datetime.now()}")
    print(f"Operating System: {os.name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            torch.cuda.empty_cache()
            print("GPU cache cleared successfully")
        except Exception as e:
            print(f"Warning: GPU detection failed: {e}")
            print("Falling back to CPU training")
            torch.cuda.is_available = lambda: False
    else:
        print("CUDA not available - using CPU training")

    # Initialize W&B
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_ENTITY = os.getenv('WANDB_ENTITY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')
    
    wandb_run = None
    if all([WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT]):
        try:
            os.environ['WANDB_API_KEY'] = WANDB_API_KEY
            wandb_run = wandb.init(
                project=WANDB_PROJECT, 
                entity=WANDB_ENTITY, 
                name=f"yolo11_seg_improved_{timestamp}",
                tags=["vehicle_damage", "segmentation", "yolo11", "cardd", "balanced_training"],
                resume="allow"
            )
            print("W&B initialized successfully")
        except Exception as e:
            print(f"W&B initialization failed: {e}")
            wandb_run = None
    else:
        print("W&B not initialized - missing environment variables")
    
    return timestamp, wandb_run


def validate_dataset(DATA_FOLDER):
    """Validate dataset configuration and files"""
    data_config = Path(DATA_FOLDER) / 'external' / 'cardd.yaml'
    
    if not data_config.exists():
        raise FileNotFoundError(f"Data configuration file not found: {data_config}")
    
    with open(data_config, 'r') as f:
        data_info = yaml.safe_load(f)
    
    print(f"Dataset: {data_info.get('path', 'Unknown')}")
    print(f"Classes: {data_info.get('nc', 'Unknown')} - {data_info.get('names', [])}")
    
    dataset_base = Path(data_info.get('path', ''))
    train_rel_path = data_info.get('train', 'train')
    
    if 'images' in train_rel_path:
        train_images_path = dataset_base / train_rel_path
    else:
        train_images_path = dataset_base / train_rel_path / 'images'
    
    print(f"Checking dataset at: {train_images_path}")
    
    if not train_images_path.exists():
        raise FileNotFoundError(f"Training images not found at {train_images_path}")
    
    image_files = list(train_images_path.glob('*.jpg')) + list(train_images_path.glob('*.png'))
    print(f"Found {len(image_files)} training images")
    
    if len(image_files) == 0:
        raise ValueError("No images found in training directory")
    
    labels_path = train_images_path.parent / 'labels'
    if not labels_path.exists():
        raise FileNotFoundError("Labels directory not found")
    
    label_files = list(labels_path.glob('*.txt'))
    print(f"Found {len(label_files)} label files")
    
    if len(label_files) == 0:
        raise ValueError("No label files found")
    
    if len(label_files) > 0:
        try:
            with open(label_files[0], 'r') as f:
                first_line = f.readline().strip()
                print(f"Sample label: {first_line}")
        except Exception as e:
            raise RuntimeError(f"Error reading label file: {e}")
    
    print("Dataset validation passed")
    print("="*50)
    
    return data_config, data_info


def get_model(HOME_FOLDER):
    """Initialize YOLO model with checkpoints"""
    CHECKPOINT_DIR = Path(HOME_FOLDER) / 'models' / 'yolo11_seg'
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / 'last.pt'
    
    model_configs = {
        'nano': 'yolo11n-seg.pt',
        'small': 'yolo11s-seg.pt',
        'medium': 'yolo11m-seg.pt',
        'large': 'yolo11l-seg.pt',
        'xlarge': 'yolo11x-seg.pt'
    }
    
    MODEL_SIZE = 'nano'
    YOLO_CONFIG = model_configs[MODEL_SIZE]
    
    if LAST_CHECKPOINT_PATH.exists():
        print("Resuming training from last checkpoint.")
        model = YOLO(LAST_CHECKPOINT_PATH)
    else:
        print("Starting training from fresh pre-trained YOLO11 weights.")
        print(f"Using YOLO11 model: {YOLO_CONFIG}")
        model = YOLO(YOLO_CONFIG)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on: {device}")
    
    return model, MODEL_SIZE, YOLO_CONFIG, CHECKPOINT_DIR


def setup_training_callbacks(wandb_run, class_names):
    """Setup training callbacks for monitoring with per-class metrics"""
    def on_train_epoch_end(trainer):
        """Log detailed metrics to W&B after each epoch"""
        if wandb_run and hasattr(trainer, 'metrics'):
            epoch = trainer.epoch
            metrics = trainer.metrics
            
            # Basic metrics
            wandb_metrics = {
                'epoch': epoch,
                'train/loss': getattr(trainer, 'loss', 0),
                'train/box_loss': metrics.get('train/box_loss', 0),
                'train/seg_loss': metrics.get('train/seg_loss', 0),  
                'train/cls_loss': metrics.get('train/cls_loss', 0),
                'train/dfl_loss': metrics.get('train/dfl_loss', 0),
                'val/box_loss': metrics.get('val/box_loss', 0),
                'val/seg_loss': metrics.get('val/seg_loss', 0),
                'val/cls_loss': metrics.get('val/cls_loss', 0),
                'val/dfl_loss': metrics.get('val/dfl_loss', 0),
                'lr/pg0': metrics.get('lr/pg0', 0),
                'lr/pg1': metrics.get('lr/pg1', 0),
                'lr/pg2': metrics.get('lr/pg2', 0)
            }
            
            # Segmentation metrics (primary focus)
            seg_precision = metrics.get('metrics/precision(M)', 0)
            seg_recall = metrics.get('metrics/recall(M)', 0)
            seg_map50 = metrics.get('metrics/mAP50(M)', 0)
            seg_map50_95 = metrics.get('metrics/mAP50-95(M)', 0)
            
            wandb_metrics.update({
                'seg/precision': seg_precision,
                'seg/recall': seg_recall,
                'seg/mAP50': seg_map50,
                'seg/mAP50-95': seg_map50_95,
            })
            
            # Detection metrics (secondary)
            wandb_metrics.update({
                'det/precision': metrics.get('metrics/precision(B)', 0),
                'det/recall': metrics.get('metrics/recall(B)', 0),
                'det/mAP50': metrics.get('metrics/mAP50(B)', 0),
                'det/mAP50-95': metrics.get('metrics/mAP50-95(B)', 0),
            })
            
            # Per-class metrics
            if hasattr(trainer.validator, 'metrics') and hasattr(trainer.validator.metrics, 'ap_class_index'):
                try:
                    validator_metrics = trainer.validator.metrics
                    
                    # Per-class segmentation mAP50
                    if hasattr(validator_metrics, 'ap50'):
                        ap50_values = validator_metrics.ap50
                        if len(ap50_values) > 0 and len(ap50_values) == len(class_names):
                            for i, class_name in enumerate(class_names):
                                wandb_metrics[f'class_seg_mAP50/{class_name}'] = ap50_values[i]
                    
                    # Per-class segmentation mAP50-95
                    if hasattr(validator_metrics, 'ap'):
                        ap_values = validator_metrics.ap
                        if len(ap_values) > 0 and len(ap_values) == len(class_names):
                            for i, class_name in enumerate(class_names):
                                wandb_metrics[f'class_seg_mAP50-95/{class_name}'] = ap_values[i]
                    
                    # Per-class precision and recall
                    if hasattr(validator_metrics, 'p') and hasattr(validator_metrics, 'r'):
                        precision_values = validator_metrics.p
                        recall_values = validator_metrics.r
                        
                        if len(precision_values) == len(class_names):
                            for i, class_name in enumerate(class_names):
                                wandb_metrics[f'class_seg_precision/{class_name}'] = precision_values[i]
                        
                        if len(recall_values) == len(class_names):
                            for i, class_name in enumerate(class_names):
                                wandb_metrics[f'class_seg_recall/{class_name}'] = recall_values[i]
                
                except Exception as e:
                    print(f"Warning: Could not extract per-class metrics: {e}")
            
            wandb.log(wandb_metrics)
            
            # Print progress
            print(f"Epoch {epoch}: Precision(M)={seg_precision:.4f}, "
                  f"Recall(M)={seg_recall:.4f}, "
                  f"mAP50(M)={seg_map50:.4f}")
            
            # Print per-class performance
            try:
                if hasattr(trainer.validator, 'metrics') and hasattr(trainer.validator.metrics, 'ap50'):
                    ap50_values = trainer.validator.metrics.ap50
                    if len(ap50_values) == len(class_names):
                        print("Per-class mAP50 (Segmentation):")
                        for i, class_name in enumerate(class_names):
                            print(f"  {class_name}: {ap50_values[i]:.3f}")
            except:
                pass
            
            # Stability check
            if epoch > 1:
                prev_precision = getattr(trainer, 'prev_precision', seg_precision)
                precision_change = abs(seg_precision - prev_precision)
                if precision_change > 0.1:
                    print(f"WARNING: Large precision change of {precision_change:.4f} - possible instability")
            
            trainer.prev_precision = seg_precision
    
    return on_train_epoch_end


def train(HOME_FOLDER, data_config, data_info, model, MODEL_SIZE, YOLO_CONFIG, CHECKPOINT_DIR, timestamp, wandb_run):
    """Main training function with improved parameters for segmentation"""
    # Determine device with error handling
    device = 'cpu'
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            test_tensor = torch.randn(10, 10).cuda()
            del test_tensor
            device = 'cuda'
            print("CUDA test passed - using GPU")
        except Exception as e:
            print(f"CUDA test failed: {e}")
            print("Falling back to CPU training")
            device = 'cpu'
    
    # Get class names for monitoring
    class_names = data_info.get('names', [])
    print(f"Monitoring per-class metrics for: {class_names}")
    
    # Set up callback
    callback = setup_training_callbacks(wandb_run, class_names)
    model.add_callback('on_train_epoch_end', callback)

    # Improved training parameters for segmentation with class imbalance
    training_params = {
        'data': data_config,
        'epochs': 150,  # More epochs for difficult classes
        'imgsz': 640,   # Good size for small objects
        'batch': 2,     # Small batch for memory stability
        'patience': 50, # More patience for class imbalance
        'save': True,
        'save_period': 20,
        'cache': False,
        'device': device,
        'workers': 0,
        'project': CHECKPOINT_DIR.parent,
        'name': f'yolo11_seg_improved_{timestamp}',
        
        # Optimizer settings for segmentation
        'optimizer': 'AdamW',
        'lr0': 0.0005,  # Lower learning rate for segmentation
        'lrf': 0.001,   # Very low final LR
        'weight_decay': 0.0001,
        'warmup_epochs': 20,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.0001,
        
        # Segmentation-focused augmentation
        'hsv_h': 0.01,     # Minimal hue change
        'hsv_s': 0.3,      # Moderate saturation
        'hsv_v': 0.2,      # Moderate brightness
        'degrees': 5.0,    # Less rotation
        'translate': 0.05, # Minimal translation
        'scale': 0.7,      # Less aggressive scaling
        'shear': 1.0,      # Minimal shear
        'perspective': 0.0, # No perspective
        'flipud': 0.0,     # No vertical flip
        'fliplr': 0.5,     # Horizontal flip OK
        'mosaic': 0.5,     # Reduced mosaic
        'mixup': 0.0,      # No mixup - bad for segmentation
        'copy_paste': 0.5, # Good for minority classes
        'erasing': 0.05,   # Minimal erasing
        
        'val': True,
        'plots': True,
        'verbose': True,
    }
    
    # Log hyperparameters to W&B
    if wandb_run:
        try:
            wandb.config.update({
                "model_size": MODEL_SIZE,
                "model_config": YOLO_CONFIG,
                "dataset": data_info.get('path', 'Unknown'),
                "num_classes": data_info.get('nc', 'Unknown'),
                "class_names": class_names,
                "training_type": "segmentation_optimized",
                "device": device,
                **training_params
            })
        except Exception as e:
            print(f"Warning: Failed to update W&B config: {e}")

    print("Starting segmentation-optimized training...")
    print(f"Device: {device.upper()}")
    print(f"Image size: {training_params['imgsz']}")
    print(f"Batch size: {training_params['batch']}")
    print(f"Learning rate: {training_params['lr0']}")
    print(f"Epochs: {training_params['epochs']}")
    print("Segmentation-focused augmentation enabled")
    print("="*50)
    
    try:
        results = model.train(**training_params)
        
        print("Training completed!")
        print(f"Best weights: {model.trainer.best}")
        print(f"Last weights: {model.trainer.last}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    # Extract final metrics
    final_metrics = results.results_dict if hasattr(results, 'results_dict') else {}
    
    print("Final metrics:")
    for key, value in final_metrics.items():
        is_nan = np.isnan(value) if isinstance(value, (int, float)) else False
        print(f"  {key}: {value} {'(NaN!)' if is_nan else ''}")
    
    # Log final results to W&B
    if wandb_run:
        try:
            final_wandb_metrics = {
                'final/mAP50_M': final_metrics.get('metrics/mAP50(M)', 0),
                'final/mAP50-95_M': final_metrics.get('metrics/mAP50-95(M)', 0),
                'final/precision_M': final_metrics.get('metrics/precision(M)', 0),
                'final/recall_M': final_metrics.get('metrics/recall(M)', 0),
                'final/mAP50_B': final_metrics.get('metrics/mAP50(B)', 0),
                'final/mAP50-95_B': final_metrics.get('metrics/mAP50-95(B)', 0),
                'final/precision_B': final_metrics.get('metrics/precision(B)', 0),
                'final/recall_B': final_metrics.get('metrics/recall(B)', 0),
            }
            
            wandb.log(final_wandb_metrics)
            
            # Log artifacts
            wandb.log_artifact(str(model.trainer.best), name=f"best_model_improved_{timestamp}", type="model")
            
            # Log plots
            plots_dir = Path(model.trainer.save_dir)
            for plot_file in plots_dir.glob("*.png"):
                wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
                
        except Exception as e:
            print(f"Warning: Failed to log to W&B: {e}")
    
    # Save summary
    summary_path = CHECKPOINT_DIR / f'training_summary_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write(f"SEGMENTATION TRAINING COMPLETED: {datetime.now()}\n")
        f.write(f"Model: {YOLO_CONFIG}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Image size: {training_params['imgsz']}\n")
        f.write(f"Learning rate: {training_params['lr0']}\n")
        f.write(f"Epochs: {training_params['epochs']}\n")
        f.write(f"Batch size: {training_params['batch']}\n")
        f.write(f"Classes monitored: {class_names}\n")
        f.write("\nFinal metrics:\n")
        for key, value in final_metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Summary saved: {summary_path}")
    
    return model, final_metrics


def evaluate_test(HOME_FOLDER, DATA_FOLDER, model, data_info, timestamp, wandb_run):
    """Evaluate the trained model on test set"""
    print("="*50)
    print("Starting test set evaluation...")
    
    test_path = data_info.get('test', None)
    if not test_path:
        print("No test set defined in data configuration. Skipping test evaluation.")
        return
    
    dataset_base = Path(data_info.get('path', ''))
    if 'images' in test_path:
        test_images_path = dataset_base / test_path
    else:
        test_images_path = dataset_base / test_path / 'images'
    
    if not test_images_path.exists():
        print(f"Test images not found at {test_images_path}. Skipping test evaluation.")
        return
    
    test_image_files = list(test_images_path.glob('*.jpg')) + list(test_images_path.glob('*.png'))
    print(f"Found {len(test_image_files)} test images")
    
    if len(test_image_files) == 0:
        print("No test images found. Skipping test evaluation.")
        return
    
    # Create evaluation output directory
    CHECKPOINT_DIR = Path(HOME_FOLDER) / 'models' / 'yolo11_seg'
    eval_dir = CHECKPOINT_DIR / f'test_evaluation_{timestamp}'
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = eval_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Running model validation on test set...")
        
        val_batch_size = 1 if not torch.cuda.is_available() else 4
        print(f"Using validation batch size: {val_batch_size}")
        
        test_results = model.val(
            data=Path(DATA_FOLDER) / 'external' / 'cardd.yaml',
            split='test',
            batch=val_batch_size,
            save_json=True,
            save_hybrid=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
            half=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dnn=False,
            plots=True,
            verbose=True,
            project=str(eval_dir),
            name='validation_results',
            workers=0
        )
        
        test_metrics = test_results.results_dict if hasattr(test_results, 'results_dict') else {}
        
        print("\n" + "="*60)
        print("TEST SET EVALUATION RESULTS")
        print("="*60)
        
        print("\nDETECTION METRICS (Bounding Boxes):")
        detection_metrics = {
            'Precision': test_metrics.get('metrics/precision(B)', 0),
            'Recall': test_metrics.get('metrics/recall(B)', 0),
            'mAP@0.5': test_metrics.get('metrics/mAP50(B)', 0),
            'mAP@0.5:0.95': test_metrics.get('metrics/mAP50-95(B)', 0)
        }
        for metric, value in detection_metrics.items():
            is_nan = np.isnan(value) if isinstance(value, (int, float)) else False
            status = " (NaN!)" if is_nan else " [OK]"
            print(f"  {metric:15}: {value:.4f}{status}")
        
        print("\nSEGMENTATION METRICS (Masks):")
        segmentation_metrics = {
            'Precision': test_metrics.get('metrics/precision(M)', 0),
            'Recall': test_metrics.get('metrics/recall(M)', 0),
            'mAP@0.5': test_metrics.get('metrics/mAP50(M)', 0),
            'mAP@0.5:0.95': test_metrics.get('metrics/mAP50-95(M)', 0)
        }
        for metric, value in segmentation_metrics.items():
            is_nan = np.isnan(value) if isinstance(value, (int, float)) else False
            status = " (NaN!)" if is_nan else " [OK]"
            print(f"  {metric:15}: {value:.4f}{status}")
        
        print("\nALL RAW METRICS:")
        for key, value in test_metrics.items():
            is_nan = np.isnan(value) if isinstance(value, (int, float)) else False
            status = " (NaN!)" if is_nan else ""
            print(f"  {key}: {value}{status}")
        
        # Generate predictions on sample test images
        print(f"\nGenerating predictions on {min(50, len(test_image_files))} test images...")
        sample_images = test_image_files[:50]
        
        prediction_summary = []
        for i, img_path in enumerate(sample_images):
            try:
                results = model.predict(
                    source=str(img_path),
                    conf=0.25,
                    iou=0.45,
                    save=True,
                    save_txt=True,
                    save_conf=True,
                    project=str(predictions_dir),
                    name=f'image_{i:04d}',
                    exist_ok=True
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    num_detections = len(result.boxes) if result.boxes is not None else 0
                    num_masks = len(result.masks) if result.masks is not None else 0
                    
                    prediction_summary.append({
                        'image': img_path.name,
                        'detections': num_detections,
                        'masks': num_masks,
                        'confidence_scores': result.boxes.conf.tolist() if result.boxes is not None and len(result.boxes) > 0 else []
                    })
                    
                    if i % 10 == 0:
                        print(f"  Processed {i+1}/{len(sample_images)} images...")
                        
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                continue
        
        print(f"Saved predictions to: {predictions_dir}")
        
        # Log test results to W&B
        if wandb_run:
            test_wandb_metrics = {
                'test/mAP50_M': test_metrics.get('metrics/mAP50(M)', 0),
                'test/mAP50-95_M': test_metrics.get('metrics/mAP50-95(M)', 0),
                'test/precision_M': test_metrics.get('metrics/precision(M)', 0),
                'test/recall_M': test_metrics.get('metrics/recall(M)', 0),
                'test/mAP50_B': test_metrics.get('metrics/mAP50(B)', 0),
                'test/mAP50-95_B': test_metrics.get('metrics/mAP50-95(B)', 0),
                'test/precision_B': test_metrics.get('metrics/precision(B)', 0),
                'test/recall_B': test_metrics.get('metrics/recall(B)', 0),
                'test/num_images': len(test_image_files),
                'test/avg_detections_per_image': np.mean([p['detections'] for p in prediction_summary]) if prediction_summary else 0
            }
            wandb.log(test_wandb_metrics)
            
            # Log prediction summary
            if prediction_summary:
                wandb.log({"test_predictions_summary": wandb.Table(
                    columns=["image", "detections", "masks", "max_confidence"],
                    data=[[p['image'], p['detections'], p['masks'], 
                          max(p['confidence_scores']) if p['confidence_scores'] else 0] 
                         for p in prediction_summary[:20]]
                )})
        
        # Save test results summary
        test_summary_path = eval_dir / 'test_summary.txt'
        with open(test_summary_path, 'w') as f:
            f.write(f"TEST SET EVALUATION: {datetime.now()}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Dataset: {data_info.get('path', 'Unknown')}\n")
            f.write(f"Number of test images: {len(test_image_files)}\n")
            f.write(f"Classes: {data_info.get('nc', 'Unknown')} - {data_info.get('names', [])}\n\n")
            
            f.write("DETECTION METRICS (Bounding Boxes):\n")
            for metric, value in detection_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\nSEGMENTATION METRICS (Masks):\n")
            for metric, value in segmentation_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\nALL RAW METRICS:\n")
            for key, value in test_metrics.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nPREDICTION SUMMARY ({len(prediction_summary)} images processed):\n")
            for p in prediction_summary[:10]:
                f.write(f"  {p['image']}: {p['detections']} detections, {p['masks']} masks\n")
            
            f.write(f"\nOutput files saved to: {eval_dir}\n")
            f.write(f"Prediction images saved to: {predictions_dir}\n")
        
        print(f"Test summary saved: {test_summary_path}")
        print("="*60)
        
        return test_metrics, prediction_summary
        
    except Exception as e:
        print(f"Error during test evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def cleanup(wandb_run):
    """Cleanup resources"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if hasattr(sys.stdout, 'close') and sys.stdout != sys.__stdout__:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    
    if wandb_run:
        wandb.finish()


def main():
    """Main execution function"""
    try:
        print("Step 1: Validating setup...")
        HOME_FOLDER, DATA_FOLDER = validate_setup()
        
        print("\nStep 2: Setting up logging and W&B...")
        timestamp, wandb_run = setup_logging_and_wandb(HOME_FOLDER)
        
        print("\nStep 3: Validating dataset...")
        data_config, data_info = validate_dataset(DATA_FOLDER)
        
        print("\nStep 4: Initializing model...")
        model, MODEL_SIZE, YOLO_CONFIG, CHECKPOINT_DIR = get_model(HOME_FOLDER)
        
        print("\nStep 5: Training model...")
        trained_model, final_metrics = train(
            HOME_FOLDER, data_config, data_info, model, 
            MODEL_SIZE, YOLO_CONFIG, CHECKPOINT_DIR, timestamp, wandb_run
        )
        
        print("\nStep 6: Evaluating on test set...")
        test_metrics, prediction_summary = evaluate_test(HOME_FOLDER, DATA_FOLDER, trained_model, data_info, timestamp, wandb_run)
        
        print("\n" + "="*50)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        if test_metrics:
            print(f"Test evaluation completed with {len(prediction_summary) if prediction_summary else 0} predictions saved")
        print("="*50)
        
        return trained_model, final_metrics, test_metrics, prediction_summary
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        raise
    finally:
        print("\nStep 7: Cleaning up...")
        cleanup(wandb_run if 'wandb_run' in locals() else None)


if __name__ == '__main__':
    main()