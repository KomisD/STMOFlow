import argparse
from ultralytics import YOLO
from multiprocessing import freeze_support
import os

def get_model(stage: int, save_folder: str):
    """
    Initialize the model based on the training stage.
    
    Stage 1:
        Initialize using the YAML file that defines the model architecture.
    Stage 2:
        Initialize using the best weights from stage 1.
    Stage 3:
        Initialize using the best weights from stage 2.
    
    Args:
        stage (int): The current training stage.
        save_folder (str): The base folder where models are saved.
    
    Returns:
        YOLO model instance.
    """
    if stage == 1:
        # Use the architecture file for stage 1.
        arch_path = "C:\\Users\\Dcube_User\\Desktop\\Codes\\New-instance\\yolov8s-Second_stream.yaml"
        print(f"Initializing Stage 1 model from architecture: {arch_path}")
        model = YOLO(arch_path)
    elif stage == 2:
        # Load weights from stage 1.
        weight_path = os.path.join(save_folder, "stage1_augmented_640", "weights", "best.pt")
        print(f"Initializing Stage 2 model from weights: {weight_path}")
        model = YOLO(weight_path)
    elif stage == 3:
        # Load weights from stage 2.
        weight_path = os.path.join(save_folder, "stage2_finetune_640", "weights", "best.pt")
        print(f"Initializing Stage 3 model from weights: {weight_path}")
        model = YOLO(weight_path)
    else:
        raise ValueError("Invalid stage provided. Choose stage 1, 2, or 3.")
    return model

def train_stage(stage: int, config: dict, save_folder: str):
    """
    Train the model for a given stage using its specific configuration.
    
    Args:
        stage (int): The stage number (1, 2, or 3).
        config (dict): Dictionary of training parameters.
        save_folder (str): Base folder where training outputs are saved.
    
    Returns:
        Training results.
    """
    print(f"Starting Stage {stage} training...")
    model = get_model(stage, save_folder)
    results = model.train(**config)
    print(f"Completed Stage {stage} training.")
    return results

def main(args):
    freeze_support()
    # Define the base folder to save training outputs.
    save_folder = 'Experiment1/'
    os.makedirs(save_folder, exist_ok=True)
    print(f"Saving training outputs to: {save_folder}")
    
    # Define configuration dictionaries for each stage.
    stage_configs = {
        1: {
            "data": "D:\\Datasets\\TestedDatasets\\DETECTION\\DroneVsBird_track\\data.yaml",  # Augmented dataset YAML
            "project": save_folder,
            "name": "stage1_augmented_640",
            "epochs": 80,
            "imgsz": 640,
            "close_mosaic": 0,   # Full mosaic augmentation enabled
            "optimizer": 'auto',
            "amp": True,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "batch": 16,

            "hsv_v":0.4,  
            "hsv_h":0.01,
            "hsv_s":0.4,

            "copy_paste": 0.5, 
            "degrees":10, 
            "translate":0.1, 
            "scale":0.5,
            "shear":5,
            "perspective":0.0005, 
            "fliplr":0.5,
            "mosaic":1.0,
            "erasing":0.2,
            "crop_fraction":0.2,
        },
        2: {
            "data": "D:\\Datasets\\TestedDatasets\\DETECTION\\DroneVsBird_track\\data.yaml",  # Original dataset YAML
            "project": save_folder,
            "name": "stage2_finetune_640",
            "epochs": 50,
            "imgsz": 640,
            "close_mosaic": 5,   # Reduced mosaic effect
            "optimizer": 'auto',
            "amp": True,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "batch": 16,
            
            "hsv_v":0.4,  
            "hsv_h":0.01,
            "hsv_s":0.4,
        },
        3: {
            "data": "D:\\Datasets\\TestedDatasets\\DETECTION\\DroneVsBird_track\\data.yaml",  # Original dataset YAML
            "project": save_folder,
            "name": "stage3_finetune_1080",
            "epochs": 10,
            "imgsz": 1080,
            "close_mosaic": 10,  # High value to disable mosaic augmentation
            "optimizer": 'auto',
            "amp": True,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "batch": 16,
        }
    }
    
    if args.stage:
        # Run a specific stage.
        if args.stage in stage_configs:
            train_stage(args.stage, stage_configs[args.stage], save_folder)
        else:
            print(f"Stage {args.stage} configuration not found.")
    else:
        # Run all stages sequentially.
        for stage in sorted(stage_configs.keys()):
            train_stage(stage, stage_configs[stage], save_folder)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv8 model in multi-stage fashion.")
    parser.add_argument('--stage', type=int, choices=[1, 2, 3],
                        help="Specify the training stage to run. If omitted, all stages run sequentially.")
    args = parser.parse_args()
    main(args)
