import argparse
from ultralytics import YOLO
from multiprocessing import freeze_support

def evaluate_model(checkpoint, data, split, imgsz, conf, iou, device):
    """
    Evaluate a YOLOv8 model using the specified settings.

    Args:
        checkpoint (str): Path to the model checkpoint (or YAML for architecture).
        data (str): Path to the dataset YAML file.
        split (str): Dataset split to evaluate (e.g., 'val').
        imgsz (int): Image size for evaluation.
        conf (float): Confidence threshold.
        iou (float): IoU threshold.
        device (str): Device to use for evaluation (e.g., '0' for GPU, 'cpu').
    """
    print(f"Loading model from {checkpoint}")
    model = YOLO(checkpoint)
    print("Starting evaluation...")
    results = model.val(
        data=data,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device
    )
    print("Evaluation complete.")
    return results

def main(args):
    freeze_support()
    evaluate_model(
        checkpoint=args.checkpoint,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 model on a specified dataset."
    )
    parser.add_argument(
        '--checkpoint', type=str, default='\\model_weights\\best.pt',
        help="Path to the model checkpoint (.pt) or YAML file."
    )
    parser.add_argument(
        '--data', type=str, default="D:\\Datasets\\TestedDatasets\\DETECTION\\DroneVsBird_track\\data.yaml",
        help="Path to the dataset YAML file."
    )
    parser.add_argument(
        '--split', type=str, default='val',
        help="Dataset split to evaluate (e.g., 'val')."
    )
    parser.add_argument(
        '--imgsz', type=int, default=1080,
        help="Image size for evaluation."
    )
    parser.add_argument(
        '--conf', type=float, default=0.15,
        help="Confidence threshold for detections."
    )
    parser.add_argument(
        '--iou', type=float, default=0.5,
        help="IoU threshold for evaluation."
    )
    parser.add_argument(
        '--device', type=str, default='0',
        help="Device to use for evaluation ('0' for GPU, 'cpu' for CPU)."
    )
    args = parser.parse_args()
    main(args)
