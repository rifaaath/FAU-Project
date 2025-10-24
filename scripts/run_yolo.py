import argparse
from ultralytics import YOLO
from pathlib import Path

def train(args):
    """Function to handle the training process."""
    # (This function does not need to change)
    print("Starting YOLOv8 Training ")
    model = YOLO(args.model_start)
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project_name,
        name='train'
    )
    print("\nTraining Complete ")
    best_model_path = Path(args.project_name) / 'train' / 'weights' / 'best.pt'
    print(f"Best model saved to: {best_model_path}")

import json

def predict(args):
    """Function to handle the prediction process."""
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Model weights not found at '{weights_path}'")
        return

    print("Starting YOLOv8 Prediction")
    model = YOLO(weights_path)

    results = model.predict(
        source=args.source_dir,
        save=True,           # Save images with bounding boxes
        conf=args.conf,
        project=args.project_name,
        name='predict_visuals'
    )

    print("\nPrediction Complete ")

    # Custom JSON export
    output_dir = Path(args.project_name) / 'predict_visuals'
    predictions_json_path = output_dir / 'predictions.json'

    all_preds = []
    for result in results:
        image_path = Path(result.path).name  # Only use filename
        for box in result.boxes:
            pred = {
                "image": image_path,
                "class_id": int(box.cls.item()),
                "confidence": float(box.conf.item()),
                "bbox_xywh": [float(x) for x in box.xywh[0].tolist()]
            }
            all_preds.append(pred)

    with open(predictions_json_path, 'w') as f:
        json.dump(all_preds, f, indent=2)

    print(f"Predictions JSON saved to: {predictions_json_path}")
    print(f"Visualized images saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A unified script to train and run inference with YOLOv8.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select mode: "train" or "predict"')

    # (train_parser does not need to change)
    train_parser = subparsers.add_parser('train', help='Train a new YOLOv8 model.')
    train_parser.add_argument('--data_yaml', type=str, default='datasets/yolo_glyphs.yaml', help='Path to the dataset .yaml file.')
    train_parser.add_argument('--model_start', type=str, default='yolov8s.pt', help='Starting model weights.')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size.')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size.')
    train_parser.add_argument('--project_name', type=str, default='yolo_runs', help='Directory to save runs.')
    train_parser.set_defaults(func=train)

    # PARSER FOR PREDICTION (UPDATED) 
    predict_parser = subparsers.add_parser('predict', help='Run inference with a trained YOLOv8 model.')
    predict_parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights.')
    predict_parser.add_argument('--source_dir', type=str, default='datasets/yolo_glyphs/images/train/', help='Directory of images to predict on.')
    predict_parser.add_argument('--project_name', type=str, default='yolo_runs', help='Directory to save runs.')
    # New argument for confidence threshold 
    predict_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for predictions.')
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)